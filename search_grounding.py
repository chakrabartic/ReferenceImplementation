## Demo of Grounding Engine

## Grounding on Google Search: https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-google-search?hl=en

## When using grounding with Google Search, and you receive Search suggestions in your response, you must display the Search suggestions in production and in your applications: https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-search-suggestions

import logging
import sys
import time
import re

from datetime import date
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Configure standard logging
logger = logging.getLogger("VertexGroundingEngine")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s - %(levelname)s - %(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass(frozen=True)
class GroundingConfig:
    """Immutable config params"""
    project_id: str = "chayan-genai-bb"  # Vertex AI GCP Project ID
    location: str = "us-central1"        # Vertex AI Location API Endpoint
    model_name: str = "gemini-2.0-flash-001"
    temperature: float = 0.0
    top_p: float = 0.95
    max_output_tokens: int = 8192

    # Backoff retry limits
    retry_max_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_backoff_factor: float = 2.0

class MarkdownProcessor:
    """Markdown parser layer"""

    @staticmethod
    def extract_title(markdown_text: str) -> Tuple[Optional[str], str]:
        """
        Extracts Heading 1 (# Title) and removes it from the text.
        """
        if not markdown_text:
            return None, ""

        pattern = re.compile(r'^\s*#\s+(.+)$', re.MULTILINE)
        match = pattern.search(markdown_text)
        if match:
            title = match.group(1).strip()
            modified_text = pattern.sub('', markdown_text, count=1).strip()
            return title, modified_text
        return None, markdown_text.strip()

    @staticmethod
    def extract_followup_questions(markdown_text: str) -> Tuple[Optional[List[str]], str]:
        """
        Parses bullet list under '## Followup Questions' and isolates the block
        from remaining markdown sections.
        """
        if not markdown_text:
            return None, ""

        lines = markdown_text.splitlines()
        followup_questions = []
        modified_lines = []
        in_followup_section = False

        heading_pattern = re.compile(r'^\s*##\s*Followup\s+Questions', re.IGNORECASE)
        list_item_pattern = re.compile(r'^\s*[\*\-]\s+(.+)$')
        any_heading_pattern = re.compile(r'^\s*##+\s+')

        for line in lines:
            stripped = line.strip()
            if heading_pattern.match(stripped):
                in_followup_section = True
                continue

            if in_followup_section:
                if not stripped:
                    continue
                list_match = list_item_pattern.match(line)
                if list_match:
                    followup_questions.append(list_match.group(1).strip())
                elif any_heading_pattern.match(line):
                    in_followup_section = False
                    modified_lines.append(line)
                else:
                    in_followup_section = False
                    modified_lines.append(line)
            else:
                modified_lines.append(line)

        final_qs = followup_questions if followup_questions else None
        return final_qs, "\n".join(modified_lines).strip()

class GroundingResponseParser:
    """Convert Search Grounding Metadata."""

    @staticmethod
    def convert_gcs_uri(uri: Optional[str]) -> Optional[str]:
        """Map the GCS URI to storage endpoints"""
        if not uri:
            return None
        if uri.startswith("gs://"):
            return uri.replace("gs://", "https://storage.googleapis.com/", 1).replace(" ", "%20")
        return uri

    @classmethod
    def safe_slice_bytes(cls, text: str, start_byte: int, end_byte: int, encoding: str = "utf-8") -> str:
        """
        safe slicing with boundary checking
        """
        if not text:
            return ""
        try:
            text_bytes = text.encode(encoding)
            start_byte = max(0, min(start_byte, len(text_bytes)))
            end_byte = max(start_byte, min(end_byte, len(text_bytes)))

            sliced_bytes = text_bytes[start_byte:end_byte]
            return sliced_bytes.decode(encoding, errors="replace")
        except Exception as e:
            logger.error(f"Slicing encountered an unexpected byte offset error: {e}. Slicing characters instead.")
            return text[start_byte:end_byte]

    @classmethod
    def parse_response(cls, response: Any) -> Dict[str, Any]:
        """Parsing of candidate structures, maps citations, and extracts titles and questions."""
        if not (response and response.candidates and response.candidates[0].grounding_metadata):
            logger.warning("Raw candidate metadata is missing. Failsafe parsing used.")
            return {
                'output_text': response.text if response else "",
                'keywords': [],
                'results': [],
                'title': None,
                'followup_questions': []
            }

        metadata = response.candidates[0].grounding_metadata
        text = response.text or ""
        keywords = metadata.web_search_queries or []

        # Extract search engines reference links
        results = []
        for i, chunk in enumerate(metadata.grounding_chunks or [], start=1):
            context = chunk.web or chunk.retrieved_context
            if not context:
                continue
            results.append({
                'number': i,
                'url': cls.convert_gcs_uri(context.uri),
                'title': context.title or f"Source {i}"
            })

        # Map footnotes dynamically over sorted boundaries
        markdown_parts = []
        last_byte_index = 0
        supports = metadata.grounding_supports or []
        sorted_supports = sorted(supports, key=lambda s: s.segment.end_index)

        for support in sorted_supports:
            end_byte = support.segment.end_index
        segment_text = cls.safe_slice_bytes(text, last_byte_index, end_byte)
        markdown_parts.append(segment_text)

        footnotes = "".join([f"[{idx + 1}]" for idx in support.grounding_chunk_indices])
        markdown_parts.append(f" {footnotes}")
        last_byte_index = end_byte

    # Append trails safely
    text_bytes_len = len(text.encode("utf-8"))
    if last_byte_index < text_bytes_len:
        markdown_parts.append(cls.safe_slice_bytes(text, last_byte_index, text_bytes_len))

    full_markdown = "".join(markdown_parts)
    followups, text_no_followups = MarkdownProcessor.extract_followup_questions(full_markdown)
    title, final_text = MarkdownProcessor.extract_title(text_no_followups)

    return {
        'output_text': final_text,
        'keywords': keywords,
        'results': results,
        'title': title,
        'followup_questions': followups
    }

class VertexSearchEngine:
    """orchestrating of queries, observability & retries."""

    def __init__(self, config: GroundingConfig):
        self.config = config
        self._client = None

    def get_client(self) -> Any:

        if self._client is None:
            from google import genai
            self._client = genai.Client(
                vertexai=True,
                project=self.config.project_id,
                location=self.config.location,
            )
        return self._client

    def generate_with_grounding(self, user_query: str, local_answer: str, prompt_template: str) -> Dict[str, Any]:
        """
        main execution
        """
        from google.genai import types

        today_date = date.today().strftime("%Y-%m-%d")
        prompt = prompt_template.format(
            USER_QUERY=user_query,
            LOCAL_ANSWER=local_answer,
            TODAY_DATE=today_date
        )

        generate_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_output_tokens=self.config.max_output_tokens,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            ],
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        ]

        client = self.get_client()
        attempt = 0
        delay = self.config.retry_initial_delay

        while attempt < self.config.retry_max_attempts:
            attempt += 1
            start_time = time.time()
            try:
                logger.info(f"Posting LLM Request [Attempt {attempt}/{self.config.retry_max_attempts}]...")
                response = client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=generate_config
                )

                latency = time.time() - start_time
                usage = response.usage_metadata if hasattr(response, 'usage_metadata') else None
                prompt_tokens = usage.prompt_token_count if usage else 0
                candidates_tokens = usage.candidates_token_count if usage else 0

                logger.info(f"API Succeeded. Latency: {latency:.2f}s | Total Tokens: {prompt_tokens + candidates_tokens}")

                parsed_output = GroundingResponseParser.parse_response(response)

                # Log usage and performance metrics
                parsed_output['metadata'] = {
                    'latency_seconds': latency,
                    'prompt_tokens': prompt_tokens,
                    'candidates_tokens': candidates_tokens,
                    'total_tokens': prompt_tokens + candidates_tokens,
                    'model_version': response.model_version if hasattr(response, 'model_version') else self.config.model_name
                }

                return parsed_output

            except Exception as e:
                latency = time.time() - start_time
            logger.error(f"Transient error in Attempt {attempt}: {e} (Duration: {latency:.2f}s)")

            if attempt == self.config.retry_max_attempts:
                logger.critical("Max retries exhausted. Operations terminated.")
                raise RuntimeError(f"Grounding failed after {self.config.retry_max_attempts} attempts: {e}") from e

            logger.warning(f"Backoff: Pausing execution for {delay:.2f}s...")
            time.sleep(delay)
            delay *= self.config.retry_backoff_factor

class TestGroundingEngine(unittest.TestCase):
    """unit testing framework"""

    def test_extract_title_valid(self):
        text = "# Standard Title\nThis is some body content."
        title, modified = MarkdownProcessor.extract_title(text)
        self.assertEqual(title, "Standard Title")
        self.assertEqual(modified, "This is some body content.")

    def test_extract_title_missing(self):
        text = "This is some body content without H1 header."
        title, modified = MarkdownProcessor.extract_title(text)
        self.assertIsNone(title)
        self.assertEqual(modified, "This is some body content without H1 header.")

    def test_extract_followup_questions_valid(self):
        text = "Body text.\n## Followup Questions\n- Is this an API?\n* Can I scale this?\n## Next Heading"
        questions, modified = MarkdownProcessor.extract_followup_questions(text)
        self.assertEqual(questions, ["Is this an API?", "Can I scale this?"])
        self.assertEqual(modified, "Body text.\n## Next Heading")

    def test_safe_slice_bytes_multibyte(self):
        # Japanese character set '新宿御苑' (12 Bytes in UTF-8)
        text = "新宿御苑 is in Tokyo"
        # Slicing precisely on boundaries (0 to 6 bytes) should be '新宿' (6 bytes)
        sliced = GroundingResponseParser.safe_slice_bytes(text, 0, 6)
        self.assertEqual(sliced, "新宿")

        # Slicing down the middle of a boundary (0 to 5) must use replace flags without crash
        sliced_broken = GroundingResponseParser.safe_slice_bytes(text, 0, 5)
        self.assertTrue("新" in sliced_broken)

# Run unit testing suite on script variables
suite = unittest.TestLoader().loadTestsFromTestCase(TestGroundingEngine)
unittest.TextTestRunner(verbosity=2).run(suite)

prompt_template = """<role><!--Role information for generating responses-->
Name: AndesGPT
developer: OPPO Group
description: The AndesGPT is designed to answer all questions that need web search based on user's query and local answer.
</role>

<task><!--Description of the current task-->
You will receive a user's query, and a local answer based on local content;
Understand the query thoroughly and semantically and decide whether the query need a web search:
        - If not, just reply "No online content needed";
        - If so, you judge whether the query has enough information to search online:
                - If so, you search the answer online for the query;
                - If not, you revise the query with the local answer to make it complete, then search online.
Be bold and get to the the point, don't tell me what you are going to do, just do it and get the answer.
</task>

<userQuery><!--User question to be answered-->
{USER_QUERY}
</userQuery>

<localsearchQA><!--answer retrived locally-->
{LOCAL_ANSWER}
</localsearchQA>

<additionalInfo><!--Additional information-->
## Time-related

- Today's date is {TODAY_DATE},
</additionalInfo>

<style><!--Stylistic requirements for generating responses-->
Use markdown format for output;
- Use bold format to highlight key elements in the answer; each answer must include at least one bolded part;
- For any parallel elements, use unordered list format with line breaks, and bold each list item keyword;
- Use short sentences and unordered lists with line breaks as much as possible to clearly present the key points in the answer;
- Avoid using terms like "chunk," "reference," or "userQuery" in the answer;
- Limit the answer to no more than 6 sentences.
- Write a title in heading 1 for the current conversation based on the question and answer, not exceeding 20 characters;
- Include a "Followup Questions" heading 2 and a bullet list of followup questions at the end of the answer.
</style>

<commonConstraint><!--General constraints-->
All ordered/unordered lists must be formatted with line breaks after each item.
For any counting-related requests, such as "How many unfinished tasks do I have?", you should first list all relevant information that meets the requirements and then answer based on the listed information.
</commonConstraint>"""

# Initialize core engine
engine = VertexSearchEngine(config=GroundingConfig())

# Test Case 1: Coldplay Grounded Query
user_query_1 = "Based on the previously saved Coldplay concert tickets, output more online information about Coldplay: band introduction, Spotify playlist, and details of the next concert."
local_answer_1 = "Here’s your agenda for Coldplay concert  Location: Goyang Stadium at Goyang Sports Complex, Goyang-si, South Korea. Seat：15A、15B Date：May 1st source 1 source 2"

try:
    result_1 = engine.generate_with_grounding(user_query_1, local_answer_1, prompt_template)

    print("\n=== TEST CASE 1: PROCESSED OUTPUT ===")
    print(f"Parsed Title: {result_1['title']}")
    print(f"Parsed Followup Questions: {result_1['followup_questions']}")
    print("Grounded Text Content:")
    display(Markdown(result_1['output_text']))

    print("\n=== REFERENCED SOURCES ===")
    for src in result_1['results']:
        print(f"[{src['number']}] {src['title']} -> {src['url']}")

    print("\n=== INSTRUMENTATION METADATA ===")
    print(result_1['metadata'])

except Exception as err:
    print(f"Execution aborted on test case 1: {err}")

# Test Case 2: Multi-byte Japanese Character Travel Guide and Weather
user_query_2 = "What is the weather today? I collected several Japanese tourist attractions yesterday. Please make me a three-day travel guide to Japan, including the collected attractions"
local_answer_2 = "Based on your query, I found the following Japanese tourist attractions from memory: Arakurayama Sengen Park, Osaka Castle, and Shinjuku Gyoen."

try:
    result_2 = engine.generate_with_grounding(user_query_2, local_answer_2, prompt_template)

    print("\n=== TEST CASE 2: PROCESSED OUTPUT ===")
    print(f"Parsed Title: {result_2['title']}")
    print(f"Parsed Followup Questions: {result_2['followup_questions']}")
    print("Grounded Text Content:")
    display(Markdown(result_2['output_text']))

    print("\n=== REFERENCED SOURCES ===")
    for src in result_2['results']:
        print(f"[{src['number']}] {src['title']} -> {src['url']}")

    print("\n=== INSTRUMENTATION METADATA ===")
    print(result_2['metadata'])

except Exception as err:
    print(f"Execution aborted on test case 2: {err}")
