## Overview

# This is a demo on how to analyze an entire codebase with Gemini 1.5 Pro.
# It uses long context window to analyze, classify and summarize large amounts of content within a given prompt.

## Setup & Installation
import os
import shutil
import logging
import vertexai
import git
import magika
import requests

from pathlib import Path
from typing import Tuple, List, Optional, Any
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import Markdown, display

from vertexai.generative_models import (
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Tool,
)

PROJECT_ID = "chayan-genai-bb"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)

InteractiveShell.ast_node_interactivity = "all"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CodebaseManager:
    """Manages the lifecycle and extraction of a Git repository for AI analysis."""

    def __init__(self, repo_url: str, repo_dir: str):
        self.repo_url = repo_url
        self.repo_dir = Path(repo_dir)
        self._magika = magika.Magika()

    def clone(self) -> None:
        """Clones the repository, ensuring a clean state if the directory already exists."""
        if self.repo_dir.exists():
            logging.info(f"Removing existing directory at {self.repo_dir}")
            shutil.rmtree(self.repo_dir)

        logging.info(f"Cloning {self.repo_url} into {self.repo_dir}...")
        git.Repo.clone_from(self.repo_url, self.repo_dir)
        logging.info("Cloning complete.")

    def extract_code(self) -> Tuple[str, str]:
        """Extracts the index and textual content of code files recursively."""
        index_files = []
        text_blocks = []

        for filepath in self.repo_dir.rglob("*"):
            if not filepath.is_file() or ".git" in filepath.parts:
                continue

            rel_path = filepath.relative_to(self.repo_dir)
            index_files.append(str(rel_path))

            try:
                file_type = self._magika.identify_path(filepath)
                if file_type.output.group in ("text", "code"):
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                    text_blocks.append(f"----- File: {rel_path} -----\n{content}\n-------------------------\n")
            except Exception as e:
                logging.warning(f"Failed to process {rel_path}: {e}")

        return "\n".join(index_files), "\n".join(text_blocks)


class GeminiCodeAnalyzer:
    """Encapsulates interaction with the Gemini API for robust code analysis."""

    def __init__(self, model_id: str, code_index: str, code_text: str):
        self.model = GenerativeModel(
            model_id,
            system_instruction=[
                "You are a coding expert.",
                "Your mission is to answer all code related questions with given context and instructions.",
            ],
        )
        self.code_index = code_index
        self.code_text = code_text

    def _build_prompt(self, question: str, add_context: bool = True) -> str:
        if not add_context:
            return question

        return (
            f"Questions: {question}\n\n"
            f"Context:\n"
            f"- The entire codebase is provided below.\n"
            f"- Here is an index of all of the files in the codebase:\n\n{self.code_index}\n\n."
            f"- Then each of the files is concatenated together. You will find all of the code you need:\n\n"
            f"{self.code_text}\n\n"
            f"Answer:\n"
        )

    def analyze_and_display(self, question: str, stream: bool = True, tools: Optional[List[Tool]] = None, generation_config: Optional[GenerationConfig] = None, add_context: bool = True) -> Any:
        """Queries the model and safely displays the response, guarding against safety blocks."""
        prompt = self._build_prompt(question, add_context)

        try:
            if stream and not tools:
                responses = self.model.generate_content([prompt], stream=True)
                for response in responses:
                    display(Markdown(response.text))
            else:
                response = self.model.generate_content(
                    [prompt],
                    tools=tools,
                    generation_config=generation_config
                )
                # Support for tool invocation returns
                if response.candidates and response.candidates[0].function_calls:
                    return response.candidates[0].function_calls[0]

                display(Markdown(response.text))

        except ValueError as ve:
            display(Markdown(f"> **Safety Filter Triggered:** The model generation was blocked by safety filters.\n> *Details: {ve}*"))
        except Exception as e:
            display(Markdown(f"> **Execution Error:** {e}"))


class GitHubHelper:
    """Utility class for interfacing with the GitHub API."""

    @staticmethod
    def get_issue_body(owner: str, repo: str, issue_number: str) -> str:
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json().get("body", "")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching issue {issue_number}: {e}")
            return ""

def main():
    # Prepare the target codebase for analysis.
    
    REPO_URL = "https://github.com/GoogleCloudPlatform/microservices-demo"
    REPO_DIR = "./repo"
    
    manager = CodebaseManager(repo_url=REPO_URL, repo_dir=REPO_DIR)
    manager.clone()
    code_index, code_text = manager.extract_code()
    
    analyzer = GeminiCodeAnalyzer(
        model_id="gemini-1.5-pro",
        code_index=code_index,
        code_text=code_text
    )
    
    ## 1. Summarizing the codebase
    
    analyzer.analyze_and_display(
        "Give me a summary of this codebase, and tell me the top 3 things that I can learn from it.",
        stream=False
    )
    
    ## 2. Creating a developer getting started guide
    
    analyzer.analyze_and_display("Provide a getting started guide to onboard new developers to the codebase.")
    
    ## 3. Finding bugs
    # *Note: This showcases the robust error handling built in to prevent exceptions from safety filters.*
    
    
    analyzer.analyze_and_display("Find the top 3 most severe issues in the codebase.")
    
    ## 4. Fixing a bug
    
    analyzer.analyze_and_display("Find the most severe bug in the codebase that you can provide a code fix for.")
    
    ## 5. Implementing a feature request using Function Calling
    
    extract_details_from_url_func = FunctionDeclaration(
        name="extract_details_from_url",
        description="Extracts owner, repository name, and issue number details from a GitHub issue URL",
        parameters={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "The owner of the GitHub repository."},
                "repo": {"type": "string", "description": "The name of the GitHub repository."},
                "issue_number": {"type": "string", "description": "The issue number to fetch the body of."},
            },
        },
    )
    
    extraction_tool = Tool(function_declarations=[extract_details_from_url_func])
    FEATURE_REQUEST_URL = "https://github.com/GoogleCloudPlatform/microservices-demo/issues/2205"
    
    function_call = analyzer.analyze_and_display(
        f"What is the feature request of the following {FEATURE_REQUEST_URL}",
        stream=False,
        tools=[extraction_tool],
        generation_config=GenerationConfig(temperature=0),
        add_context=False
    )
    
    if function_call and function_call.name == "extract_details_from_url":
        issue_body = GitHubHelper.get_issue_body(
            function_call.args["owner"],
            function_call.args["repo"],
            function_call.args["issue_number"]
        )
    
        display(Markdown(f"### Fetched Feature Request\n{issue_body}"))
    
        analyzer.analyze_and_display(
            f"Implement the following feature request: {FEATURE_REQUEST_URL}\n{issue_body}"
        )
    
    ## 6. Creating a troubleshooting guide
    
    analyzer.analyze_and_display("Provide a troubleshooting guide to help resolve common issues.")
    
    ## 7. Making the app more reliable
    
    analyzer.analyze_and_display("How can I make this application more reliable? Consider best practices from https://www.r9y.dev/")
    
    ## 8. Making the app more secure
    
    analyzer.analyze_and_display("How can you secure the application?")
    
    ## 9. Learning the codebase (Quiz)
    
    analyzer.analyze_and_display("Create a quiz about the concepts used in my codebase to help me solidify my understanding.")
    
    ## 10. Creating a quickstart tutorial
    
    analyzer.analyze_and_display(
        "Please write an end-to-end quickstart tutorial that introduces AlloyDB,\n"
        "shows how to configure it with the CartService,\n"
        "and highlights key capabilities of AlloyDB in context of the Online Boutique application."
    )
    
    ## 11. Creating a Git Changelog Generator
    
    repo = git.Repo(manager.repo_dir)
    commits = list(repo.iter_commits("main", max_count=2))
    
    if len(commits) >= 2:
        diff_text = repo.git.diff(commits[0].hexsha, commits[1].hexsha)
        analyzer.analyze_and_display(
            f"Given the following git diff output, summarize the important changes made.\n\n{diff_text}"
        )
    else:
        logging.info("Not enough commits to generate a diff.")

if __name__ == "__main__":
    main()
