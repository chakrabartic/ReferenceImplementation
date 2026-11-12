## LLM Evaluation Pipeline Demo
## Demo of an evaluation pipeline designed to orchestrate LLM inference across Vertex AI (Chat Bison), Azure OpenAI (GPT-3.5), and vLLM (Mistral), followed by an automated pairwise evaluation using GPT-4 via LangChain.

## Global Imports

import os
import sys
import time
import random
import logging
import functools
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd
import pandas_gbq
from google.cloud import bigquery
from google.cloud import aiplatform
import openai

from vertexai.language_models import ChatModel
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatVertexAI
from langchain.evaluation import load_evaluator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure structured enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Eval-Pipeline")

## Configuration Management

@dataclass(frozen=True)
class PipelineConfig:

    # GCP Configuration
    PROJECT_ID: str = "genai-blackbelt-fishfooding"
    REGION: str = "us-central1"

    # Inference Hyperparameters
    TEMPERATURE: float = 0.2
    MAX_NEW_TOKENS: int = 512
    TOP_P: float = 0.98
    TOP_K: int = 40
    CANDIDATES: int = 2

    # System Prompt
    DEFAULT_SYSTEM_PROMPT: str = ""

    # BigQuery Tables
    BQ_SOURCE_TABLE: str = "genai-blackbelt-fishfooding.bqdset_eu_prompt_evaluation.prompt-evaluation-sheet-salm"
    BQ_DEST_TABLE: str = "bqdset_eu_prompt_evaluation.prompt-evaluation-result-v2"

    # Local Checkpoints and Outputs
    CSV_RESPONSES_CHECKPOINT: str = "responses_dataset_checkpoint.csv"
    CSV_RESPONSES_FINAL: str = "responses_dataset.csv"
    CSV_EVAL_FINAL: str = "final_df_GPT4.csv"

    # API Settings (with secure environment variable fallbacks)
    OPENAI_API_TYPE: str = field(default_factory=lambda: os.getenv("OPENAI_API_TYPE", "azure"))
    OPENAI_API_BASE: str = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", "YOUR_DEDICATED_API_BASE"))
    OPENAI_API_VERSION: str = field(default_factory=lambda: os.getenv("OPENAI_API_VERSION", "YOUR_API_VERSION"))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "YOUR_API_KEY"))

    @property
    def bison_parameters(self) -> Dict[str, Any]:
        return {
            "candidate_count": self.CANDIDATES,
            "max_output_tokens": self.MAX_NEW_TOKENS,
            "temperature": self.TEMPERATURE,
            "top_p": self.TOP_P,
            "top_k": self.TOP_K
        }

config = PipelineConfig()

## Encapsulated Model Inference Engine

class ModelInferenceEngine:
    """
    Handles authentication, client initialization, and execution abstraction
    for Vertex AI, Azure OpenAI, and Custom vLLM endpoints with integrated retries.
    """
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._init_gcp()
        self._init_openai()

    def _init_gcp(self) -> None:
        logger.info("Initializing Vertex AI SDK...")
        # Authenticate with Colab environment safely
        if "google.colab" in str(get_ipython()):
            from google.colab import auth as google_auth
            logger.info("Colab detected. Authenticating Google user...")
            google_auth.authenticate_user()

        aiplatform.init(project=self.cfg.PROJECT_ID, location=self.cfg.REGION)

        try:
            self.chat_bison_private = ChatModel.from_pretrained("chat-bison@private")
            self.chat_bison_latest = ChatModel.from_pretrained("chat-bison@latest")
            logger.info("Vertex AI models successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load Vertex models: {e}. Check IAM roles.")
            raise e

    def _init_openai(self) -> None:
        logger.info("Configuring OpenAI/Azure SDK credentials...")
        openai.api_type = self.cfg.OPENAI_API_TYPE
        openai.api_base = self.cfg.OPENAI_API_BASE
        openai.api_version = self.cfg.OPENAI_API_VERSION
        openai.api_key = self.cfg.OPENAI_API_KEY

    # Standardized retry logic: Truncated exponential backoff with Jitter
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Transient issue calling API. Attempt {retry_state.attempt_number} failed. Retrying..."
        )
    )
    def predict_bison_private(self, question: str) -> str:
        chat = self.chat_bison_private.start_chat(context=self.cfg.DEFAULT_SYSTEM_PROMPT)
        response = chat.send_message(question, **self.cfg.bison_parameters)
        return response.text

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        reraise=True
    )
    def predict_bison_latest(self, question: str) -> str:
        chat = self.chat_bison_latest.start_chat(context=self.cfg.DEFAULT_SYSTEM_PROMPT)
        response = chat.send_message(question, **self.cfg.bison_parameters)
        return response.text

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        reraise=True
    )
    def predict_gpt35(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.cfg.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        response = openai.ChatCompletion.create(
            engine="oa-coeml-gpt35-turbo",
            messages=messages,
            temperature=self.cfg.TEMPERATURE,
            max_tokens=self.cfg.MAX_NEW_TOKENS,
            top_p=self.cfg.TOP_P,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            stop=None
        )
        return response.choices[0].message.content

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1.5, min=2, max=15),
        reraise=True
    )
    def predict_mistral(self, question: str, endpoint: Any) -> str:
        prompt = self.build_prompt_chatml(question)
        prompt_length = len(prompt)
        instances = [{
            "prompt": prompt,
            "use_beam_search": False,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.2,
            "n": self.cfg.CANDIDATES,
            "top_p": self.cfg.TOP_P ,
            "top_k": self.cfg.TOP_K ,
            "max_tokens": self.cfg.MAX_NEW_TOKENS ,
            "temperature": self.cfg.TEMPERATURE
        }]
        # Call custom vLLM endpoint
        prediction = endpoint.predict(instances=instances).predictions[0]
        return prediction[prompt_length + 5:]

    def build_prompt_llama2(self, instruction: str) -> str:
        return f"<s>[INST]<<SYS>>\n{self.cfg.DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n\n{instruction}[/INST]\n"

    def build_prompt_chatml(self, instruction: str) -> str:
        return f"<|system|>\n{self.cfg.DEFAULT_SYSTEM_PROMPT}\n<|user|>\n{instruction}\n<|assistant|>\n"

## Concurrent Inference Manager with Checkpointing

class InferenceBatchProcessor:
    """
    Inference over Pandas dataframes with checkpoint state persistence.
    """
    def __init__(self, engine: ModelInferenceEngine, cfg: PipelineConfig):
        self.engine = engine
        self.cfg = cfg

    def process_all_models(
        self,
        df: pd.DataFrame,
        endpoint_openorca: Optional[Any] = None,
        max_workers: int = 4
    ) -> pd.DataFrame:
        # Initialize target columns in dataframe safely if not present
        target_cols = ["bisonprivate_response", "bisonlatest_response", "gpt_response", "mistral_response"]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        # State Recovery: Load checkpoint file if it exists
        if os.path.exists(self.cfg.CSV_RESPONSES_CHECKPOINT):
            logger.info(f"Found checkpoint: '{self.cfg.CSV_RESPONSES_CHECKPOINT}'. Loading state...")
            checkpoint_df = pd.read_csv(self.cfg.CSV_RESPONSES_CHECKPOINT)
            # Safely align any structural differences
            for col in target_cols:
                if col in checkpoint_df.columns:
                    df[col] = checkpoint_df[col]

        # Identify indices needing processing
        pending_idx = df[
            df["bisonprivate_response"].isna() |
            df["bisonlatest_response"].isna() |
            df["gpt_response"].isna() |
            (df["mistral_response"].isna() if endpoint_openorca is not None else False)
        ].index.tolist()

        if not pending_idx:
            logger.info("All rows are already fully processed. Skipping inference phase.")
            return df

        logger.info(f"Orchestrating concurrent execution. total_rows={len(df)}, pending_rows={len(pending_idx)}")

        def process_row(idx: int) -> Tuple[int, Dict[str, str]]:
            question = df.at[idx, "question"]
            results = {}

            # Bison Private
            if pd.isna(df.at[idx, "bisonprivate_response"]):
                try:
                    results["bisonprivate_response"] = self.engine.predict_bison_private(question)
                except Exception as e:
                    logger.error(f"Row {idx} BisonPrivate failed: {e}")

            # Bison Latest
            if pd.isna(df.at[idx, "bisonlatest_response"]):
                try:
                    results["bisonlatest_response"] = self.engine.predict_bison_latest(question)
                except Exception as e:
                    logger.error(f"Row {idx} BisonLatest failed: {e}")

            # GPT-3.5
            if pd.isna(df.at[idx, "gpt_response"]):
                try:
                    results["gpt_response"] = self.engine.predict_gpt35(question)
                except Exception as e:
                    logger.error(f"Row {idx} GPT3.5 failed: {e}")

            # Mistral
            if endpoint_openorca is not None and pd.isna(df.at[idx, "mistral_response"]):
                try:
                    results["mistral_response"] = self.engine.predict_mistral(question, endpoint_openorca)
                except Exception as e:
                    logger.error(f"Row {idx} Mistral failed: {e}")

            return idx, results

        # Execute parallel processing
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="WorkerPool") as executor:
            future_to_idx = {executor.submit(process_row, idx): idx for idx in pending_idx}

            for future in as_completed(future_to_idx):
                idx, row_results = future.result()
                for col, value in row_results.items():
                    df.at[idx, col] = value

                completed_count += 1

                # Transactional Checkpointing: Commit to disk every 10 complete steps
                if completed_count % 10 == 0 or completed_count == len(pending_idx):
                    df.to_csv(self.cfg.CSV_RESPONSES_CHECKPOINT, index=False)
                    logger.info(f"Checkpoint persisted on step {completed_count}/{len(pending_idx)}")

        logger.info("Inference batch processing successfully finished.")
        return df

## Secure BigQuery Data Access Object (DAO)

class BigQueryDAO:
    """
    Handles all read, write, and table creation logic with BigQuery.
    """
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.client = bigquery.Client(project=self.cfg.PROJECT_ID)

    def fetch_prompts(self) -> pd.DataFrame:
        query = f"""
            SELECT string_field_1 as question
            FROM `{self.cfg.BQ_SOURCE_TABLE}`
        """
        logger.info(f"Fetching evaluation dataset from BigQuery source: '{self.cfg.BQ_SOURCE_TABLE}'")
        try:
            _query_job = self.client.query(query)
            df = _query_job.to_dataframe()
            logger.info(f"Successfully pulled {len(df)} questions from BigQuery.")
            return df
        except Exception as e:
            logger.error(f"BigQuery ingestion failed: {e}")
            raise e

    def upload_results(self, df: pd.DataFrame) -> None:
        logger.info(f"Uploading execution results dataset to BQ table: '{self.cfg.BQ_DEST_TABLE}'")
        try:
            pandas_gbq.to_gbq(
                df,
                self.cfg.BQ_DEST_TABLE,
                project_id=self.cfg.PROJECT_ID,
                if_exists="replace"
            )
            logger.info("BigQuery upload complete.")
        except Exception as e:
            logger.error(f"Failed uploading dataset to BigQuery: {e}")
            raise e

## Dynamic Pairwise Evaluation Engine (LangChain Battle)

class PairwiseEvaluationEngine:
    """
    Executes LangChain's pairwise string evaluation frameworks.
    """
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.eval_llm = AzureChatOpenAI(
            openai_api_base=self.cfg.OPENAI_API_BASE,
            openai_api_version=self.cfg.OPENAI_API_VERSION,
            deployment_name="oa-coeml-gpt4",
            openai_api_key=self.cfg.OPENAI_API_KEY,
            openai_api_type="azure"
        )

        self.custom_criteria = {
            "simplicity": "Is the language straightforward and unpretentious?",
            "clarity": "Are the sentences clear and easy to understand?",
            "precision": "Is the writing precise, with no unnecessary words or details?",
            "truthfulness": "Does the writing feel honest and sincere?",
            "helpfull": "Does the writing feel helpfull?",
            "language": "Does the writing is using the language of the user?",
            "missions": "Does the writing is following the system missions?"
        }

    def evaluate_pairwise(
        self,
        df: pd.DataFrame,
        eval_params: List[Dict[str, str]]
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        # Dynamic column construction
        new_columns_list = []
        for p in eval_params:
            track_var = p["track_var"]
            new_columns_list.extend([track_var, f"reasoning_{track_var}", f"score_{track_var}"])

        # FIX CRITICAL BUG: list.append() returns NoneType, crashing original reindex call.
        # Clean list concatenation resolves this.
        extended_cols = list(df.columns) + new_columns_list
        df = df.reindex(columns=extended_cols, fill_value=None)

        evaluator = load_evaluator(
            "pairwise_string",
            llm=self.eval_llm,
            criteria=self.custom_criteria
        )

        # Process row evaluations
        for idx in df.index:
            logger.info(f"Evaluating row index: {idx}")

            for p in eval_params:
                name = p["name"]
            track_var = p["track_var"]
            col_a = p["llmA_response"]
            col_b = p["llmB_response"]

            try:
               pred_a = df.at[idx, col_a]
               pred_b = df.at[idx, col_b]
               question = df.at[idx, "question"]

               if pd.isna(pred_a) or pd.isna(pred_b):
                    logger.warning(f"Skipping {name} eval on row {idx} due to missing model prediction.")
               continue

               eval_result = evaluator.evaluate_string_pairs(
                prediction=str(pred_a),
                prediction_b=str(pred_b),
                input=str(question)
              )

               df.at[idx, track_var] = eval_result.get("value")
               df.at[idx, f"reasoning_{track_var}"] = eval_result.get("reasoning")
               df.at[idx, f"score_{track_var}"] = eval_result.get("score")

            except Exception as e:
                logger.error(f"Failed {name} evaluation on row {idx}: {e}")
                df.at[idx, track_var] = None
                df.at[idx, f"reasoning_{track_var}"] = None
                df.at[idx, f"score_{track_var}"] = None

    # Compile aggregation performance metrics metadata-driven style
    metrics_summary = {}
    for p in eval_params:
        track_var = p["track_var"]
        # Clean count calculation
        metrics_summary[track_var] = df[[track_var, "question"]].groupby(track_var).count()

    return df, metrics_summary

## Main Pipeline Orchestration Entry Point

def main():
    logger.info("Starting Pipeline Job...")

    # Initialize Core Classes
    dao = BigQueryDAO(config)
    engine = ModelInferenceEngine(config)
    processor = InferenceBatchProcessor(engine, config)
    evaluator_engine = PairwiseEvaluationEngine(config)

    # Phase 1: Data Ingestion
    df_raw = dao.fetch_prompts()

    # Optional: Dry Run Slice for Development (Change to full dataset for production runs)
    df_execution = df_raw.copy()

    # Phase 2: Orchestrated LLM Inference with concurrency and checkpointing
    df_responses = processor.process_all_models(df_execution, endpoint_openorca=None, max_workers=5)

    # Safe persistence of the inference responses
    df_responses.to_csv(config.CSV_RESPONSES_FINAL, index=False)
    dao.upload_results(df_responses)

    # Phase 3: Dynamic Evaluation execution
    evaluation_parameters = [
        {
            "name": "BisonLatest vs Bisonprivate",
            "track_var": "Latest_Private",
            "llmA_response": "bisonlatest_response",
            "llmB_response": "bisonprivate_response"
        },
        {
            "name": "GPT vs Bisonprivate",
            "track_var": "GPT_Private",
            "llmA_response": "gpt_response",
            "llmB_response": "bisonprivate_response"
        },
        {
            "name": "GPT vs BisonLatest",
            "track_var": "GPT_Latest",
            "llmA_response": "gpt_response",
            "llmB_response": "bisonlatest_response"
        }
    ]

    final_evaluated_df, summary_metrics = evaluator_engine.evaluate_pairwise(
        df=df_responses,
        eval_params=evaluation_parameters
    )

    # Final State persistence
    final_evaluated_df.to_csv(config.CSV_EVAL_FINAL, index=False)
    logger.info(f"Evaluation Complete. Datasets written to '{config.CSV_EVAL_FINAL}'")

    # Visual Output Display
    for comparison, df_metric in summary_metrics.items():
        print(f"\n=== Comparison Results for: {comparison} ===")
        print(df_metric)

if __name__ == "__main__":
    main()
