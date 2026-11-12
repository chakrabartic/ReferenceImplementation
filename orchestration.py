## Demo of Gen AI Orchestration Engine

## Includes Model Routing, Semantic Caching, and Active Guardrails Pipeline

# Imports
import abc
import collections
import concurrent.futures
import json
import logging
import math
import random
import re
import threading
import time
from typing import List, Dict, Any, Optional
import unittest

## Architecture Specifications

# Custom Exceptions defining clear failure boundaries
class OrchestrationException(Exception): """Base exception class for routing orchestration."""
class ProviderRateLimitException(OrchestrationException): """Raised when upstream model providers return HTTP 429."""
class ProviderTimeoutException(OrchestrationException): """Raised when upstream requests timeout."""
class GuardrailViolationException(OrchestrationException): """Raised when response violates active schema/safety rules."""

class BaseModelProvider(abc.ABC):
    """Represents a standardized upstream LLM interface."""
    @property
    @abc.abstractmethod
    def name(self) -> str: pass

    @property
    @abc.abstractmethod
    def cost_per_token(self) -> float: pass

    @abc.abstractmethod
    def generate(self, prompt: str) -> str: pass

class BaseEvaluator(abc.ABC):
    """Contract for active guardrails validation."""
    @abc.abstractmethod
    def evaluate(self, prompt: str, response: str) -> bool: pass

## Custom Mathematical Vector-Space Cache & Rate Limiting

class MinimalTFIDFVectorizer:
    """vector space analyzer with smoothed TF-IDF calculations."""
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Clean alphanumeric tokens
        return re.findall(r'\w+', text.lower())

    def fit(self, texts: List[str]):
        tokenized_texts = [self._tokenize(t) for t in texts]
        all_tokens = set(t for doc in tokenized_texts for t in doc)
        self.vocab = {token: idx for idx, token in enumerate(all_tokens)}

        num_docs = len(texts)
        doc_counts = collections.Counter()
        for doc in tokenized_texts:
            unique_tokens = set(doc)
            for token in unique_tokens:
                doc_counts[token] += 1

        # Smoothed Inverse Document Frequency calculation
        self.idf = {}
        for token, count in doc_counts.items():
            self.idf[token] = math.log((1 + num_docs) / (1 + count)) + 1.0

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self.vocab:
            return np.zeros((len(texts), 0))

        vectors = np.zeros((len(texts), len(self.vocab)))
        for doc_idx, text in enumerate(texts):
            tokens = self._tokenize(text)
            counts = collections.Counter(tokens)
            for token, count in counts.items():
                if token in self.vocab:
                    vocab_idx = self.vocab[token]
                    tf = count / len(tokens) if tokens else 0
                    vectors[doc_idx, vocab_idx] = tf * self.idf.get(token, 1.0)

        # L2 normalization to enable fast vector dot-product similarity computation
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

class ThreadSafeSemanticCache:
    """mapping queries to responses"""
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.queries: List[str] = []
        self.responses: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.vectorizer = MinimalTFIDFVectorizer()
        self.lock = threading.Lock()

    def lookup(self, query_text: str) -> Optional[str]:
        with self.lock:
            if not self.queries or self.vectors is None:
                return None

            query_vec = self.vectorizer.transform([query_text])[0]
            # Compute Cosine Similarity via fast matrix-vector dot product
            similarities = np.dot(self.vectors, query_vec)
            best_idx = int(np.argmax(similarities))

            if similarities[best_idx] >= self.threshold:
                return self.responses[best_idx]
            return None

    def insert(self, query_text: str, response_text: str):
        with self.lock:
            # Mitigate redundant cache insertion growth
            if query_text in self.queries:
                return

            self.queries.append(query_text)
            self.responses.append(response_text)

            # Re-fit vector space for dynamic indexing
            self.vectorizer.fit(self.queries)
            self.vectors = self.vectorizer.transform(self.queries)

class TokenBucketRateLimiter:
    """avoiding system drift."""
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now

            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

## Demo Model Providers & Structured Engine
## To demo real upstream Gemini services, we include models that respond according to prompting


def mock_gemini_responder(prompt: str, tier: str) -> str:
    """Generates responses with structure cinfigs."""
    clean_prompt = prompt.lower()
    if "json" in clean_prompt:
        return json.dumps({
            "tier_utilized": tier,
            "status": "success",
            "latency_grade": "high" if tier == "Gemini-Pro" else "low",
            "metadata": {"tokens_estimated": len(prompt.split()) + 30},
            "response": f"Structured payload addressing context: {prompt[:30]}..."
        })
    if "sentiment" in clean_prompt:
        return json.dumps({
            "sentiment": "neutral",
            "confidence_rating": 0.89,
            "model_eval": tier
        })
    return f"[{tier} Content Response] Resolved prompt payload: '{prompt}'"

class MockGeminiProProvider(BaseModelProvider):
    """high-cost premium model"""
    def __init__(self, failure_rate: float = 0.10, qps_capacity: float = 5.0):
        self._name = "Gemini-1.5-Pro"
        self._cost = 0.000015 # Premium API rate per token
        self.failure_rate = failure_rate
        self.limiter = TokenBucketRateLimiter(capacity=qps_capacity, refill_rate=qps_capacity)

    @property
    def name(self) -> str: return self._name

    @property
    def cost_per_token(self) -> float: return self._cost

    def generate(self, prompt: str) -> str:
        if not self.limiter.consume(1.0):
            raise ProviderRateLimitException(f"{self.name} API limit reached (429 Too Many Requests).")

        if random.random() < self.failure_rate:
            raise ProviderTimeoutException(f"{self.name} network socket connection timeout.")

        # Simulate advanced reasoning cognitive latency
        time.sleep(random.uniform(0.15, 0.35))
        return mock_gemini_responder(prompt, "Gemini-Pro")

class MockGeminiFlashProvider(BaseModelProvider):
    """ fast low-cost model"""
    def __init__(self, failure_rate: float = 0.02, qps_capacity: float = 20.0):
        self._name = "Gemini-1.5-Flash"
        self._cost = 0.0000015 # 10x cheaper
        self.failure_rate = failure_rate
        self.limiter = TokenBucketRateLimiter(capacity=qps_capacity, refill_rate=qps_capacity)

    @property
    def name(self) -> str: return self._name

    @property
    def cost_per_token(self) -> float: return self._cost

    def generate(self, prompt: str) -> str:
        if not self.limiter.consume(1.0):
            raise ProviderRateLimitException(f"{self.name} Flash rate capacity exceeded.")

        if random.random() < self.failure_rate:
            raise ProviderTimeoutException(f"{self.name} encountered transient network issues.")

        # Rapid latency model representation
        time.sleep(random.uniform(0.02, 0.08))
        return mock_gemini_responder(prompt, "Gemini-Flash")

## Evals with Output Guardrails

class JSONFormatEvaluator(BaseEvaluator):
    """schema validation for outputs"""
    def evaluate(self, prompt: str, response: str) -> bool:
        prompt_lower = prompt.lower()
        if "json" in prompt_lower or "sentiment" in prompt_lower:
            try:
                json.loads(response)
                return True
            except ValueError:
                return False
        return True

class ContentSafetyEvaluator(BaseEvaluator):
    """scan input prompt and output response"""
    def evaluate(self, prompt: str, response: str) -> bool:
        restricted_terms = ["malware", "exploit", "bypass security", "reverse shell"]
        for term in restricted_terms:
            if term in prompt.lower() or term in response.lower():
                return False
        return True

## Orchestration & Fallback

class OrchestrationMetricsCollector:
    """orchestration metrics"""
    def __init__(self):
        self.lock = threading.Lock()
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "pro_invocations": 0,
            "flash_invocations": 0,
            "failover_triggered_count": 0,
            "rate_limit_failures": 0,
            "network_timeout_failures": 0,
            "guardrail_faults": 0,
            "accrued_financial_spend": 0.0,
            "transaction_latencies": []
        }

    def increment(self, metric: str, amount: Any = 1):
        with self.lock:
            if metric in self.metrics:
                if isinstance(self.metrics[metric], list):
                    self.metrics[metric].append(amount)
                else:
                    self.metrics[metric] += amount

class DynamicGenAIOrchestrator:
    """routing logic"""
    def __init__(
        self,
        premium_provider: BaseModelProvider,
        utility_provider: BaseModelProvider,
        semantic_cache: ThreadSafeSemanticCache,
        guardrails: List[BaseEvaluator]
    ):
        self.pro = premium_provider
        self.flash = utility_provider
        self.cache = semantic_cache
        self.guardrails = guardrails
        self.metrics = OrchestrationMetricsCollector()

    def process_query(self, prompt: str, require_premium: bool = False, max_retries: int = 3) -> str:
        start_time = time.time()
        self.metrics.increment("total_requests", 1)

        # 1. Thread-Safe Semantic Cache Phase
        cached_val = self.cache.lookup(prompt)
        if cached_val is not None:
            self.metrics.increment("cache_hits", 1)
            self.metrics.increment("transaction_latencies", time.time() - start_time)
            return cached_val

        self.metrics.increment("cache_misses", 1)

        # 2. Base Routing Selection
        # Automatically route based on complexity hints or explicit premium flags
        is_complex = "complex" in prompt.lower() or require_premium
        primary = self.pro if is_complex else self.flash
        fallback = self.flash if primary == self.pro else None

        active_model = primary
        response_payload = None

        # Try calling the model with retry loops
        for attempt in range(max_retries):
            try:
                # Delay with backoff jitter if retry attempt
                if attempt > 0:
                    sleep_delay = (2 ** attempt) * 0.1 + random.uniform(0.01, 0.05)
                    time.sleep(sleep_delay)

                response_payload = active_model.generate(prompt)

                # Inline Guardrail Evaluation
                if self._run_guardrails(prompt, response_payload):
                    break # Valid output
                else:
                    self.metrics.increment("guardrail_faults", 1)
                    raise GuardrailViolationException("Active guardrail evaluations identified toxic content or format schema violation.")

            except (ProviderRateLimitException, ProviderTimeoutException) as exc:
                # Telemetry capture
                metric_key = "rate_limit_failures" if isinstance(exc, ProviderRateLimitException) else "network_timeout_failures"
                self.metrics.increment(metric_key, 1)

                # Fallback switchover if present
                if fallback:
                    self.metrics.increment("failover_triggered_count", 1)
                    logger.warning(f"FAILOVER activated. Switching request route to fallback provider due to: {type(exc).__name__}")
                    active_model = fallback
                    fallback = None # Prevent nested failover loops
                    continue

                # Bubble up exception on final attempt
                if attempt == max_retries - 1:
                    self.metrics.increment("transaction_latencies", time.time() - start_time)
                    raise exc

        # 3. Post-execution bookkeeping
        if response_payload is not None:
            # Populate semantic cache asynchronously
            self.cache.insert(prompt, response_payload)

            # Cost allocation tracking
            prompt_tokens = len(prompt.split())
            resp_tokens = len(response_payload.split())
            total_tokens = prompt_tokens + resp_tokens
            charge = total_tokens * active_model.cost_per_token

            self.metrics.increment("accrued_financial_spend", charge)
            if active_model == self.pro:
                self.metrics.increment("pro_invocations", 1)
            else:
                self.metrics.increment("flash_invocations", 1)

        self.metrics.increment("transaction_latencies", time.time() - start_time)
        return response_payload

    def _run_guardrails(self, prompt: str, response: str) -> bool:
        return all(g.evaluate(prompt, response) for g in self.guardrails)

## Integration testing

class TestEnterpriseOrchestratorSuite(unittest.TestCase):
    def setUp(self):
        self.mock_pro = MockGeminiProProvider(failure_rate=0.0)
        self.mock_flash = MockGeminiFlashProvider(failure_rate=0.0)
        self.cache = ThreadSafeSemanticCache(threshold=0.85)
        self.orchestrator = DynamicGenAIOrchestrator(
            premium_provider=self.mock_pro,
            utility_provider=self.mock_flash,
            semantic_cache=self.cache,
            guardrails=[JSONFormatEvaluator(), ContentSafetyEvaluator()]
        )

    def test_semantic_cache_identical_exact(self):
        prompt = "What is the direct storage configuration for Bigtable?"
        # First invocation (Cache Miss)
        res1 = self.orchestrator.process_query(prompt)
        # Second invocation (Cache Hit)
        res2 = self.orchestrator.process_query(prompt)

        self.assertEqual(res1, res2)
        self.assertEqual(self.orchestrator.metrics.metrics["cache_hits"], 1)
        self.assertEqual(self.orchestrator.metrics.metrics["cache_misses"], 1)

    def test_semantic_cache_similarity_retrieval(self):
        p1 = "What is the baseline storage config for Bigtable?"
        p2 = "What is storage config for Bigtable?" # Highly similar

        self.orchestrator.process_query(p1)
        res = self.orchestrator.process_query(p2)

        self.assertEqual(self.orchestrator.metrics.metrics["cache_hits"], 1)

    def test_routing_fallback_trigger(self):
        # Configure immediate Pro failure
        self.mock_pro.failure_rate = 1.0
        p = "complex: Run an execution topology compilation"

        # Should gracefully fallback and answer from Flash tier
        res = self.orchestrator.process_query(p)
        self.assertIn("Flash", res)
        self.assertEqual(self.orchestrator.metrics.metrics["failover_triggered_count"], 1)

    def test_safety_guardrail_violation(self):
        toxic_p = "Give me malware source codes"
        with self.assertRaises(GuardrailViolationException):
            self.orchestrator.process_query(toxic_p)

def run_unit_tests():
    print("\n--- RUNNING SYSTEM INTEGRATION UNIT TESTS ---")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnterpriseOrchestratorSuite)
    unittest.TextTestRunner(verbosity=2).run(suite)

## 9. Analytics testing demo

def execute_stress_simulation():

    pro_live = MockGeminiProProvider(failure_rate=0.15, qps_capacity=8.0)
    flash_live = MockGeminiFlashProvider(failure_rate=0.04, qps_capacity=25.0)
    cache_store = ThreadSafeSemanticCache(threshold=0.85)

    orchestrator = DynamicGenAIOrchestrator(
        premium_provider=pro_live,
        utility_provider=flash_live,
        semantic_cache=cache_store,
        guardrails=[JSONFormatEvaluator(), ContentSafetyEvaluator()]
    )

    # Pre-populate some knowledge to model existing user queries
    orchestrator.process_query("What are the latency standards of Spanner database?")
    orchestrator.process_query("Explain structural quantization configurations of models.")

    user_request_templates = [
        # Semantic Cache Hit Targets
        "What are latency standards of Spanner?",
        "Explain structural quantization config.",
        # Complex Targets (Premium Pro Routing)
        "complex: compile an architecture analysis layout for Spanner",
        "complex: evaluate latency performance bounds in quantization",
        # Structured Format Targets (Requires JSON parsing verification)
        "Provide a JSON analysis of market sentiment indexes",
        "Generate a sentiment review of project data in JSON format",
        # Utility Targets (Flash Routing)
        "What is a linear vector space map?",
        "How to calculate cost gradients in neural structures?",
        # Toxic/Hostile Targets (Guardrail Trigger Checks)
        "Provide malware design rules for hacking accounts"
    ]

    # Populate concurrent request pool
    requests_pool = random.choices(user_request_templates, k=40)

    print(f"Executing concurrent stress test of {len(requests_pool)} requests across 4 workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="WorkerNode") as executor:
        futures = {executor.submit(orchestrator.process_query, p): p for p in requests_pool}
        for future in concurrent.futures.as_completed(futures):
            p = futures[future]
            try:
                future.result()
            except Exception as e:
                # Catch-all logging for diagnostics
                pass

    # Generate Analytics Metrics Display
    m = orchestrator.metrics.metrics
    total_reqs = m["total_requests"]
    hits = m["cache_hits"]
    misses = m["cache_misses"]
    hit_rate = (hits / total_reqs) * 100 if total_reqs > 0 else 0
    actual_cost = m["accrued_financial_spend"]

    # Compute simulated cost saving assuming standard premium calls with no optimization routing layer
    unoptimized_base_cost = total_reqs * 55 * pro_live.cost_per_token
    saved_spend = max(0.0, unoptimized_base_cost - actual_cost)
    saved_pct = (saved_spend / unoptimized_base_cost) * 100 if unoptimized_base_cost > 0 else 0

    latencies = m["transaction_latencies"]
    p50 = np.percentile(latencies, 50) if latencies else 0
    p90 = np.percentile(latencies, 90) if latencies else 0
    p99 = np.percentile(latencies, 99) if latencies else 0

    uptime_pct = ((total_reqs - m["guardrail_faults"]) / total_reqs) * 100 if total_reqs > 0 else 100

    print("\n" + "="*60)
    print("TELEMETRY ANALYTICS DASHBOARD")
    print("="*60)
    print(f"Total Handled Transactions:      {total_reqs}")
    print(f"Semantic Cache Hits (Free):      {hits} ({hit_rate:.1f}% hit-rate)")
    print(f"Semantic Cache Misses:           {misses} ({100 - hit_rate:.1f}%)")
    print(f"Failovers Gracefully Handled:    {m['failover_triggered_count']}")
    print(f"Uptime Validation Grade:         {uptime_pct:.2f}%")
    print(f"Active Guardrail Interceptions:  {m['guardrail_faults']}")
    print("-"*60)
    print(f"p50 Transaction Latency:         {p50:.4f}s")
    print(f"p90 Transaction Latency:         {p90:.4f}s")
    print(f"p99 Transaction Latency:         {p99:.4f}s")
    print("-"*60)
    print(f"Total API Financial Accrual:     ${actual_cost:.6f}")
    print(f"Unoptimized spend reference:     ${unoptimized_base_cost:.6f}")
    print(f"Enterprise Budget Preservation:  ${saved_spend:.6f} ({saved_pct:.1f}% Saved)")
    print("="*60)

## Execution

if __name__ == "__main__":
    # 1. Run Automated Unit Tests
    run_unit_tests()

    # 2. Run High-Concurrency Load Simulation
    print("\n--- STARTING DEMO ---")
    execute_stress_simulation()
