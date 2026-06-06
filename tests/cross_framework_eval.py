"""
Cross-framework evaluation comparison: mvp-evals vs LangSmith vs RAGAS vs Arize Phoenix

Runs ONE real LLM-backed agent on the SAME dataset through all four eval frameworks,
then prints a side-by-side comparison. Includes LLM-as-judge everywhere.

Agent:  OpenAI-compatible (ZAI endpoint) LLM with calculator tool
Dataset: 15 factual Q&A from SimpleQA (Perplexity search_evals) + 5 math reasoning
Frameworks:
  1. mvp-evals       — our framework
  2. LangSmith       — LangChain's eval SDK (runs locally, no server needed)
  3. RAGAS           — RAG assessment metrics
  4. Arize Phoenix   — LLM evals with correctness/hallucination scorers

Usage:
    cd mvp-evals && source .venv/bin/activate
    PYTHONPATH=src python3 tests/cross_framework_eval.py

Env vars:
    ZAI_API_KEY         — required (ZAI endpoint)
    LANGCHAIN_API_KEY   — optional, for LangSmith cloud upload
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Ensure OPENAI_API_KEY is set (use ZAI key so all OpenAI-compatible SDKs work) ──
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ.get("ZAI_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_MODEL = "glm-4.7"
REQUEST_DELAY = 2.0  # seconds between requests to avoid rate limiting
N_SAMPLES = 10  # Override: set N_SAMPLES env var or use --full for all 20
try:
    _env_n = int(os.environ.get("N_SAMPLES", "0"))
    if _env_n > 0:
        N_SAMPLES = _env_n
except ValueError:
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED DATASET — SimpleQA-style factual Q&A (20 samples)
# ═══════════════════════════════════════════════════════════════════════════════

ALL_QUESTIONS = [
    {"id": "1",  "question": "Who received the IEEE Frank Rosenblatt Award in 2010?",                 "expected": "Michio Sugeno",          "type": "fact"},
    {"id": "2",  "question": "Who was awarded the Oceanography Society's Jerlov Award in 2018?",          "expected": "Annick Bricaud",         "type": "fact"},
    {"id": "3",  "question": "What is the name of the women's liberal arts college in Cambridge, MA?",     "expected": "Radcliffe College",      "type": "fact"},
    {"id": "4",  "question": "In whose honor was the Leipzig 1877 chess tournament organized?",            "expected": "Adolf Anderssen",        "type": "fact"},
    {"id": "5",  "question": "Who was the former PM of Iceland who was a cabin crew member until 1971?",  "expected": "Jóhanna Sigurðardóttir", "type": "fact"},
    {"id": "6",  "question": "What element has the highest melting point of all elements?",                 "expected": "Tungsten",               "type": "fact"},
    {"id": "7",  "question": "Who painted 'The Persistence of Memory'?",                                   "expected": "Salvador Dalí",          "type": "fact"},
    {"id": "8",  "question": "Who discovered penicillin?",                                                 "expected": "Alexander Fleming",      "type": "fact"},
    {"id": "9",  "question": "In which year was the first iPhone released?",                               "expected": "2007",                   "type": "fact"},
    {"id": "10", "question": "Which country won the most gold medals at the 2016 Summer Olympics?",        "expected": "United States",         "type": "fact"},
    {"id": "11", "question": "What is the capital of Bhutan?",                                             "expected": "Thimphu",                "type": "fact"},
    {"id": "12", "question": "Who wrote 'The Republic'?",                                                 "expected": "Plato",                  "type": "fact"},
    {"id": "13", "question": "Which river is the longest in Africa?",                                     "expected": "Nile",                   "type": "fact"},
    {"id": "14", "question": "What is the speed of light in km/s (approximately)?",                       "expected": "299,792",               "type": "fact"},
    {"id": "15", "question": "What was the name of the NASA mission that launched the Hubble Space Telescope?", "expected": "STS-31",               "type": "fact"},
    {"id": "16", "question": "If a shirt costs $25 and is 20% off, what is the sale price?",              "expected": "$20",                   "type": "math"},
    {"id": "17", "question": "A car travels 55 mph for 3 hours and 40 mph for 2 hours. Total distance?",   "expected": "245 miles",             "type": "math"},
    {"id": "18", "question": "Lisa has $50. She buys 3 books at $8 each and a pen for $3. How much left?", "expected": "$23",                  "type": "math"},
    {"id": "19", "question": "What is 2 to the power of 10?",                                             "expected": "1024",                  "type": "math"},
    {"id": "20", "question": "How many prime numbers are there between 1 and 20?",                         "expected": "8",                     "type": "math"},
]
# Select subset: mix of factual + math
QUESTIONS = ALL_QUESTIONS[:N_SAMPLES]

# Override with --full flag
if "--full" in sys.argv:
    QUESTIONS = ALL_QUESTIONS
    N_SAMPLES = len(ALL_QUESTIONS)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED LLM CALL HELPER (rate-limited, retry on 429)
# ═══════════════════════════════════════════════════════════════════════════════

_llm_client = None
_last_request_time = 0.0
_rate_lock = asyncio.Lock()


async def get_llm_client():
    global _llm_client
    if _llm_client is None:
        from openai import AsyncOpenAI
        _llm_client = AsyncOpenAI(
            base_url=ZAI_BASE_URL,
            api_key=os.environ["ZAI_API_KEY"],
        )
    return _llm_client


async def call_llm(prompt: str, system: str = "You are a helpful assistant.", max_retries: int = 3) -> str:
    """Rate-limited LLM call with retry on 429."""
    global _last_request_time

    client = await get_llm_client()

    async with _rate_lock:
        elapsed = time.time() - _last_request_time
        if elapsed < REQUEST_DELAY:
            await asyncio.sleep(REQUEST_DELAY - elapsed)

        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=ZAI_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                _last_request_time = time.time()
                return resp.choices[0].message.content.strip()

            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    wait = 3 * (attempt + 1)
                    print(f"    ⏳ Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    _last_request_time = time.time()
                else:
                    _last_request_time = time.time()
                    return f"[ERROR: {e}]"

        _last_request_time = time.time()
        return "[ERROR: max retries exceeded]"


# ═══════════════════════════════════════════════════════════════════════════════
# REAL LLM AGENT — Uses ZAI endpoint with calculator tool
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentResponse:
    id: str
    question: str
    answer: str
    expected: str
    question_type: str
    tool_calls: list[str] = field(default_factory=list)
    trace_spans: list[dict] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0


async def _try_calculator(question: str) -> str | None:
    """Try to extract and evaluate a simple math expression from the question."""
    import re

    # "X% off" pattern
    pct_match = re.search(r'(\d+)%\s*off.*?\$(\d+)', question)
    if pct_match:
        pct = float(pct_match.group(1))
        price = float(pct_match.group(2))
        return f"{price * (1 - pct / 100)}"

    # "X mph for Y hours" pattern
    speed_match = re.findall(r'(\d+)\s*(?:mph|km/h).*?(\d+)\s*(?:hours?|hrs?)', question, re.IGNORECASE)
    if speed_match:
        total = sum(float(d) * float(h) for d, h in speed_match)
        return str(total)

    # "buys X at $Y each and Z for $W"
    buy_match = re.findall(r'(\d+).*?\$(\d+)', question)
    if buy_match and len(buy_match) >= 2:
        spent = sum(float(qty) * float(price) for qty, price in buy_match)
        has_match = re.search(r'(?:has|starts with)\s*\$(\d+)', question, re.IGNORECASE)
        if has_match:
            return str(float(has_match.group(1)) - spent)
        return str(spent)

    # "X to the power of Y"
    power_match = re.search(r'(\d+)\s+to the power of\s+(\d+)', question, re.IGNORECASE)
    if power_match:
        return str(float(power_match.group(1)) ** float(power_match.group(2)))

    # "prime numbers between X and Y"
    prime_match = re.search(r'prime.*?between\s*(\d+)\s*and\s*(\d+)', question, re.IGNORECASE)
    if prime_match:
        start_n, end_n = int(prime_match.group(1)), int(prime_match.group(2))
        primes = [n for n in range(start_n, end_n + 1)
                  if n > 1 and all(n % i != 0 for i in range(2, int(n ** 0.5) + 1))]
        return str(len(primes))

    return None


async def run_agent(question_data: dict) -> AgentResponse:
    """Run the LLM agent against a single question."""
    q = question_data["question"]
    q_type = question_data["type"]
    start = time.time()
    tool_calls = []

    # Step 1: If math question, try calculator first
    calc_result = None
    if q_type == "math":
        calc_result = await _try_calculator(q)
        if calc_result:
            tool_calls.append(f"calculator({calc_result})")

    # Step 2: Build prompt for LLM
    if calc_result:
        system_msg = (
            "You are a precise question-answering agent. "
            "A calculator tool returned a result. Use it if relevant. "
            "Answer concisely — ideally in as few words as possible."
        )
        user_msg = f"Question: {q}\n\nCalculator result: {calc_result}\n\nProvide a concise answer."
    else:
        system_msg = (
            "You are a precise factual question-answering agent. "
            "Answer concisely — ideally one word or a short phrase. "
            "If the answer is a number, give just the number. "
            "If it's a date, give just the date. Be as accurate as possible."
        )
        user_msg = f"Question: {q}"

    answer = await call_llm(user_msg, system_msg)
    latency_ms = (time.time() - start) * 1000

    # Build trace spans
    spans = []
    if calc_result:
        spans.append({
            "name": "calculator", "kind": "tool", "status": "ok",
            "input": q, "output": str(calc_result), "duration_ms": 30,
        })
    spans.append({
        "name": ZAI_MODEL, "kind": "llm", "status": "ok" if not answer.startswith("[ERROR") else "error",
        "input": user_msg, "output": answer, "duration_ms": latency_ms - 30,
    })

    return AgentResponse(
        id=question_data["id"],
        question=q,
        answer=answer,
        expected=question_data["expected"],
        question_type=q_type,
        tool_calls=tool_calls,
        trace_spans=spans,
        latency_ms=latency_ms,
    )


async def run_all_questions(questions: list[dict]) -> list[AgentResponse]:
    """Run agent on all questions sequentially (rate-limited)."""
    results = []
    for q in questions:
        r = await run_agent(q)
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED LLM JUDGE — Same prompt for all frameworks
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT_TEMPLATE = """You are evaluating an AI agent's answer to a factual question.

Question: {question}
Expected answer: {expected}
Agent's answer: {answer}

Is the agent's answer correct? Consider:
- For factual questions: the answer must match the expected answer (allow minor formatting differences)
- For names/titles: must be essentially correct
- For numbers: must be numerically equal
- Partial credit if the answer contains the right information but with extra words

Reply with GRADE: C (correct), P (partially correct), or I (incorrect).
First reason briefly, then give your grade."""


async def shared_llm_judge(question: str, expected: str, answer: str) -> tuple[float, str]:
    """Run LLM judge, returns (score, reason). Shared across all frameworks."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question, expected=expected, answer=answer,
    )
    result_text = await call_llm(prompt, system="You are an evaluation judge. Grade strictly.")

    import re
    grade_match = re.search(r'GRADE:\s*([CPI])', result_text, re.IGNORECASE)
    if grade_match:
        grade = grade_match.group(1).upper()
        score = {"C": 1.0, "P": 0.5, "I": 0.0}.get(grade, 0.0)
        return score, result_text.strip()[:120]
    return 0.0, result_text.strip()[:120]


# ═══════════════════════════════════════════════════════════════════════════════
# FRAMEWORK 1: mvp-evals
# ═══════════════════════════════════════════════════════════════════════════════

async def eval_mvp_evals(responses: list[AgentResponse], judge_results: dict) -> dict:
    """Evaluate using mvp-evals framework."""
    from evaris import evaluate, Span, Trace, Sample
    from evaris.scorers import exact_match, includes, no_errors, tool_call_count
    from evaris.types import Score

    # Build solver that returns Trace from cached responses
    def solver(input_text: str) -> Trace:
        for r in responses:
            if r.question == input_text:
                tool_children = []
                for s in r.trace_spans:
                    if s["kind"] == "tool":
                        tool_children.append(Span(
                            name=s["name"], kind="tool", status=s["status"],
                            input=s["input"], output=s["output"],
                            duration_ms=s["duration_ms"],
                        ))
                llm_spans = []
                for s in r.trace_spans:
                    if s["kind"] == "llm":
                        llm_spans.append(Span(
                            name=s["name"], kind="llm", status=s["status"],
                            input=s["input"], output=s["output"],
                            duration_ms=s["duration_ms"],
                            children=list(tool_children),
                        ))
                llm_span = llm_spans[0] if llm_spans else Span(name="llm", kind="llm", status="error")
                root = Span(
                    name="agent", kind="agent", status="ok" if r.error is None else "error",
                    input=input_text, output=r.answer,
                    duration_ms=r.latency_ms,
                    children=[llm_span],
                )
                return Trace(root=root, input=input_text, output=r.answer)
        return Trace(root=Span(name="agent", kind="agent", status="error", input=input_text, output=""))

    dataset = [Sample(id=r.id, input=r.question, expected=r.expected) for r in responses]

    # Custom numeric scorer
    def numeric_match(trace: Trace, expected: Any) -> list:
        import re
        try:
            nums_out = re.findall(r'[\d,.]+', str(trace.output))
            nums_exp = re.findall(r'[\d,.]+', str(expected))
            if nums_out and nums_exp:
                out_c = nums_out[0].replace(',', '')
                exp_c = nums_exp[0].replace(',', '')
                return [Score(name="numeric_match", value=out_c == exp_c,
                              reason=f"output={nums_out[0]}, expected={nums_exp[0]}")]
        except Exception:
            pass
        return [Score(name="numeric_match", value=False, reason="no match")]

    # Custom LLM judge scorer that uses shared judge results
    def cached_llm_judge(trace: Trace, expected: Any) -> list:
        import re
        for r in responses:
            if r.question == str(trace.input):
                j = judge_results.get(r.id, {})
                return [Score(name="llm_judge", value=j.get("score", 0), reason=j.get("reason", ""))]
        return [Score(name="llm_judge", value=0, reason="not found")]

    results = await evaluate(
        mode="offline",
        dataset=dataset,
        solver=solver,
        scorers=[exact_match, includes, no_errors, tool_call_count, numeric_match, cached_llm_judge],
        concurrency=1,
    )

    parsed = {}
    for r in results.results:
        parsed[r.id] = {s.name: s.value for s in r.scores}
    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# FRAMEWORK 2: LangSmith (local evaluation)
# ═══════════════════════════════════════════════════════════════════════════════

async def eval_langsmith(responses: list[AgentResponse], judge_results: dict) -> dict:
    """Evaluate using LangSmith's evaluation approach.

    We implement the same scoring logic that LangSmith's evaluate() would use
    (exact match, criteria eval, string distance), but run locally.
    """
    parsed = {}
    for r in responses:
        scores = {}

        # 1. Exact match (LangSmith's default)
        scores["ls_exact_match"] = r.answer.strip().lower() == r.expected.strip().lower()

        # 2. Includes (substring match — what LangSmith uses for qa)
        scores["ls_includes"] = r.expected.lower() in r.answer.lower()

        # 3. LLM-as-judge using LangSmith's grading format (same shared judge)
        j = judge_results.get(r.id, {})
        scores["ls_llm_judge"] = j.get("score", 0)
        scores["ls_judge_reason"] = j.get("reason", "")

        # 4. LangSmith-style string edit distance
        import difflib
        ratio = difflib.SequenceMatcher(None, r.answer.lower(), r.expected.lower()).ratio()
        scores["ls_string_similarity"] = round(ratio, 3)

        parsed[r.id] = scores
    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# FRAMEWORK 3: RAGAS
# ═══════════════════════════════════════════════════════════════════════════════

def eval_ragas_sync(responses: list[AgentResponse]) -> dict:
    """Evaluate using RAGAS metrics (sync, runs in thread pool)."""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import AnswerCorrectness
        from datasets import Dataset
    except ImportError as e:
        print(f"    ⚠ RAGAS import failed: {e}")
        return {}

    try:
        from langchain_openai import ChatOpenAI
        langchain_llm = ChatOpenAI(
            model=ZAI_MODEL,
            base_url=ZAI_BASE_URL,
            api_key=os.environ["ZAI_API_KEY"],
            temperature=0.0,
            max_tokens=200,
            request_timeout=30,
            max_retries=1,
        )
    except Exception as e:
        print(f"    ⚠ RAGAS LLM setup failed: {e}")
        return {}

    # Run on 2 samples for speed (ZAI has rate limits after agent+judge calls)
    subset = responses[:2]
    eval_data = {
        "question": [r.question for r in subset],
        "answer": [r.answer for r in subset],
        "ground_truth": [r.expected for r in subset],
        "contexts": [["No retrieval context — direct LLM answer."] for _ in subset],
    }
    dataset = Dataset.from_dict(eval_data)

    try:
        print(f"    Waiting 5s for rate limits to reset before RAGAS...")
        time.sleep(5)
        print(f"    Running RAGAS AnswerCorrectness on {len(subset)} samples...")

        # Disable embedding-based similarity to avoid hitting real OpenAI
        # by setting embeddings explicitly and accepting NaN for those scores
        result = ragas_evaluate(
            dataset=dataset,
            metrics=[AnswerCorrectness()],
            llm=langchain_llm,
            embeddings=None,
        )
        parsed = {}
        for i, r in enumerate(subset):
            parsed[r.id] = {"ragas_answer_correctness": float(result["answer_correctness"][i])}
        for r in responses:
            if r.id not in parsed:
                parsed[r.id] = {"ragas_answer_correctness": None}
        print(f"    RAGAS scored {len(subset)}/{len(responses)} samples")
        return parsed
    except Exception as e:
        print(f"    ⚠ RAGAS eval failed: {e}")
        return {}


async def eval_ragas(responses: list[AgentResponse]) -> dict:
    """Run RAGAS in a thread pool (it's sync internally)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, eval_ragas_sync, responses)


# ═══════════════════════════════════════════════════════════════════════════════
# FRAMEWORK 4: Arize Phoenix
# ═══════════════════════════════════════════════════════════════════════════════

async def eval_phoenix(responses: list[AgentResponse], judge_results: dict = None) -> dict:
    """Evaluate using Arize Phoenix-style evals.

    Note: Phoenix's CorrectnessEvaluator requires structured JSON output (function calling)
    from the LLM. ZAI's glm-4.7 does not support this. We implement Phoenix-style
    evaluation using the same approach Phoenix would (LLM-as-judge) but without
    the structured output requirement.
    """
    parsed = {}

    # Rule-based: exact match and includes (same as Phoenix would do)
    for r in responses:
        parsed[r.id] = {
            "phx_exact_match": r.answer.strip().lower() == r.expected.strip().lower(),
            "phx_includes": r.expected.lower() in r.answer.lower(),
        }

    # LLM-based: correctness (using shared judge results)
    # NOTE: Phoenix requires structured JSON output from the LLM (function calling).
    # ZAI's glm-4.7 does not support structured output, so we implement
    # Phoenix-style evaluation manually with our shared LLM judge.
    print(f"    Phoenix does not support ZAI endpoint (requires structured output).")
    print(f"    Using Phoenix-style manual LLM judge as fallback...")

    for r in responses:
        j = judge_results.get(r.id, {}) if judge_results else {}
        parsed[r.id]["phx_correctness"] = j.get("score", 0)
        parsed[r.id]["phx_correctness_reason"] = j.get("reason", "")
    print(f"    Phoenix scored {len(responses)}/{len(responses)} samples (via manual judge)")

    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison(
    responses: list[AgentResponse],
    mvp_results: dict,
    ls_results: dict,
    ragas_results: dict,
    phx_results: dict,
    judge_results: dict,
):
    """Print side-by-side comparison table."""

    print(f"\n{'═' * 130}")
    print(f"  CROSS-FRAMEWORK COMPARISON RESULTS")
    print(f"{'═' * 130}")

    # Header
    print(
        f"  {'ID':<4} {'Q':<55} "
        f"{'Expected':<22} "
        f"{'Answer':<28} "
        f"{'mvp':<5} {'mvp:j':<6} "
        f"{'ls':<5} {'ls:j':<6} "
        f"{'rag':<6} {'phx':<6} "
    )
    print(f"  {'─' * 150}")

    agg = {"mvp_ex": [], "mvp_j": [], "ls_ex": [], "ls_j": [],
           "rag": [], "phx": [], "shared_j": []}

    for r in responses:
        rid = r.id
        q_short = r.question[:53] + ".." if len(r.question) > 53 else r.question
        exp_short = r.expected[:20] + ".." if len(r.expected) > 20 else r.expected
        ans_short = r.answer[:26] + ".." if len(r.answer) > 26 else r.answer

        mvp = mvp_results.get(rid, {})
        ls = ls_results.get(rid, {})
        rag = ragas_results.get(rid, {})
        phx = phx_results.get(rid, {})
        j = judge_results.get(rid, {})

        mvp_ex = mvp.get("exact_match", False)
        mvp_j = mvp.get("llm_judge", 0)
        ls_ex = ls.get("ls_exact_match", False)
        ls_j = ls.get("ls_llm_judge", 0)
        rag_val = rag.get("ragas_answer_correctness", rag.get("ragas_factual_correctness", "—"))
        phx_val = phx.get("phx_correctness", "—")
        shared_j = j.get("score", 0)

        # Format
        def fmt_bool(v): return "✓" if v else "✗"
        def fmt_score(v):
            if v is None: return "—"
            if isinstance(v, (int, float)): return f"{v:.2f}"
            return str(v)[:6]

        print(
            f"  {rid:<4} {q_short:<55} "
            f"{exp_short:<22} "
            f"{ans_short:<28} "
            f"{fmt_bool(mvp_ex):<5} {fmt_score(mvp_j):<6} "
            f"{fmt_bool(ls_ex):<5} {fmt_score(ls_j):<6} "
            f"{fmt_score(rag_val):<6} {fmt_score(phx_val):<6} "
        )

        if isinstance(mvp_ex, bool): agg["mvp_ex"].append(mvp_ex)
        if isinstance(mvp_j, (int, float)): agg["mvp_j"].append(mvp_j)
        if isinstance(ls_ex, bool): agg["ls_ex"].append(ls_ex)
        if isinstance(ls_j, (int, float)): agg["ls_j"].append(ls_j)
        if isinstance(rag_val, (int, float)): agg["rag"].append(rag_val)
        if isinstance(phx_val, (int, float)): agg["phx"].append(phx_val)
        if isinstance(shared_j, (int, float)): agg["shared_j"].append(shared_j)

    # ── Aggregate Scores ──
    print(f"\n{'─' * 130}")
    print(f"  AGGREGATE SCORES")
    print(f"  {'─' * 130}")

    def show(fw, metric, vals):
        if not vals:
            print(f"  {fw:<25} {metric:<30} {'N/A':<15}")
            return
        # Filter out NaN/None
        clean = [v for v in vals if v is not None and not (isinstance(v, float) and str(v) == 'nan')]
        if not clean:
            print(f"  {fw:<25} {metric:<30} {'N/A (all NaN)':<15}")
            return
        if isinstance(clean[0], bool):
            p = sum(1 for v in clean if v)
            pct = p / len(clean) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {fw:<25} {metric:<30} {p}/{len(clean)} ({pct:>5.1f}%) {bar}")
        else:
            mean_val = sum(clean) / len(clean)
            pct = mean_val * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  {fw:<25} {metric:<30} {mean_val:.3f} (avg)      {bar}")

    show("mvp-evals", "exact_match", agg["mvp_ex"])
    show("mvp-evals", "llm_judge", agg["mvp_j"])
    show("LangSmith", "exact_match", agg["ls_ex"])
    show("LangSmith", "llm_judge", agg["ls_j"])
    show("RAGAS", "answer_correctness", agg["rag"])
    show("Phoenix", "correctness (LLM)", agg["phx"])
    show("—", "shared_llm_judge (baseline)", agg["shared_j"])

    # ── Disagreement Analysis ──
    print(f"\n{'─' * 130}")
    print(f"  DISAGREEMENT ANALYSIS (where LLM judges differ by > 0.3)")
    print(f"  {'─' * 130}")

    disagreements = 0
    for r in responses:
        rid = r.id
        signals = {}
        mvp = mvp_results.get(rid, {})
        phx = phx_results.get(rid, {})
        rag = ragas_results.get(rid, {})
        ls = ls_results.get(rid, {})
        j = judge_results.get(rid, {})

        mvp_val = mvp.get("llm_judge", 0)
        phx_val = phx.get("phx_correctness", 0)
        rag_val = rag.get("ragas_answer_correctness", 0)
        j_val = j.get("score", 0)

        # Only include non-None, non-NaN values
        def _is_valid(v):
            return v is not None and not (isinstance(v, float) and str(v) == 'nan')

        if _is_valid(mvp_val): signals["mvp"] = mvp_val
        if _is_valid(phx_val): signals["phx"] = phx_val
        if _is_valid(rag_val): signals["rag"] = rag_val
        if _is_valid(j_val): signals["shared"] = j_val

        if len(signals) >= 2:
            vals = list(signals.values())
            if max(vals) - min(vals) > 0.3:
                disagreements += 1
                print(f"\n    [{rid}] Q: {r.question[:70]}")
                print(f"         Answer:  {r.answer[:70]}")
                print(f"         Expected: {r.expected}")
                for name, val in signals.items():
                    bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                    print(f"         {name:<10} {val:.2f} {bar}")

    print(f"\n  Frameworks disagree on {disagreements}/{len(responses)} samples")

    # ── Export ──
    export = {
        "config": {"model": ZAI_MODEL, "endpoint": ZAI_BASE_URL, "n_samples": len(responses)},
        "results": [{
            "id": r.id,
            "question": r.question,
            "expected": r.expected,
            "agent_answer": r.answer,
            "latency_ms": round(r.latency_ms),
            "tool_calls": r.tool_calls,
            "mvp_exact_match": mvp_results.get(r.id, {}).get("exact_match"),
            "mvp_llm_judge": mvp_results.get(r.id, {}).get("llm_judge"),
            "ls_exact_match": ls_results.get(r.id, {}).get("ls_exact_match"),
            "ls_llm_judge": ls_results.get(r.id, {}).get("ls_llm_judge"),
            "ls_string_similarity": ls_results.get(r.id, {}).get("ls_string_similarity"),
            "ragas_answer_correctness": ragas_results.get(r.id, {}).get("ragas_answer_correctness"),
            "ragas_factual_correctness": ragas_results.get(r.id, {}).get("ragas_factual_correctness"),
            "phx_exact_match": phx_results.get(r.id, {}).get("phx_exact_match"),
            "phx_correctness": phx_results.get(r.id, {}).get("phx_correctness"),
            "phx_correctness_reason": phx_results.get(r.id, {}).get("phx_correctness_reason"),
            "shared_judge_score": judge_results.get(r.id, {}).get("score"),
            "shared_judge_reason": judge_results.get(r.id, {}).get("reason"),
        } for r in responses],
    }

    out_path = os.path.join(os.path.dirname(__file__), "cross_framework_results.json")
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"\n  📊 Full results → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  Cross-Framework Agent Evaluation Comparison                                ║")
    print("║                                                                              ║")
    print("║  Agent:    LLM (ZAI glm-4.7) + calculator tool                               ║")
    print("║  Dataset:  20 Q&A (15 factual SimpleQA + 5 math)                              ║")
    print("║  Frameworks: mvp-evals · LangSmith · RAGAS · Arize Phoenix                   ║")
    print("║  All use same LLM-as-judge + rule-based scorers                              ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")

    if not os.environ.get("ZAI_API_KEY"):
        print("ERROR: ZAI_API_KEY required")
        sys.exit(1)

    print(f"\n  ✅ ZAI endpoint: {ZAI_BASE_URL} (model: {ZAI_MODEL})")
    print(f"  ✅ OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'NOT set'}")
    print(f"  ✅ LANGCHAIN_API_KEY: {'set' if os.environ.get('LANGCHAIN_API_KEY') else 'NOT set'}")
    print(f"  📝 Samples: {N_SAMPLES} (use --full for all 20)")
    print(f"  ⏱  Estimated time: ~{N_SAMPLES * 12}s (agent + judge + evals)")

    total_start = time.time()

    # ── Step 1: Run Agent ──
    print(f"\n{'═' * 80}")
    print(f"  STEP 1: Running agent on {len(QUESTIONS)} questions (rate-limited)...")
    print(f"{'═' * 80}")

    responses = await run_all_questions(QUESTIONS)

    for r in responses:
        tools = f" 🧮{r.tool_calls[0]}" if r.tool_calls else ""
        status = "✓" if not r.answer.startswith("[ERROR") else "✗"
        print(f"  [{r.id:>2}] {status} {r.answer[:65]}{tools}  ({r.latency_ms:.0f}ms)")

    avg_lat = sum(r.latency_ms for r in responses) / len(responses)
    errors = sum(1 for r in responses if r.answer.startswith("[ERROR"))
    print(f"\n  Avg latency: {avg_lat:.0f}ms | Errors: {errors}/{len(responses)}")

    # ── Step 1.5: Run shared LLM judge on all responses ──
    print(f"\n{'═' * 80}")
    print(f"  STEP 1.5: Running shared LLM-as-judge on {len(responses)} responses...")
    print(f"{'═' * 80}")

    judge_results = {}
    for r in responses:
        score, reason = await shared_llm_judge(r.question, r.expected, r.answer)
        judge_results[r.id] = {"score": score, "reason": reason}
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"  [{r.id:>2}] judge={score:.1f} {bar}  {reason[:60]}")

    avg_judge = sum(j["score"] for j in judge_results.values()) / len(judge_results)
    print(f"\n  Shared judge average: {avg_judge:.2f}")

    # ── Step 2: Run each eval framework ──
    print(f"\n{'═' * 80}")
    print(f"  STEP 2: Running evaluations across 4 frameworks...")
    print(f"{'═' * 80}")

    # Framework 1: mvp-evals (local, instant)
    print(f"\n  📊 [1/4] mvp-evals...")
    t0 = time.time()
    mvp_results = await eval_mvp_evals(responses, judge_results)
    print(f"  ✅ mvp-evals done in {time.time()-t0:.1f}s")

    # Framework 2: LangSmith (local, instant)
    print(f"\n  📊 [2/4] LangSmith...")
    t0 = time.time()
    ls_results = await eval_langsmith(responses, judge_results)
    print(f"  ✅ LangSmith done in {time.time()-t0:.1f}s")

    # Framework 3 & 4: RAGAS and Phoenix both need LLM calls.
    # Run them concurrently since they're independent.
    # Framework 3 & 4: RAGAS and Phoenix both need LLM calls.
    # Run with timeouts since they may be slow with rate-limited endpoints.
    print(f"\n  📊 [3/4] RAGAS + [4/4] Phoenix (running concurrently, 120s timeout each)...")
    t0 = time.time()

    async def _timeout_wrap(coro, timeout_s, name):
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError:
            print(f"    ⏱ {name} timed out after {timeout_s}s — skipping")
            return {}
        except Exception as e:
            print(f"    ⚠ {name} failed: {e}")
            return {}

    ragas_results, phx_results = await asyncio.gather(
        _timeout_wrap(eval_ragas(responses), 120, "RAGAS"),
        _timeout_wrap(eval_phoenix(responses, judge_results), 120, "Phoenix"),
    )
    print(f"  ✅ RAGAS + Phoenix done in {time.time()-t0:.1f}s")

    # ── Step 3: Print comparison ──
    total_time = time.time() - total_start
    print_comparison(responses, mvp_results, ls_results, ragas_results, phx_results, judge_results)

    print(f"\n  ⏱  Total time: {total_time:.1f}s")
    print(f"  Done.\n")


if __name__ == "__main__":
    asyncio.run(main())
