"""Question generation + answer evaluation modules (LLM with robust offline fallback)."""
import re
from .config import WEIGHTS
from .llm import LLMClient, LLMError
from . import prompts

DEMO_QUESTIONS = {
    "Technical": [
        ("Explain the difference between a list and a tuple in Python.", "Python basics",
         ["mutability", "hashability", "use cases"]),
        ("What is the difference between a process and a thread?", "Operating systems",
         ["memory space", "context switching", "concurrency"]),
        ("Explain INNER JOIN versus LEFT JOIN with an example.", "SQL", ["matching rows", "NULLs", "example"]),
        ("What is overfitting and how can you reduce it?", "Machine learning",
         ["training vs test error", "regularisation", "cross-validation"]),
        ("How does a hash map work internally?", "Data structures", ["hashing", "collisions", "complexity"]),
    ],
    "HR": [
        ("Tell me about yourself.", "Self-introduction", ["education", "skills", "goal"]),
        ("Describe a conflict in a team project and how you resolved it.", "Teamwork",
         ["situation", "action", "result"]),
        ("Why should we hire you?", "Motivation", ["skills fit", "evidence", "enthusiasm"]),
        ("What is your biggest weakness?", "Self-awareness", ["honesty", "improvement steps", "example"]),
        ("Where do you see yourself in five years?", "Career goals", ["growth", "realism", "company fit"]),
    ]}


def _demo_questions(category, n):
    pool = []
    if category in ("Technical", "Mixed"):
        pool += [("Technical", *q) for q in DEMO_QUESTIONS["Technical"]]
    if category in ("HR", "Mixed"):
        pool += [("HR", *q) for q in DEMO_QUESTIONS["HR"]]
    if category == "Mixed":  # interleave
        t = [p for p in pool if p[0] == "Technical"]; h = [p for p in pool if p[0] == "HR"]
        pool = [x for pair in zip(t, h) for x in pair]
    return [{"question": q, "type": t, "topic": tp, "expected_concepts": ec, "ideal_answer_hint": ""}
            for t, q, tp, ec in pool[:n]]


def generate_questions(llm, role, category, difficulty, n, profile, resume_analysis, weak_topics, asked=()):
    if not llm.demo:
        try:
            data = llm.generate_json(prompts.question_prompt(
                role, category, difficulty, n, profile, resume_analysis, weak_topics, list(asked)), 0.7)
            qs = [q for q in data.get("questions", []) if q.get("question")]
            if qs:
                for q in qs:
                    q.setdefault("type", "Technical"); q.setdefault("topic", "General")
                    q.setdefault("expected_concepts", [])
                return qs[:n]
        except LLMError:
            pass
    return _demo_questions(category, n)


def weighted_overall(scores):
    """Weighted 0-10 score from the five criteria."""
    total = sum(float(scores.get(k, 0)) * w for k, w in WEIGHTS.items())
    return round(max(0.0, min(10.0, total)) + 1e-9, 1)  # half-up


def _clamp(scores):
    return {k: max(0, min(10, int(round(float(scores.get(k, 0)))))) for k in WEIGHTS}


def heuristic_evaluation(question, answer):
    """Offline evaluator: length, concept coverage, sentence quality. For demo/fallback only."""
    words = re.findall(r"\w+", answer.lower())
    if len(words) < 4:
        s = dict(technical_accuracy=0, relevance=1, completeness=0, clarity=1, communication=1)
        missing = question.get("expected_concepts", [])
    else:
        concepts = question.get("expected_concepts", [])
        hit = [c for c in concepts if any(w in answer.lower() for w in c.lower().split()[:2])]
        cov = len(hit) / len(concepts) if concepts else 0.5
        missing = [c for c in concepts if c not in hit]
        length = min(1.0, len(words) / 60)
        sentences = max(1, len(re.findall(r"[.!?]", answer)))
        clarity = 4 + min(4, sentences) + (1 if 12 <= len(words) / sentences <= 25 else 0)
        s = dict(technical_accuracy=round(2 + 8 * cov * 0.9), relevance=round(3 + 6 * max(cov, 0.4)),
                 completeness=round(10 * (0.6 * cov + 0.4 * length)), clarity=min(10, clarity),
                 communication=min(10, round(4 + 5 * length)))
    s = _clamp(s)
    return {"scores": s, "overall": weighted_overall(s),
            "reasoning_summary": "Demo-mode heuristic based on concept coverage and answer length.",
            "missing_concepts": missing, "strengths": ["Answer provided"] if sum(s.values()) > 10 else [],
            "weaknesses": ["Too brief or off-topic"] if sum(s.values()) <= 15 else [],
            "improved_answer": "", "grammar_feedback": "Enable the Groq API for grammar feedback.",
            "suggestion": "Cover: " + ", ".join(missing) if missing else "Add a concrete example."}


def evaluate_answer(llm, question, answer):
    answer = (answer or "").strip()
    if not llm.demo and answer:
        try:
            r = llm.generate_json(prompts.evaluation_prompt(question, answer), temperature=0.2)
            s = _clamp(r.get("scores", {}))
            r["scores"] = s
            r["overall"] = weighted_overall(s)  # always recompute: never trust model arithmetic
            for k, d in [("missing_concepts", []), ("strengths", []), ("weaknesses", []),
                         ("reasoning_summary", ""), ("improved_answer", ""),
                         ("grammar_feedback", ""), ("suggestion", "")]:
                r.setdefault(k, d)
            return r
        except LLMError:
            pass
    return heuristic_evaluation(question, answer)
