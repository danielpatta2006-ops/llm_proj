"""Analytics, final report and personalised learning recommendations."""
from collections import defaultdict
import pandas as pd
from .llm import LLMClient, LLMError
from .evaluator import weighted_overall
from . import prompts


def weak_topics_from_history(rows, threshold=6.0, limit=4):
    agg = defaultdict(list)
    for r in rows:
        if r["topic"]:
            agg[r["topic"]].append(r["score"])
    avg = {t: sum(v) / len(v) for t, v in agg.items()}
    return [t for t, a in sorted(avg.items(), key=lambda x: x[1]) if a < threshold][:limit]


def criterion_averages(evals):
    keys = ["technical_accuracy", "relevance", "completeness", "clarity", "communication"]
    return {k: round(sum(e["scores"][k] for e in evals) / len(evals), 1) for k in keys} if evals else {}


def build_report(llm, role, items):
    """items: list of dicts {question, topic, type, score, eval}. Returns report dict."""
    evals = [i["eval"] for i in items]
    overall = round(sum(i["score"] for i in items) / len(items), 1) if items else 0.0
    crit = criterion_averages(evals)
    missing = sorted({m for e in evals for m in e.get("missing_concepts", [])})
    rows = [{"q": i["question"], "topic": i["topic"], "score": i["score"],
             "missing": i["eval"].get("missing_concepts", [])} for i in items]
    ai = None
    if not llm.demo:
        try:
            ai = llm.generate_json(prompts.report_prompt(role, rows), 0.4)
        except LLMError:
            ai = None
    if not ai:
        weak = [i["topic"] for i in sorted(items, key=lambda x: x["score"])[:3]]
        label = "excellent" if overall >= 8 else "good" if overall >= 6 else "developing"
        ai = {"summary": f"Overall performance is {label} ({overall}/10). Focus on the weakest topics below.",
              "strengths": [k.replace("_", " ").title() for k, v in crit.items() if v >= 6],
              "weaknesses": [k.replace("_", " ").title() for k, v in crit.items() if v < 6],
              "study_topics": [{"topic": t, "why": "Lowest scores this session",
                                "resources": ["Official documentation", "freeCodeCamp", "NPTEL"]} for t in weak],
              "seven_day_plan": [f"Day {d}: revise {weak[(d - 1) % len(weak)]} and re-answer its question aloud"
                                 for d in range(1, 8)] if weak else []}
    ai.update({"overall": overall, "criteria": crit, "missing_concepts": missing})
    return ai


def history_frame(interviews):
    df = pd.DataFrame(interviews)
    if df.empty:
        return df
    df["started_at"] = pd.to_datetime(df["started_at"])
    return df.sort_values("started_at")
