"""Resume processing module: PDF text extraction + LLM analysis (with offline fallback)."""
import re
from io import BytesIO
from .llm import LLMClient, LLMError
from . import prompts

SKILL_BANK = ["python", "java", "c++", "c", "javascript", "typescript", "react", "node", "django", "flask",
              "sql", "mysql", "mongodb", "html", "css", "machine learning", "deep learning", "nlp", "pandas",
              "numpy", "tensorflow", "pytorch", "excel", "power bi", "tableau", "git", "docker", "aws",
              "linux", "data structures", "algorithms", "streamlit", "langchain", "prompt engineering"]


def extract_pdf_text(file_bytes):
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(file_bytes))
    text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    if len(text) < 50:
        raise ValueError("Could not read text from this PDF (it may be a scanned image).")
    return text


def heuristic_analysis(text):
    low = text.lower()
    skills = [s.title() if len(s) > 3 else s.upper() for s in SKILL_BANK
              if re.search(r"(?<![a-z])" + re.escape(s) + r"(?![a-z])", low)]
    projects = [l.strip()[:80] for l in text.splitlines() if "project" in l.lower()][:4]
    score = min(10, 3 + len(skills) // 2 + (2 if projects else 0))
    return {"skills": skills, "qualifications": [], "experience": [], "projects": projects,
            "strengths": ["Skills detected by keyword matching (demo mode)"],
            "gaps": [], "resume_score": score,
            "suggestions": ["Add measurable achievements", "List projects with technologies used"]}


def analyse_resume(text, llm: LLMClient):
    if llm.demo:
        return heuristic_analysis(text)
    try:
        return llm.generate_json(prompts.resume_prompt(text), temperature=0.2)
    except LLMError:
        return heuristic_analysis(text)
