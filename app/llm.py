"""LLM integration layer: Groq chat completions, JSON parsing and retries."""
import json, re
from .config import GROQ_MODEL, ENV_API_KEY


class LLMError(Exception):
    pass


def extract_json(text):
    """Robustly parse JSON from model output (handles ```json fences and extra prose)."""
    if not text:
        raise LLMError("Empty response")
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    raise LLMError("Could not parse JSON from model output")


class LLMClient:
    def __init__(self, api_key=None, model=None):
        self.api_key = (api_key or ENV_API_KEY or "").strip()
        self.model = model or GROQ_MODEL
        self.demo = not self.api_key
        self._client = None

    def _get(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def generate_json(self, prompt, temperature=0.4, retries=2):
        """Send prompt, require JSON back. Raises LLMError after retries."""
        if self.demo:
            raise LLMError("Demo mode: no API key")
        last = None
        for _ in range(retries + 1):
            try:
                resp = self._get().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON. Do not wrap it in Markdown."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                return extract_json(content)
            except Exception as e:  # network, quota, parse
                last = e
        raise LLMError(str(last))
