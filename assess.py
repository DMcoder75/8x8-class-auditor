"""LLM assessment: turns signals + transcript into a rating with evidence."""
from __future__ import annotations

import json
import re

from signals import fmt_time

RATINGS = ["Poor", "Average", "Good", "Excellent"]

# Starting hypotheses from the sample set. Refine them with calibrate.py.
RUBRIC = """\
- Excellent: calls students by name individually, asks many questions, builds rapport at the start, sets homework.
- Good: sets homework, includes every student, solid intro, but the delivery could be happier or warmer.
- Average: picture has clarity or lighting problems, interaction is basic, or the teacher's energy is low.
- Poor: jumps straight into the work with no intro, sets no homework, or the camera is off.
"""

SCHEMA_HINT = """\
Return ONLY a JSON object with exactly these keys:
{
  "rating": "Poor" | "Average" | "Good" | "Excellent",
  "confidence": "low" | "medium" | "high",
  "summary": "2-3 sentences justifying the rating",
  "evidence": [
    {"observation": "specific, concrete observation",
     "timestamp": "mm:ss or n/a",
     "signal": "transcript" | "audio" | "visual",
     "impact": "positive" | "negative" | "neutral"}
  ],
  "rubric_check": {
    "intro_rapport": "...", "student_name_usage": "...", "interaction": "...",
    "homework": "...", "energy_tone": "...", "camera_background": "...", "privacy": "..."
  },
  "improvements": ["specific, actionable recommendation", "..."],
  "signal_caveats": "which computed signals you trusted or discounted, and why"
}
Give 5-8 evidence items. Cite timestamps from the transcript wherever possible."""


def format_transcript(cues, max_chars: int = 150_000) -> str:
    lines = [f"[{fmt_time(c.start)}] {c.text}" for c in cues]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n[... middle of transcript omitted for length ...]\n" + text[-half:]


def build_prompt(result: dict, max_chars: int = 150_000) -> str:
    audio = {k: v for k, v in result["audio"].items() if k != "speech_per_min"}
    signals = {"audio": audio, "video": result["video"],
               "transcript": {k: v for k, v in result["text"].items()}}
    flags = "\n".join(f"- ({f['kind']}) {f['msg']}" for f in result["flags"]) or "- none"
    transcript = format_transcript(result["cues"], max_chars) or "(no transcript available)"
    return f"""You are a senior quality auditor for 8x8, an online education company that teaches children.
Assess ONE recorded class and rate it Poor, Average, Good or Excellent.

Rubric (starting hypotheses):
{RUBRIC}
How to use the inputs:
- The timestamped transcript is your primary evidence. Names of students appear as [STUDENT_n].
- The computed signals below are noisy hints, NOT ground truth. They come from simple
  heuristics (a face may be missed, silence can be children working quietly, a keyword match can
  be the teacher *enforcing* a privacy rule, which is good). Say when you discount one.
- A personal-information mention is only a problem if the teacher or a child actually shares
  private details. A teacher reminding children not to share them is positive.
- Do not invent observations. If the transcript is missing or thin, lower your confidence.
- Do not reproduce long passages of the transcript; cite a timestamp and paraphrase.

Computed signals (JSON):
{json.dumps(signals, indent=2)}

Rule-based flags:
{flags}

Transcript source: {result['transcript_source']}
Transcript:
{transcript}

{SCHEMA_HINT}"""


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    return json.loads(raw)


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"   # the /openai/v1 part is required
DEFAULT_MODEL = "llama-3.3-70b-versatile"
# Free tiers cap tokens per minute, so retry with a shorter transcript if rejected.
TRANSCRIPT_SIZES = [60_000, 30_000, 15_000]


def _is_size_or_rate_error(msg: str) -> bool:
    msg = msg.lower()
    return any(k in msg for k in ("413", "429", "too large", "rate_limit", "tokens per minute",
                                  "request too", "context length", "reduce your"))


def assess(result: dict, api_key: str, model: str = DEFAULT_MODEL,
           base_url: str = DEFAULT_BASE_URL) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    last_error = None
    for max_chars in TRANSCRIPT_SIZES:
        prompt = build_prompt(result, max_chars)
        try:
            try:
                response = client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"})
            except Exception as e:  # provider without JSON mode: retry plain
                if "response_format" in str(e).lower() or "json" in str(e).lower():
                    response = client.chat.completions.create(
                        model=model, temperature=0,
                        messages=[{"role": "user", "content": prompt}])
                else:
                    raise
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            if not _is_size_or_rate_error(str(e)):
                raise
    else:
        raise RuntimeError(f"Transcript still too large for this model's limits: {last_error}")

    data = _parse_json(response.choices[0].message.content)
    rating = str(data.get("rating", "")).strip().title()
    data["rating"] = rating if rating in RATINGS else "Unrated"
    data.setdefault("confidence", "low")
    data.setdefault("evidence", [])
    data.setdefault("improvements", [])
    data.setdefault("rubric_check", {})
    return data
