"""LLM assessment: turns signals + transcript into a rating with evidence."""
from __future__ import annotations

import json
import re

from signals import fmt_time

RATINGS = ["Poor", "Average", "Good", "Excellent"]

# Starting hypotheses from the sample set. Refine them with calibrate.py.
RUBRIC = """\
- Excellent: children are actively involved throughout. The teacher asks frequent questions,
  gives every child repeated turns, praises and encourages, builds warm rapport from the start,
  and keeps the lesson moving even when technical problems happen. Homework is a bonus, not a
  requirement.
- Good: solid structure and participation, but some children get little airtime, or warmth,
  energy or clarity is uneven.
- Average: basic interaction (mostly teacher talk or procedural questions), little rapport, or
  audio/video problems that clearly hurt the lesson.
- Poor: no introduction, little or no student interaction, long unexplained dead air, or the
  camera is off / audio unusable.
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
    lines = [f"[{fmt_time(c.start)}] {c.speaker + ': ' if c.speaker else ''}{c.text}" for c in cues]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n[... middle of transcript omitted for length ...]\n" + text[-half:]


import os

_LABEL_NAMES = {"weak": "Poor", "poor": "Poor", "average": "Average",
                "good": "Good", "excellent": "Excellent"}


def load_reference() -> dict:
    """Mean signals per tier from the labelled sample set (written by calibrate.py)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_profiles.json")
    try:
        with open(path) as f:
            raw = json.load(f)
        return {_LABEL_NAMES.get(k.lower(), k): v for k, v in raw.items()}
    except Exception:  # noqa: BLE001
        return {}


def build_prompt(result: dict, max_chars: int = 150_000) -> str:
    audio = {k: v for k, v in result["audio"].items() if k != "speech_per_min"}
    video = {k: v for k, v in result["video"].items() if v is not None}
    if "face_ratio" not in video:
        video["face_detection"] = "unavailable (not measured; infer nothing from it)"
    signals = {"audio": audio, "video": video,
               "transcript": {k: v for k, v in result["text"].items()}}
    flags = "\n".join(f"- ({f['kind']}) {f['msg']}" for f in result["flags"]) or "- none"
    transcript = format_transcript(result["cues"], max_chars) or "(no transcript available)"
    ref = load_reference()
    reference_block = ""
    if ref:
        reference_block = (
            "Reference: average signal values per tier in the labelled sample set "
            "(n = recordings per tier). Use them for context on how this class compares; "
            "they are not hard cutoffs.\n" + json.dumps(ref, indent=2) + "\n\n")
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
- Speaker labels are pseudonymised: TEACHER and STUDENT_n. Name-call counts are undercounts
  because speech-to-text mangles names.
- Tone and energy cannot be judged from text. Use audio signals only as weak hints and say
  when you are unsure instead of asserting "low energy" or "monotone".
- frozen_ratio is also high when children sit still or a slide is static, and sharpness is
  driven by on-screen content, so do not read either as camera quality without other evidence.
- Only state that something happened (a reminder, a rule, a breach) if you can point to the
  transcript line. Never assume.
- Technical problems the teacher handles well are not a teaching fault; judge how she recovered.
- A missing homework mention alone must not lower the rating by more than one step.
- Credit positives you can see: praise, giving every child a turn, questions, clear explanations.
- Do not invent observations. If the transcript is missing or thin, lower your confidence.
- Do not reproduce long passages of the transcript; cite a timestamp and paraphrase.

{reference_block}Computed signals (JSON):
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

# Non-chat models to hide from the picker.
_SKIP = ("whisper", "tts", "guard", "safeguard", "embed", "orpheus", "playai", "moderation")
_PREFERRED = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
              "llama-3.1-8b-instant"]


def list_models(api_key: str, base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """Ask the provider which models this key can actually use (chat models only)."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key)
    ids = sorted(m.id for m in client.models.list().data)
    return [i for i in ids if not any(k in i.lower() for k in _SKIP)]


def pick_default(models: list[str], preferred: str | None = None) -> int:
    """Index of the best default in `models`."""
    for name in ([preferred] if preferred else []) + _PREFERRED:
        if name in models:
            return models.index(name)
    return 0


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
