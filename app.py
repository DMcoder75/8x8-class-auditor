import importlib.util
import os
import tempfile
import json
import pandas as pd
import streamlit as st

from assess import assess
from signals import analyze

st.set_page_config(page_title="8x8 Class Quality Auditor", layout="wide")


def secret(name: str, default=None):
    """st.secrets raises when no secrets file exists, so wrap it."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


# ---- Optional password gate (recordings are confidential) -------------------
app_password = secret("APP_PASSWORD")
if app_password:
    if st.sidebar.text_input("Access password", type="password") != app_password:
        st.title("8x8 Class Quality Auditor")
        st.info("Enter the access password in the sidebar.")
        st.stop()

st.title("8x8 Class Quality Auditor")
st.caption("Rates an online class recording as Poor, Average, Good or Excellent, "
           "from the transcript, audio and video, and explains why.")

# ---- Sidebar ----------------------------------------------------------------
whisper_available = importlib.util.find_spec("faster_whisper") is not None
with st.sidebar:
    st.header("Input")
    video_file = st.file_uploader("Class recording", type=["mp4", "mov", "m4v", "webm", "mkv"])
    vtt_file = st.file_uploader("Transcript (.vtt), optional", type=["vtt"])
    whisper_model = None
    if whisper_available:
        whisper_model = st.selectbox(
            "Transcription if no .vtt is uploaded", ["tiny", "base", "small"], index=1,
            help="Runs on this server; the audio is not sent anywhere. Slower on free hosting.")
    names_text = st.text_area(
        "Student/teacher names to redact (comma-separated)",
        help="Replaced with [STUDENT_n] before anything is sent to the LLM. "
             "Name-calling can still be counted.")

    st.header("LLM assessment")
    api_key = secret("GEMINI_API_KEY") or st.text_input("Gemini API key", type="password")
    model_name = st.text_input("Model", secret("GEMINI_MODEL", "gemini-2.5-flash"),
                               help="Use any current Gemini text model your key can access.")
    consent = st.checkbox(
        "I confirm the organiser permits sending the name-redacted transcript and computed "
        "metrics (never the video or audio) to the Gemini API.")
    run = st.button("Run audit", type="primary", use_container_width=True,
                    disabled=video_file is None)

# ---- Run --------------------------------------------------------------------
if run and video_file is not None:
    st.session_state.pop("assessment", None)
    st.session_state.pop("llm_error", None)
    bar = st.progress(0.0, text="Starting...")
    with tempfile.TemporaryDirectory() as tmp:
        suffix = os.path.splitext(video_file.name)[1] or ".mp4"
        video_path = os.path.join(tmp, "class" + suffix)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        vtt_path = None
        if vtt_file is not None:
            vtt_path = os.path.join(tmp, "class.vtt")
            with open(vtt_path, "wb") as f:
                f.write(vtt_file.getbuffer())
        names = [n.strip() for n in names_text.replace("\n", ",").split(",") if n.strip()]
        try:
            st.session_state["result"] = analyze(
                video_path, vtt_path, whisper_model, names,
                progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg))
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("result", None)
            bar.empty()
            st.error(f"Could not process this recording: {e}")
            st.stop()

    result = st.session_state["result"]
    if consent and api_key:
        bar.progress(0.95, text="Asking the LLM to assess the class...")
        try:
            st.session_state["assessment"] = assess(result, api_key, model_name)
        except Exception as e:  # noqa: BLE001
            st.session_state["llm_error"] = str(e)
    elif not consent:
        st.session_state["llm_error"] = ("LLM step skipped: consent box not ticked. "
                                         "Showing local signals only.")
    else:
        st.session_state["llm_error"] = "LLM step skipped: no Gemini API key provided."
    bar.empty()

# ---- Results ----------------------------------------------------------------
result = st.session_state.get("result")
if result is None:
    st.info("Upload a class recording in the sidebar and click **Run audit**. "
            "A .vtt transcript is optional.")
    st.stop()

left, right = st.columns([2, 3], gap="large")

with left:
    st.subheader("Measured signals")
    a, v, t = result["audio"], result["video"], result["text"]
    c1, c2 = st.columns(2)
    c1.metric("Duration", f"{result['duration_s'] / 60:.1f} min")
    if a.get("has_audio"):
        c2.metric("Time speaking", f"{a['speech_ratio']:.0%}")
        c1.metric("Dead air (>5 s)", f"{a['long_silence_ratio']:.0%}")
        c2.metric("Longest silence", f"{a['longest_silence_s']} s")
    if v.get("readable"):
        c1.metric("Camera off/black", f"{v['dark_ratio']:.0%}")
        if v.get("face_ratio") is not None:
            c2.metric("Face detected", f"{v['face_ratio']:.0%}")
        c1.metric("Brightness", f"{v['mean_brightness']:.1f}")
        c2.metric("Sharpness", f"{v['mean_sharpness']:.1f}")
    if t.get("available"):
        c1.metric("Questions / 10 min", t["questions_per_10min"])
        c2.metric("Words / min", t["words_per_min"])
    st.caption(f"Transcript source: {result['transcript_source']}")

    st.subheader("Rule-based flags")
    for f in result["flags"]:
        (st.warning if f["kind"] == "risk" else st.info)(f["msg"])
    if not result["flags"]:
        st.success("No rule-based flags raised.")

    if a.get("speech_per_min"):
        st.subheader("Speech activity per minute")
        st.line_chart(pd.DataFrame({"share of minute spoken": a["speech_per_min"]}))

with right:
    st.subheader("Assessment")
    assessment = st.session_state.get("assessment")
    if st.session_state.get("llm_error"):
        st.warning(st.session_state["llm_error"])

    if assessment:
        if isinstance(assessment, str):
            try:
                assessment = json.loads(assessment)
            except Exception:
                pass

        if isinstance(assessment, dict):
            rating = assessment.get("rating", "Unrated")
            confidence = assessment.get("confidence", "n/a")
            summary_text = assessment.get("summary", "")
            evidence_list = assessment.get("evidence", [])
            rubric_check_dict = assessment.get("rubric_check", {})
            improvements_list = assessment.get("improvements", assessment.get("recommendations", []))
            caveats_text = assessment.get("signal_caveats", "")
        else:
            rating = "Unrated"
            confidence = "n/a"
            summary_text = str(assessment)
            evidence_list = []
            rubric_check_dict = {}
            improvements_list = []
            caveats_text = ""

        banner = {"Excellent": st.success, "Good": st.info,
                  "Average": st.warning, "Poor": st.error}.get(rating, st.info)
        banner(f"**Rating: {rating}**  (confidence: {confidence})")
        st.write(summary_text)

        st.markdown("#### Evidence")
        icons = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
        for e in evidence_list:
            if isinstance(e, dict):
                st.markdown(f"{icons.get(e.get('impact'), '⚪')} **[{e.get('timestamp', 'n/a')}]** "
                            f"*({e.get('signal', '')})* {e.get('observation', '')}")
            else:
                st.markdown(f"⚪ {e}")

        if rubric_check_dict:
            st.markdown("#### Rubric check")
            for k, val in rubric_check_dict.items():
                st.markdown(f"- **{k.replace('_', ' ').title()}:** {val}")

        if improvements_list:
            st.markdown("#### Recommendations")
            for i, item in enumerate(improvements_list, 1):
                st.markdown(f"{i}. {item}")

        if caveats_text:
            with st.expander("Which signals were trusted or discounted?"):
                st.write(caveats_text)

    if result["text"].get("privacy_mentions"):
        with st.expander("Personal-information mentions (for human review)"):
            for m in result["text"]["privacy_mentions"]:
                st.markdown(f"**[{m['time']}]** {m['context']}")
