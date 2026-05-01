import base64
import html
import os

import requests
import streamlit as st
import streamlit.components.v1 as components
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

API_BASE = os.environ.get("FITNESS_API_BASE", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.environ.get("FITNESS_API_TIMEOUT", "600"))
ANALYZE_URL = f"{API_BASE}/analyze/" 
# Autoplay embeds video as base64; cap size so the page stays responsive.
MAX_AUTOPLAY_BYTES = int(os.environ.get("FITNESS_MAX_AUTOPLAY_MB", "20")) * 1024 * 1024

st.set_page_config(
    page_title="AI Fitness Trainer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXERCISES = ["Push-Ups", "Plank", "Jumping Jacks", "Squats"]

EXERCISE_META = {
    "Push-Ups": {"icon": "💪", "blurb": "Counts full up–down cycles from arm angle."},
    "Plank": {"icon": "🧘", "blurb": "Measures stable hold time in a straight line."},
    "Jumping Jacks": {"icon": "⭐", "blurb": "Counts open–close leg cycles vs shoulder width."},
    "Squats": {"icon": "🦵", "blurb": "Counts squats from hip–knee–ankle angle."},
}

EXERCISE_GUIDE = {
    "Push-Ups": (
        "Keep your **full upper body** in frame (side view works best). "
        "Elbows should be visible so the model can measure the arm angle."
    ),
    "Plank": (
        "Use a **side view** with shoulders, hips, and ankles visible in a straight line. "
        "Avoid loose clothing that hides your silhouette."
    ),
    "Jumping Jacks": (
        "Frame your **whole body** including feet. "
        "Lighting should separate you from the background so ankles are detected clearly."
    ),
    "Squats": (
        "**Side or 3/4 view** is ideal so the hip–knee–ankle line is visible. "
        "Go deep enough for a clear bend at the knee."
    ),
}

STYLES = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    :root {
        --primary: #ff6384;
        --primary-glow: rgba(255, 99, 132, 0.5);
        --secondary: #6366f1;
        --accent: #2dd4bf;
        --bg-dark: #0f172a;
        --card-bg: rgba(30, 41, 59, 0.7);
        --border-color: rgba(255, 255, 255, 0.1);
        --glass-bg: rgba(255, 255, 255, 0.03);
    }

    html, body, [class*="css"] {
        font-family: "Plus Jakarta Sans", "Outfit", system-ui, sans-serif;
        color: #f8fafc;
    }

    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    #MainMenu, footer, .stDeployButton {
        display: none !important;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.98); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .animate-in {
        animation: fadeInScale 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    .hero-container {
        background: linear-gradient(135deg, rgba(255, 99, 132, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border-radius: 24px;
        padding: 3rem 2rem;
        border: 1px solid var(--border-color);
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.7;
        max-width: 600px;
        line-height: 1.6;
    }

    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-4px);
    }

    .step-indicator {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        border-radius: 10px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-right: 12px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        background: rgba(45, 212, 191, 0.1);
        color: #2dd4bf;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(45, 212, 191, 0.2);
    }

    .result-container {
        padding: 2.5rem;
        text-align: center;
        background: radial-gradient(circle at center, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
    }
    .result-value {
        font-size: 6rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 1rem;
        background: linear-gradient(to bottom, #fff, #cbd5e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 99, 132, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 99, 132, 0.4) !important;
    }

    .exercise-hint {
        font-size: 0.9rem;
        opacity: 0.78;
        margin-top: 1rem;
        line-height: 1.45;
    }

    .app-footer {
        margin-top: 5rem;
        padding-top: 2rem;
        border-top: 1px solid var(--border-color);
        font-size: 0.8rem;
        opacity: 0.5;
        text-align: center;
    }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


def format_plank_hms(seconds: int) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def rep_feedback(exercise: str, reps: int) -> str:
    if reps <= 0:
        return "No full reps detected — check lighting, camera angle, and that the move matches the selected exercise."
    if reps < 5:
        return "A few reps counted. If that feels low, try a clearer side view and slower, fuller range of motion."
    if reps < 15:
        return "Solid set. Consistent form and framing will keep counts accurate."
    return "Strong volume — keep prioritizing form as reps go up."


def plank_feedback(seconds: int) -> str:
    if seconds <= 0:
        return "No stable plank time detected — stay in a straight line (shoulder–hip–ankle) for several seconds."
    if seconds < 30:
        return "Good start — aim for a steady hold with minimal sag or pike."
    if seconds < 60:
        return "Nice hold — your body stayed in the plank corridor for a meaningful duration."
    return "Impressive endurance — make sure recovery matches the effort."


def rep_verdict(reps: int) -> str:
    if reps <= 0:
        return "Keep trying"
    if reps < 8:
        return "Good start"
    if reps < 20:
        return "Nice work"
    return "Great session"


def plank_verdict(seconds: int) -> str:
    if seconds <= 0:
        return "Reset & retry"
    if seconds < 45:
        return "Solid hold"
    if seconds < 90:
        return "Strong core"
    return "Outstanding"


def render_video_preview(uploaded_file, raw: bytes, *, autoplay: bool) -> None:
    """Show preview; when autoplay (Analyze just clicked), start playback (muted — browser policy)."""
    mime = uploaded_file.type or "video/mp4"
    if not mime.startswith("video/"):
        mime = "video/mp4"

    if autoplay and len(raw) <= MAX_AUTOPLAY_BYTES:
        b64 = base64.b64encode(raw).decode("ascii")
        components.html(
            f"""
            <div style="font-family:'DM Sans',system-ui,sans-serif">
                <video width="100%" controls autoplay muted playsinline
                    style="border-radius:12px;border:1px solid rgba(255,255,255,0.12);max-height:440px;background:#111">
                    <source src="data:{mime};base64,{b64}" type="{mime}">
                </video>
                <p style="font-size:12px;opacity:0.7;margin:10px 0 0 0">
                    Auto-playing (muted) — use the video controls to unmute if your clip has sound.
                </p>
            </div>
            """,
            height=480,
        )
    else:
        st.video(uploaded_file)
        if autoplay and len(raw) > MAX_AUTOPLAY_BYTES:
            st.caption(
                "File is large — autoplay is off so the page stays fast. Press play on the preview above."
            )


if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

st.markdown(
    """
    <div class="hero-container animate-in">
        <h1 class="hero-title">AI Fitness Trainer</h1>
        <p class="hero-subtitle">Elevate your workout with real-time pose estimation. Get precision counts and performance insights from your training videos.</p>
        <div style="display: flex; gap: 10px; margin-top: 20px;">
            <span class="status-pill">YOLOv8 Powered</span>
            <span class="status-pill" style="background: rgba(99, 102, 241, 0.1); color: #6366f1; border-color: rgba(99, 102, 241, 0.2);">Real-time Pose</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns((1, 1), gap="large")

with c1:
    st.markdown(
        '<div class="glass-card animate-in stagger-1">'
        '<div style="display: flex; align-items: center; margin-bottom: 20px;">'
        '<span class="step-indicator">1</span>'
        '<span style="font-size: 1.2rem; font-weight: 700;">Select & Upload</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    exercise = st.selectbox("Current Move", EXERCISES, key="exercise_choice", label_visibility="collapsed")
    meta = EXERCISE_META[exercise]
    
    st.markdown(
        f'<div class="exercise-hint">'
        f'<div style="font-size: 1.5rem; margin-bottom: 8px;">{meta["icon"]}</div>'
        f'<strong>{exercise}</strong><br>'
        f'<span style="opacity: 0.7;">{meta["blurb"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio("Mode", ["Upload Video", "Live Camera"], horizontal=True)

    with st.expander("Pro Tips", expanded=False):
        st.markdown(EXERCISE_GUIDE[exercise])

    if mode == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "mov", "avi"],
            label_visibility="collapsed"
        )

    if mode == "Upload Video":
        if uploaded_file is not None:
            raw = uploaded_file.getvalue()
            size_mb = len(raw) / (1024 * 1024)
            short_name = uploaded_file.name[:40] + ("…" if len(uploaded_file.name) > 40 else "")
            safe_short = html.escape(short_name)
            est_note = (
                "Larger files may take longer to upload and analyze."
                if size_mb > 25
                else "Ready to analyze."
            )
            st.markdown(
                f'<dl class="file-strip">'
                f"<div><dt>File</dt><dd>{safe_short}</dd></div>"
                f"<div><dt>Size</dt><dd>{size_mb:.1f} MB</dd></div>"
                f"<div><dt>Status</dt><dd>{est_note}</dd></div>"
                f"</dl>",
                unsafe_allow_html=True,
            )
        else:
            raw = None
    else:
        uploaded_file = None
        raw = None

    if mode == "Upload Video":
        analyze = st.button(
            "Analyze video",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )
    else:
        analyze = False

with c2:
    st.markdown(
        '<div class="glass-card animate-in stagger-2" style="min-height: 100%;">'
        '<div style="display: flex; align-items: center; margin-bottom: 20px;">'
        '<span class="step-indicator">2</span>'
        '<span style="font-size: 1.2rem; font-weight: 700;">Visual Preview</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    if mode == "Upload Video":
        if uploaded_file is not None:
            render_video_preview(uploaded_file, raw, autoplay=analyze)
        else:
            st.markdown(
                '<div style="height: 300px; display: flex; flex-direction: column; align-items: center; justify-content: center; border: 2px dashed rgba(255,255,255,0.05); border-radius: 16px; color: rgba(255,255,255,0.3);">'
                '<div style="font-size: 2rem; margin-bottom: 10px;">📹</div>'
                'Waiting for video upload...'
                '</div>',
                unsafe_allow_html=True,
            )
    elif mode == "Live Camera":
        st.info("Allow camera access to start.")

        class FitnessVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.exercise = exercise
                self.count = 0
                self.stage = "start"
                self.frames = 0
                self.message = ""
                self.hint = ""

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                _, buffer = cv2.imencode(".jpg", img)

                try:
                    res = requests.post(
                        f"{API_BASE}/live/",
                        files={"file": buffer.tobytes()},
                        data={"exercise": self.exercise},
                        timeout=1
                    ).json()

                    val = res.get("value", 0)
                    self.message = res.get("message", "")
                    self.hint = res.get("hint", "")

                    if self.exercise == "Push-Ups":
                        if val < 110:
                            self.stage = "down"
                        elif val > 150 and self.stage == "down":
                            self.count += 1
                            self.stage = "up"
                    elif self.exercise == "Squats":
                        if val < 100:
                            self.stage = "down"
                        elif val > 160 and self.stage == "down":
                            self.count += 1
                            self.stage = "up"
                    elif self.exercise == "Jumping Jacks":
                        if val > 200:
                            self.stage = "open"
                        elif val < 100 and self.stage == "open":
                            self.count += 1
                            self.stage = "close"
                    elif self.exercise == "Plank":
                        if 160 < val < 200:
                            self.frames += 1
                        self.count = int(self.frames / 10)

                    # Draw count
                    cv2.putText(img, f"Count: {self.count}", (10, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                    # Draw message
                    if self.message:
                        cv2.putText(img, self.message, (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        if self.hint:
                            cv2.putText(img, self.hint, (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                except Exception as e:
                    cv2.putText(img, "Server Error or Loading...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="fitness-live",
            video_processor_factory=FitnessVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            if webrtc_ctx.video_processor.exercise != exercise:
                webrtc_ctx.video_processor.exercise = exercise
                webrtc_ctx.video_processor.count = 0
                webrtc_ctx.video_processor.frames = 0

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

st.markdown(
    '<div class="results-title"><span class="dot"></span><span class="section-title">Results</span></div>',
    unsafe_allow_html=True,
)

if analyze and uploaded_file is not None and raw is not None:
    files = {
        "file": (
            uploaded_file.name,
            raw,
            uploaded_file.type or "video/mp4",
        )
    }
    data = {"exercise": exercise}

    with st.status("Initializing AI Vision Engine...", expanded=True) as status:
        st.markdown('<div style="position: relative; height: 10px; margin-bottom: 20px;"><div class="scanner"></div></div>', unsafe_allow_html=True)
        try:
            status.update(label="Analyzing skeletal landmarks...", state="running")
            response = requests.post(
                ANALYZE_URL,
                files=files,
                data=data,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.Timeout:
            status.update(label="Timed out — try a shorter clip.", state="error")
            st.error(
                "The request took too long. Use a shorter video, or set a higher "
                "`FITNESS_API_TIMEOUT` (seconds)."
            )
        except requests.exceptions.ConnectionError:
            status.update(label="Could not reach the server.", state="error")
            st.error(
                "Cannot connect to the API. Run `uvicorn main:app --reload` (default "
                "`http://127.0.0.1:8000`) or set `FITNESS_API_BASE`."
            )
        else:
            if response.status_code == 200:
                result = response.json()
                status.update(label="Analysis complete", state="complete")
                st.session_state.last_analysis = {
                    "exercise": exercise,
                    "result": result,
                    "filename": uploaded_file.name,
                }
            elif response.status_code == 400:
                res = response.json()
                if res.get("error") == "exercise video didn't match":
                    status.update(label="Exercise mismatch detected!", state="error")
                    st.error(
                        f"**Exercise video didn't match.** "
                        f"You selected **{res.get('selected')}**, but the video appears to be **{res.get('detected')}**."
                    )
                else:
                    status.update(label="Validation error", state="error")
                    st.error(res.get("error", "The server rejected the request."))
                st.session_state.last_analysis = None
            else:
                status.update(label="Something went wrong", state="error")
                st.error(f"Server returned HTTP {response.status_code}.")
                if response.text:
                    st.code(response.text[:2000], language=None)
                st.session_state.last_analysis = None

analysis = st.session_state.last_analysis
if analysis:
    res = analysis["result"]
    ex = analysis["exercise"]
    fname = analysis["filename"]

    st.markdown(
        '<div class="status-pill animate-in" style="background: rgba(45, 212, 191, 0.1); color: #2dd4bf; margin-bottom: 1rem;">'
        '<span>✓ Analysis Complete</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    safe_name = html.escape(fname)
    safe_ex = html.escape(ex)
    st.markdown(
        f'<div class="meta-grid">'
        f'<div class="meta-item"><span>Video</span><strong>{safe_name}</strong></div>'
        f'<div class="meta-item"><span>Exercise</span><strong>{safe_ex}</strong></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()    

    if ex == "Plank":
        secs = res.get("plank_time_seconds")
        if secs is not None:
            s = int(secs)
            verdict = html.escape(plank_verdict(s))
            fb = html.escape(plank_feedback(s))
            st.markdown(
                f'<div class="glass-card animate-in" style="margin-top: 2rem; overflow: hidden; position: relative;">'
                f'<div class="result-container">'
                f'<div class="result-label">Endurance Score</div>'
                f'<div class="result-value">{s}<span class="result-unit">s</span></div>'
                f'<div style="font-size: 1.1rem; opacity: 0.8; margin-top: -10px;">{html.escape(format_plank_hms(s))}</div>'
                f'</div>'
                f'<div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05);">'
                f'<div style="font-size: 1.2rem; font-weight: 700; color: var(--accent); margin-bottom: 8px;">{verdict}</div>'
                f'<div style="opacity: 0.7; line-height: 1.6;">{fb}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error("The server did not return plank time.")
    else:
        reps = res.get("total_reps")
        if reps is not None:
            r = int(reps)
            verdict = html.escape(rep_verdict(r))
            fb = html.escape(rep_feedback(ex, r))
            st.markdown(
                f'<div class="glass-card animate-in" style="margin-top: 2rem; overflow: hidden; position: relative;">'
                f'<div class="result-container">'
                f'<div class="result-label">Total Repetitions</div>'
                f'<div class="result-value">{r}</div>'
                f'<div style="font-size: 1.1rem; opacity: 0.8; margin-top: -10px;">{ex} completed</div>'
                f'</div>'
                f'<div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05);">'
                f'<div style="font-size: 1.2rem; font-weight: 700; color: var(--accent); margin-bottom: 8px;">{verdict}</div>'
                f'<div style="opacity: 0.7; line-height: 1.6;">{fb}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.error("The server did not return a rep count.")

    if st.button("Clear results"):
        st.session_state.last_analysis = None
        st.rerun()
elif not analyze:
    st.caption("Run **Analyze video** to see your score here.")

st.markdown(
    '<p class="app-footer">Pose-based estimates — use consistent framing and lighting for the most reliable counts.</p>',
    unsafe_allow_html=True,
)

# import streamlit as st
# import requests
# import cv2
# import av
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# API_BASE = "http://127.0.0.1:8000"

# st.title("🏋️ AI Fitness Trainer")

# exercise = st.selectbox("Select Exercise", [
#     "Push-Ups", "Squats", "Plank", "Jumping Jacks"
# ])

# mode = st.radio("Choose Mode", ["Upload Video", "Live Camera"])

# # ===================================================
# # 📹 UPLOAD MODE
# # ===================================================

# if mode == "Upload Video":

#     uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])

#     if uploaded_file:
#         st.video(uploaded_file)

#         if st.button("Analyze"):
#             res = requests.post(
#                 f"{API_BASE}/analyze/",
#                 files={"file": uploaded_file},
#                 data={"exercise": exercise}
#             )

#             if res.status_code == 200:
#                 data = res.json()

#                 if exercise == "Plank":
#                     st.success(f"Plank Time: {data['plank_time_seconds']} sec")
#                 else:
#                     st.success(f"Reps: {data['total_reps']}")

#             else:
#                 err = res.json()
#                 st.error(
#                     f"Wrong Exercise! You selected {err['selected']} but did {err['detected']}"
#                 )


# # ===================================================
# # 🎥 LIVE CAMERA MODE
# # ===================================================

# if mode == "Live Camera":

#     st.warning("Allow camera access")

#     class VideoProcessor(VideoProcessorBase):

#         def recv(self, frame):
#             img = frame.to_ndarray(format="bgr24")

#             _, buffer = cv2.imencode(".jpg", img)

#             try:
#                 res = requests.post(
#                     f"{API_BASE}/live/",
#                     files={"file": buffer.tobytes()},
#                     data={"exercise": exercise},
#                     timeout=1
#                 ).json()

#                 if res["message"]:
#                     cv2.putText(img, res["message"], (50, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#                 cv2.putText(img, f"Value: {res['value']}", (10, 400),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#             except:
#                 cv2.putText(img, "Server Error", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#             return av.VideoFrame.from_ndarray(img, format="bgr24")

#     webrtc_streamer(
#         key="cam",
#         video_processor_factory=VideoProcessor
#     )


# import streamlit as st
# import requests
# import cv2
# import av
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# API_BASE = "http://127.0.0.1:8000"

# st.set_page_config(page_title="AI Trainer", layout="wide")

# EXERCISES = ["Push-Ups", "Squats", "Plank", "Jumping Jacks"]

# # ==============================
# # MODE SWITCH
# # ==============================
# mode = st.radio("Choose Mode", ["Upload Video", "Live Camera"])

# exercise = st.selectbox("Select Exercise", EXERCISES)

# # ==============================
# # UPLOAD VIDEO MODE (FIXED)
# # ==============================
# if mode == "Upload Video":

#     uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

#     if uploaded_file is not None:
#         st.video(uploaded_file)

#         if st.button("Analyze Video"):

#             raw = uploaded_file.read()

#             files = {
#                 "file": (
#                     uploaded_file.name,
#                     raw,
#                     uploaded_file.type or "video/mp4"
#                 )
#             }

#             data = {"exercise": exercise}

#             try:
#                 res = requests.post(
#                     f"{API_BASE}/analyze/",
#                     files=files,
#                     data=data,
#                     timeout=60
#                 )

#                 if res.status_code == 200:
#                     result = res.json()

#                     if exercise == "Plank":
#                         st.success(f"Plank Time: {result['plank_time_seconds']} sec")
#                     else:
#                         st.success(f"Reps: {result['total_reps']}")

#                 elif res.status_code == 400:
#                     err = res.json()
#                     st.error(
#                         f"❌ Wrong Exercise!\n\n"
#                         f"Selected: {err.get('selected')}\n"
#                         f"Detected: {err.get('detected')}"
#                     )

#                 else:
#                     st.error("Server Error")

#             except Exception as e:
#                 st.error("Connection Error")


# # ==============================
# # LIVE CAMERA MODE (WORKING)
# # ==============================
# elif mode == "Live Camera":

#     st.warning("Allow camera access")

#     class VideoProcessor(VideoProcessorBase):

#         def __init__(self):
#             self.states = {
#                 "Push-Ups": {"counter": 0, "stage": "start"},
#                 "Squats": {"counter": 0, "stage": "start"},
#                 "Jumping Jacks": {"counter": 0, "stage": "start"},
#                 "Plank": {"frames": 0}
#             }

#         def recv(self, frame):
#             img = frame.to_ndarray(format="bgr24")

#             _, buffer = cv2.imencode(".jpg", img)

#             try:
#                 res = requests.post(
#                     f"{API_BASE}/live/",
#                     files={"file": buffer.tobytes()},
#                     data={"exercise": exercise},
#                     timeout=1
#                 ).json()

#                 detected = res.get("detected", "")
#                 value = res.get("value", 0)
#                 message = res.get("message", "")

#                 # DETECTED LABEL
#                 cv2.putText(img, f"Detected: {detected}",
#                             (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#                 # ---------------- COUNTERS ----------------

#                 if exercise == "Push-Ups":
#                     if value < 110:
#                         self.states["Push-Ups"]["stage"] = "down"
#                     if value > 150 and self.states["Push-Ups"]["stage"] == "down":
#                         self.states["Push-Ups"]["counter"] += 1
#                         self.states["Push-Ups"]["stage"] = "up"
#                     count = self.states["Push-Ups"]["counter"]

#                 elif exercise == "Squats":
#                     if value < 100:
#                         self.states["Squats"]["stage"] = "down"
#                     elif value > 160 and self.states["Squats"]["stage"] == "down":
#                         self.states["Squats"]["counter"] += 1
#                         self.states["Squats"]["stage"] = "up"
#                     count = self.states["Squats"]["counter"]

#                 elif exercise == "Jumping Jacks":
#                     if value > 200:
#                         self.states["Jumping Jacks"]["stage"] = "open"
#                     elif value < 100 and self.states["Jumping Jacks"]["stage"] == "open":
#                         self.states["Jumping Jacks"]["counter"] += 1
#                         self.states["Jumping Jacks"]["stage"] = "close"
#                     count = self.states["Jumping Jacks"]["counter"]

#                 elif exercise == "Plank":
#                     if 165 < value < 195:
#                         self.states["Plank"]["frames"] += 1
#                     count = int(self.states["Plank"]["frames"] / 10)

#                 # COUNT DISPLAY
#                 cv2.putText(img, f"Count: {count}",
#                             (10, 400),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#                 # FEEDBACK
#                 if message:
#                     cv2.putText(img, message,
#                                 (50, 70),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#             except:
#                 cv2.putText(img, "Server Error",
#                             (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#             return av.VideoFrame.from_ndarray(img, format="bgr24")

#     webrtc_streamer(
#         key="fitness-cam",
#         video_processor_factory=VideoProcessor
#     )