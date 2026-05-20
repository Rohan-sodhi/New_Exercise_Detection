from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

app = FastAPI()

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL ----------------

model = YOLO("yolov8s-pose.pt")


# ---------------- HELPER FUNCTION ----------------

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    angle = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )

    return angle


def normalize_exercise_name(exercise_name: str) -> str:
    if not exercise_name:
        return ""

    normalized = exercise_name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    aliases = {
        "push up": "Push-Ups",
        "push ups": "Push-Ups",
        "pushup": "Push-Ups",
        "pushups": "Push-Ups",
        "squat": "Squats",
        "squats": "Squats",
        "jumping jack": "Jumping Jacks",
        "jumping jacks": "Jumping Jacks",
        "jumpingjack": "Jumping Jacks",
        "jumpingjacks": "Jumping Jacks",
        "plank": "Plank",
    }

    return aliases.get(normalized, exercise_name.strip())


# ---------------- API ----------------

@app.post("/live/")
async def live_analysis(
    file: UploadFile = File(...),
    exercise: str = Form(...)
):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(await file.read())
    temp.close()

    frame = cv2.imread(temp.name)

    result = {
        "message": "",
        "value": 0,
        "hint": ""
    }

    if frame is None:
        return JSONResponse(result)

    try:
        results = model(frame, conf=0.4, verbose=False)

        if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            kp = results[0].keypoints.xy[0].cpu().numpy()

            if len(kp) >= 17:
                l_shoulder = kp[5]
                r_shoulder = kp[6]
                l_elbow = kp[7]
                l_wrist = kp[9]
                l_hip = kp[11]
                l_knee = kp[13]
                l_ankle = kp[15]
                r_ankle = kp[16]

                conf = results[0].keypoints.conf[0].cpu().numpy()
                
                # Essential landmarks for ANY exercise to be valid
                # 0: nose, 5/6: shoulders, 11/12: hips
                has_upper = conf[0] > 0.5 and (conf[5] > 0.5 or conf[6] > 0.5)
                has_mid = conf[11] > 0.5 or conf[12] > 0.5
                
                # For standing exercises (Squats/Jumping Jacks), we also need ankles
                exercise_norm = normalize_exercise_name(exercise)
                has_lower = True
                if exercise_norm in ["Squats", "Jumping Jacks"]:
                    has_lower = conf[15] > 0.5 or conf[16] > 0.5

                if not (has_upper and has_mid and has_lower):
                    result["full_body_visible"] = False
                    result["message"] = "Full Body Not Visible"
                    result["hint"] = "Step back so your head, hips, and feet are clearly in frame."
                    return JSONResponse(result)
                
                result["full_body_visible"] = True


                exercise_norm = normalize_exercise_name(exercise)
                
                # --- POSTURE DETECTION ---
                is_standing = True
                if np.any(l_ankle) and np.any(l_shoulder):
                    y_dist = abs(l_ankle[1] - l_shoulder[1])
                    x_dist = abs(l_shoulder[0] - l_ankle[0])
                    # Cast to bool to avoid JSON serialization error with np.bool_
                    is_standing = bool(y_dist > x_dist * 0.4)
                result["is_standing"] = is_standing

                # PUSH-UP
                if exercise_norm == "Push-Ups":
                    angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    if angle > 160: # Threshold for top of movement
                        result["message"] = "Go Lower!"
                        result["hint"] = "Keep back straight, lower until elbows at 90 deg."
                    result["value"] = int(angle)

                # SQUAT
                elif exercise_norm == "Squats":
                    angle = calculate_angle(l_hip, l_knee, l_ankle)
                    if angle > 150: # Threshold for top of movement
                        result["message"] = "Go Deeper!"
                        result["hint"] = "Lower hips until thighs are parallel to the floor."
                    result["value"] = int(angle)

                # PLANK
                elif exercise_norm == "Plank":
                    angle = calculate_angle(l_shoulder, l_hip, l_ankle)
                    hip_y_diff = abs(l_shoulder[1] - l_hip[1])
                    if not (160 < angle < 200) or hip_y_diff >= 120:
                        result["message"] = "Fix Body Line!"
                        result["hint"] = "Keep back straight, hips aligned with shoulders."
                    result["value"] = int(angle)

                # JUMPING JACKS
                elif exercise_norm == "Jumping Jacks":
                    # For Jumping Jacks, we ideally want BOTH ankles
                    if not (np.any(kp[15]) and np.any(kp[16])):
                        result["full_body_visible"] = False
                        result["message"] = "Both feet must be visible"
                        result["hint"] = "Step back so both ankles are in frame."
                        return JSONResponse(result)

                    ankle_dist = abs(l_ankle[0] - r_ankle[0])
                    shoulder_dist = abs(l_shoulder[0] - r_shoulder[0])

                    if ankle_dist < shoulder_dist * 1.2: # Threshold for closed
                        result["message"] = "Jump Wider!"
                        result["hint"] = "Spread legs wider than shoulder-width."
                    result["value"] = int(ankle_dist)

    except Exception as e:
        print(f"Live Error: {e}")
        result["message"] = "Detection Error"

    import os
    if os.path.exists(temp.name):
        try:
            os.remove(temp.name)
        except:
            pass

    return JSONResponse(result)

@app.post("/analyze/")
async def analyze_video(
    file: UploadFile = File(...),
    exercise: str = Form(default="Push-Ups"),
):
    exercise = normalize_exercise_name(exercise)
    valid_exercises = {"Push-Ups", "Squats", "Jumping Jacks", "Plank"}

    if exercise not in valid_exercises:
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid exercise selected",
                "selected": exercise,
                "valid_options": sorted(valid_exercises),
            },
        )

    # Save uploaded video temporarily
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(await file.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)

    if not cap.isOpened():
        return JSONResponse(
            status_code=400,
            content={"error": "Unable to read uploaded video"}
        )

    # Exercise states
    states = {
        "Push-Ups": {
            "counter": 0,
            "stage": "start"
        },
        "Squats": {
            "counter": 0,
            "stage": "start",
            "down_frames": 0,
            "up_frames": 0,
            "last_count_frame": -9999
        },
        "Jumping Jacks": {
            "counter": 0,
            "stage": "start"
        },
        "Plank": {
            "frames": 0
        }
    }

    frame_skip = 2
    frame_count = 0
    processed_frames = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30

    # ---------------- VIDEO PROCESSING ----------------

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        processed_frames += 1

        # --- MOBILE UPLOAD FIX: MAINTAIN ASPECT RATIO ---
        # This prevents portrait videos from being squashed, which causes 0 reps.
        h, w = frame.shape[:2]
        target_size = 640
        if h > w:
            new_h, new_w = target_size, int(w * target_size / h)
        else:
            new_w, new_h = target_size, int(h * target_size / w)
        frame = cv2.resize(frame, (new_w, new_h))

        try:
            # Lowered confidence (0.3) for mobile video to handle blur
            results = model(frame, conf=0.3, verbose=False) 

            if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
                continue

            kp = results[0].keypoints.xy[0].cpu().numpy()

            if len(kp) < 17:
                continue

            conf = results[0].keypoints.conf[0].cpu().numpy()
            
            # More lenient visibility for uploaded videos
            has_shoulder = conf[5] > 0.3 or conf[6] > 0.3
            has_hip = conf[11] > 0.3 or conf[12] > 0.3

            if not (has_shoulder and has_hip):
                continue

            # Keypoints
            l_shoulder = kp[5]
            r_shoulder = kp[6]

            l_elbow = kp[7]
            l_wrist = kp[9]

            l_hip = kp[11]
            l_knee = kp[13]
            l_ankle = kp[15]
            r_knee = kp[14]
            r_ankle = kp[16]

            # --- LANDMARK FALLBACKS FOR ROBUSTNESS ---
            # Use knee as fallback for ankle if feet are out of frame
            eff_l_ankle = l_ankle if np.any(l_ankle) else l_knee
            eff_r_ankle = r_ankle if np.any(r_ankle) else r_knee

            # ========================================
            # PUSH-UPS
            # ========================================
            if exercise == "Push-Ups":
                if np.any(l_shoulder) and np.any(l_elbow) and np.any(l_wrist):
                    pushup_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    if pushup_angle < 110:
                        states["Push-Ups"]["stage"] = "down"
                    elif pushup_angle > 160 and states["Push-Ups"]["stage"] == "down":
                        states["Push-Ups"]["stage"] = "up"
                        states["Push-Ups"]["counter"] += 1

            # ========================================
            # PLANK
            # ========================================
            elif exercise == "Plank":
                if np.any(l_shoulder) and np.any(l_hip) and np.any(eff_l_ankle):
                    plank_angle = calculate_angle(l_shoulder, l_hip, eff_l_ankle)
                    hip_y_diff = abs(l_shoulder[1] - l_hip[1])
                    
                    # Relaxed plank validation (matching live camera logic)
                    if (160 < plank_angle < 200) and hip_y_diff < 180:
                        states["Plank"]["frames"] += frame_skip

            # ========================================
            # POSTURE CHECK (Robust for Mobile/Portrait)
            # ========================================
            
            # Default to the most likely posture for the selected exercise
            is_standing = exercise in ["Squats", "Jumping Jacks"]
            
            if np.any(eff_l_ankle) and np.any(l_shoulder):
                y_dist = abs(eff_l_ankle[1] - l_shoulder[1])
                x_dist = abs(l_shoulder[0] - eff_l_ankle[0])
                # If person is significantly taller than wide, they are likely standing
                is_standing = y_dist > x_dist * 0.4

            # ========================================
            # JUMPING JACKS
            # ========================================
            if exercise == "Jumping Jacks":
                if is_standing:
                    # For Jumping Jacks, both ankles must be visible (Lowered threshold to 0.3 for mobile uploads)
                    if not (conf[15] > 0.3 and conf[16] > 0.3):
                        continue

                    ankle_distance = abs(
                        l_ankle[0] - r_ankle[0]
                    )

                    shoulder_distance = abs(
                        l_shoulder[0] - r_shoulder[0]
                    )

                    # More lenient spread requirement (1.3 instead of 1.5)
                    if ankle_distance > shoulder_distance * 1.3:
                        states["Jumping Jacks"]["stage"] = "open"

                    elif (
                        ankle_distance < shoulder_distance
                        and states["Jumping Jacks"]["stage"] == "open"
                    ):
                        states["Jumping Jacks"]["stage"] = "close"
                        states["Jumping Jacks"]["counter"] += 1

            # ========================================
            # SQUATS
            # ========================================
            elif exercise == "Squats":
                if is_standing and np.any(l_hip) and np.any(l_knee) and np.any(l_ankle):
                    squat_angle = calculate_angle(
                        l_hip,
                        l_knee,
                        l_ankle
                    )

                    if squat_angle < 100: # Consistent with live detection
                        states["Squats"]["down_frames"] += 1
                    else:
                        states["Squats"]["down_frames"] = 0

                    if squat_angle > 160: # Consistent with live detection
                        states["Squats"]["up_frames"] += 1
                    else:
                        states["Squats"]["up_frames"] = 0

                    # Enter "down" only when the low angle is stable for multiple frames.
                    if states["Squats"]["down_frames"] >= 2:
                        states["Squats"]["stage"] = "down"

                    # Count rep only when back to stable standing + cooldown to avoid double counts.
                    squat_cooldown_frames = 8
                    if (
                        states["Squats"]["up_frames"] >= 2
                        and states["Squats"]["stage"] == "down"
                        and (frame_count - states["Squats"]["last_count_frame"]) >= squat_cooldown_frames
                    ):
                        states["Squats"]["stage"] = "up"
                        states["Squats"]["counter"] += 1
                        states["Squats"]["last_count_frame"] = frame_count

        except Exception as e:
            print("Processing Error:", str(e))
            continue

    cap.release()

    # ---------------- FINAL RESULTS ----------------

    final_results = {
        "Push-Ups": states["Push-Ups"]["counter"],
        "Jumping Jacks": states["Jumping Jacks"]["counter"],
        "Squats": states["Squats"]["counter"],
        "Plank": int(states["Plank"]["frames"] / fps)
    }

    print("FINAL RESULTS:", final_results)

    # ---------------- EXERCISE MATCH DETECTION ----------------

    jj_reps = final_results.get("Jumping Jacks", 0)
    sq_reps = final_results.get("Squats", 0)
    pu_reps = final_results.get("Push-Ups", 0)
    plank_seconds = final_results.get("Plank", 0)

    print("Selected Exercise:", exercise)

    mismatch = False
    detected_exercise = exercise

    # Priority system to handle false positives.
    best_guess = exercise

    if jj_reps >= max(3, sq_reps * 0.8, pu_reps * 0.8):
        best_guess = "Jumping Jacks"
    elif sq_reps >= max(3, pu_reps * 0.7):
        best_guess = "Squats"
    elif plank_seconds >= 5 and pu_reps < 5 and sq_reps < 4 and jj_reps < 4:
        best_guess = "Plank"
    elif pu_reps >= 3:
        best_guess = "Push-Ups"

    if best_guess != exercise:
        # Avoid penalizing the selected exercise if it has at least some detection
        if exercise == "Push-Ups" and pu_reps >= 2: best_guess = "Push-Ups"
        elif exercise == "Squats" and sq_reps >= 2: best_guess = "Squats"
        elif exercise == "Jumping Jacks" and jj_reps >= 2: best_guess = "Jumping Jacks"
        elif exercise == "Plank" and plank_seconds >= 3: best_guess = "Plank"

        if best_guess != exercise:
            # Final mismatch confirm
            mismatch = True
            detected_exercise = best_guess

    # Fallback if no clear best_guess but we still have an obvious mismatch
    if not mismatch:
        if exercise == "Plank" and plank_seconds < 2:
            if jj_reps >= 2: mismatch = True; detected_exercise = "Jumping Jacks"
            elif sq_reps >= 2: mismatch = True; detected_exercise = "Squats"
            elif pu_reps >= 2: mismatch = True; detected_exercise = "Push-Ups"
        elif exercise != "Plank" and final_results.get(exercise, 0) < 2:
            if jj_reps >= 2: mismatch = True; detected_exercise = "Jumping Jacks"
            elif sq_reps >= 2: mismatch = True; detected_exercise = "Squats"
            elif plank_seconds >= 5 and exercise != "Push-Ups": 
                mismatch = True; detected_exercise = "Plank"
            elif pu_reps >= 2 and exercise != "Push-Ups": 
                mismatch = True; detected_exercise = "Push-Ups"

    # If mismatch found → return error
    if mismatch:
        return JSONResponse(
            status_code=400,
            content={
                "error": "exercise video didn't match",
                "selected": exercise,
                "detected": detected_exercise,
                "all_results": final_results
            }
        )

    # ---------------- FINAL RESPONSE ----------------

    if exercise == "Plank":
        return JSONResponse({
            "exercise": "Plank",
            "plank_time_seconds": final_results["Plank"]
        })

    return JSONResponse({
        "exercise": exercise,
        "total_reps": final_results[exercise]
    })

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "ok"})


@app.get("/")
async def root():
    return JSONResponse({"status": "ok", "message": "AI Fitness Trainer API"})
