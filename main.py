# # uvicorn main:app --reload  for run the project
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import cv2
# import numpy as np
# import tempfile
# import traceback
# from ultralytics import YOLO

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model = YOLO("yolov8s-pose.pt")


# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#     ba = a - b
#     bc = c - b
#     cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
#     return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# @app.post("/analyze/")
# async def analyze_video(
#     file: UploadFile = File(...),
#     exercise: str = Form(default="Push-Ups"),
# ):

#     # ✅ FIX 1: Close the temp file before OpenCV reads it (critical on Windows)
#     temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#     temp.write(await file.read())
#     temp.close()  # <-- THIS LINE was missing, causing partial reads

#     cap = cv2.VideoCapture(temp.name)

#     states = {
#         "Push-Ups": {"counter": 0, "stage": "start"},
#         "Squats": {"counter": 0, "stage": "start"},
#         "Jumping Jacks": {"counter": 0, "stage": "start"},
#         "Plank": {"frames": 0}
#     }

#     frame_skip = 2
#     frame_count = 0

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps == 0 or fps is None:
#         fps = 30

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue

#         frame = cv2.resize(frame, (640, 480))

#         try:
#             results_yolo = model(frame, conf=0.4, verbose=False)

#             if (
#                 results_yolo
#                 and results_yolo[0].keypoints is not None
#                 and len(results_yolo[0].keypoints.xy) > 0
#             ):
#                 kp = results_yolo[0].keypoints.xy[0].cpu().numpy()

#                 if len(kp) < 17:
#                     continue

#                 l_shoulder = kp[5]
#                 r_shoulder = kp[6]
#                 l_elbow    = kp[7]
#                 l_wrist    = kp[9]
#                 l_hip      = kp[11]
#                 l_knee     = kp[13]
#                 l_ankle    = kp[15]
#                 r_ankle    = kp[16]

#                 # ---------------- PUSH UPS ----------------
#                 pu_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
#                 if pu_angle < 110: # Relaxed from 90
#                     states["Push-Ups"]["stage"] = "down"
#                 if pu_angle > 150 and states["Push-Ups"]["stage"] == "down": # Relaxed from 160
#                     states["Push-Ups"]["stage"] = "up"
#                     states["Push-Ups"]["counter"] += 1

#                 # ---------------- PLANK ----------------
#                 pl_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
#                 if 140 < pl_angle < 220:
#                     states["Plank"]["frames"] += frame_skip

#                 # ---------------- JUMPING JACKS ----------------
#                 ankle_distance    = abs(l_ankle[0] - r_ankle[0])
#                 shoulder_distance = abs(l_shoulder[0] - r_shoulder[0])
#                 if ankle_distance > shoulder_distance * 1.5:
#                     states["Jumping Jacks"]["stage"] = "open"
#                 elif ankle_distance < shoulder_distance and states["Jumping Jacks"]["stage"] == "open":
#                     states["Jumping Jacks"]["stage"] = "close"
#                     states["Jumping Jacks"]["counter"] += 1

#                 # ---------------- SQUATS ----------------
#                 sq_angle = calculate_angle(l_hip, l_knee, l_ankle)
#                 if sq_angle < 100: # Relaxed from 85
#                     states["Squats"]["stage"] = "down"
#                 elif sq_angle > 160 and states["Squats"]["stage"] == "down": # Relaxed from 170
#                     states["Squats"]["stage"] = "up"
#                     states["Squats"]["counter"] += 1

#         except Exception as e:
#             print("ERROR:", e)
#             continue

#     cap.release()

#     # Final tallies
#     final_results = {
#         "Push-Ups": states["Push-Ups"]["counter"],
#         "Jumping Jacks": states["Jumping Jacks"]["counter"],
#         "Squats": states["Squats"]["counter"],
#         "Plank": int(states["Plank"]["frames"] / fps)
#     }

#     # Mismatch Detection Heuristic
#     # We must treat repetition-based exercises (Push-Ups, Squats, Jumping Jacks) 
#     # differently from duration-based exercises (Plank).
#     repetition_exercises = ["Push-Ups", "Jumping Jacks", "Squats"]
#     is_mismatch = False
#     detected_exercise = exercise

#     if exercise in repetition_exercises:
#         # 1. Check for conflicts with other repetition-based exercises
#         other_reps = {k: v for k, v in final_results.items() if k in repetition_exercises}
#         dominant_rep_ex = max(other_reps, key=other_reps.get)
#         max_rep_val = other_reps[dominant_rep_ex]
#         selected_val = final_results.get(exercise, 0)

#         if max_rep_val >= 2:
#             if selected_val == 0:
#                 is_mismatch = True
#                 detected_exercise = dominant_rep_ex
#             elif dominant_rep_ex != exercise and max_rep_val > selected_val * 2:
#                 is_mismatch = True
#                 detected_exercise = dominant_rep_ex

#         # 2. Check for conflicts with Plank (duration)
#         # If reps are very low but plank time is significant, it's likely a plank video.
#         plank_time = final_results.get("Plank", 0)
#         if not is_mismatch:
#             if plank_time > 10 and selected_val <= 1:
#                 is_mismatch = True
#                 detected_exercise = "Plank"
#     else:
#         # If Plank was selected, check if a repetition exercise was clearly being performed
#         selected_val = final_results.get("Plank", 0)
#         rep_results = {k: v for k, v in final_results.items() if k in repetition_exercises}
#         best_rep_exercise = max(rep_results, key=rep_results.get)
#         best_rep_val = rep_results[best_rep_exercise]

#         if best_rep_val >= 3: # If they did 3+ reps of something else, it's a mismatch
#             is_mismatch = True
#             detected_exercise = best_rep_exercise

#     if is_mismatch:
#         return JSONResponse(
#             status_code=400,
#             content={
#                 "error": "exercise video didn't match",
#                 "selected": exercise,
#                 "detected": detected_exercise,
#                 "all_results": final_results
#             }
#         )

#     if exercise == "Plank":
#         return JSONResponse({
#             "exercise": "Plank",
#             "plank_time_seconds": final_results["Plank"]
#         })
#     else:
#         return JSONResponse({
#             "exercise": exercise,
#             "total_reps": final_results[exercise]
#         })
# uvicorn main:app --reload

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import tempfile
import tensorflow as tf

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

movenet_interpreter = tf.lite.Interpreter(model_path="movenet_lightning.tflite")
movenet_interpreter.allocate_tensors()
movenet_input_details = movenet_interpreter.get_input_details()
movenet_output_details = movenet_interpreter.get_output_details()


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
        input_image = cv2.resize(frame, (192, 192))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = input_image.astype(np.int32)
        
        movenet_interpreter.set_tensor(movenet_input_details[0]['index'], input_image)
        movenet_interpreter.invoke()
        keypoints_with_scores = movenet_interpreter.get_tensor(movenet_output_details[0]['index'])
        
        kps = keypoints_with_scores[0][0]
        
        if kps[5][2] > 0.1 or kps[6][2] > 0.1: # rudimentary confidence check
            kp = np.zeros((17, 2))
            h, w, _ = frame.shape
            for i in range(17):
                kp[i] = [kps[i][1] * w, kps[i][0] * h]

            if len(kp) >= 17:
                l_shoulder = kp[5]
                r_shoulder = kp[6]
                l_elbow = kp[7]
                l_wrist = kp[9]
                l_hip = kp[11]
                l_knee = kp[13]
                l_ankle = kp[15]
                r_ankle = kp[16]

                exercise_norm = normalize_exercise_name(exercise)

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

        frame = cv2.resize(frame, (640, 480))

        try:
            input_image = cv2.resize(frame, (192, 192))
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = np.expand_dims(input_image, axis=0)
            input_image = input_image.astype(np.int32)
            
            movenet_interpreter.set_tensor(movenet_input_details[0]['index'], input_image)
            movenet_interpreter.invoke()
            keypoints_with_scores = movenet_interpreter.get_tensor(movenet_output_details[0]['index'])
            
            kps = keypoints_with_scores[0][0]
            
            if kps[5][2] > 0.1 or kps[6][2] > 0.1:
                kp = np.zeros((17, 2))
                h, w, _ = frame.shape
                for i in range(17):
                    kp[i] = [kps[i][1] * w, kps[i][0] * h]

                if len(kp) < 17:
                    continue

                # Keypoints
                l_shoulder = kp[5]
                r_shoulder = kp[6]

                l_elbow = kp[7]
                l_wrist = kp[9]

                l_hip = kp[11]
                l_knee = kp[13]
                l_ankle = kp[15]

                r_ankle = kp[16]

                # ========================================
                # PUSH-UPS
                # ========================================

                pushup_angle = calculate_angle(
                    l_shoulder,
                    l_elbow,
                    l_wrist
                )

                if pushup_angle < 110:
                    states["Push-Ups"]["stage"] = "down"

                if (
                    pushup_angle > 150
                    and states["Push-Ups"]["stage"] == "down"
                ):
                    states["Push-Ups"]["stage"] = "up"
                    states["Push-Ups"]["counter"] += 1

                # ========================================
                # PLANK
                # ========================================


# ========================================
# PLANK (STRICT DETECTION) - FIXED          
# ========================================

                plank_angle = calculate_angle(
                    l_shoulder,
                    l_hip,
                    l_ankle
                )

                # Extra check:
                # If body is straight and hips are stable,
                # then only count as plank

                # Vertical difference between shoulder and hip  
                hip_y_diff = abs(l_shoulder[1]   - l_hip[1])

                # Horizontal body balance check
                body_length = abs(l_shoulder[0] - l_ankle[0])

                # STRICT plank validation
                if (
                    165 < plank_angle < 195      # straighter body only
                    and hip_y_diff < 120         # hips should not move too much
                    and body_length > 100        # body should be extended
                ):
                    states["Plank"]["frames"] += frame_skip

                # ========================================
                # JUMPING JACKS
                # ========================================

                ankle_distance = abs(
                    l_ankle[0] - r_ankle[0]
                )

                shoulder_distance = abs(
                    l_shoulder[0] - r_shoulder[0]
                )

                if ankle_distance > shoulder_distance * 1.5:
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

                squat_angle = calculate_angle(
                    l_hip,
                    l_knee,
                    l_ankle
                )

                if squat_angle < 95:
                    states["Squats"]["down_frames"] += 1
                else:
                    states["Squats"]["down_frames"] = 0

                if squat_angle > 165:
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

    rep_exercises = ["Push-Ups", "Squats", "Jumping Jacks"]
    rep_results = {k: final_results[k] for k in rep_exercises}
    dominant_rep_exercise = max(rep_results, key=rep_results.get)
    dominant_rep_value = rep_results[dominant_rep_exercise]
    selected_value = final_results.get(exercise, 0)
    plank_seconds = final_results["Plank"]

    detected_exercise = exercise

    print("Selected Exercise:", exercise)

    mismatch = False

    # For plank validation
    if exercise == "Plank":
        # If plank hold is too short but reps are clearly present, this is the wrong video.
        if plank_seconds < 5 and dominant_rep_value >= 3:
            mismatch = True
            detected_exercise = dominant_rep_exercise

    # For repetition-based exercises
    else:
        # Strong evidence of a different repetition exercise.
        if (
            dominant_rep_exercise != exercise
            and dominant_rep_value >= 3
            and dominant_rep_value >= selected_value + 2
        ):
            mismatch = True
            detected_exercise = dominant_rep_exercise

        # Selected exercise has almost no reps but plank posture is held for long.
        elif selected_value < 2 and plank_seconds >= 8:
            mismatch = True
            detected_exercise = "Plank"

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

# uvicorn main:app --reload

# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import cv2
# import numpy as np
# import tempfile
# from ultralytics import YOLO

# app = FastAPI()

# # ---------------- CORS ----------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------------- MODEL ----------------
# model = YOLO("yolov8s-pose.pt")


# # ---------------- HELPER ----------------
# def calculate_angle(a, b, c):
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

#     ba = a - b
#     bc = c - b

#     cosine = np.dot(ba, bc) / (
#         np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
#     )

#     return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# # =========================================================
# # 🎥 LIVE CAMERA API
# # =========================================================
# @app.post("/live/")
# async def live_analysis(
#     file: UploadFile = File(...),
#     exercise: str = Form(...)
# ):
#     temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
#     temp.write(await file.read())
#     temp.close()

#     frame = cv2.imread(temp.name)

#     result = {
#         "message": "",
#         "value": 0
#     }

#     try:
#         results = model(frame, conf=0.4, verbose=False)

#         if results and results[0].keypoints is not None:
#             kp = results[0].keypoints.xy[0].cpu().numpy()

#             if len(kp) >= 17:

#                 l_shoulder = kp[5]
#                 r_shoulder = kp[6]
#                 l_elbow = kp[7]
#                 l_wrist = kp[9]
#                 l_hip = kp[11]
#                 l_knee = kp[13]
#                 l_ankle = kp[15]
#                 r_ankle = kp[16]

#                 # PUSH-UP
#                 if exercise == "Push-Ups":
#                     angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
#                     if angle > 170:
#                         result["message"] = "Go Lower!"
#                     result["value"] = int(angle)

#                 # SQUAT
#                 elif exercise == "Squats":
#                     angle = calculate_angle(l_hip, l_knee, l_ankle)
#                     if angle > 150:
#                         result["message"] = "Go Deeper!"
#                     result["value"] = int(angle)

#                 # PLANK
#                 elif exercise == "Plank":
#                     angle = calculate_angle(l_shoulder, l_hip, l_ankle)
#                     if not (165 < angle < 195):
#                         result["message"] = "Fix Body Line!"
#                     result["value"] = int(angle)

#                 # JUMPING JACKS
#                 elif exercise == "Jumping Jacks":
#                     ankle_dist = abs(l_ankle[0] - r_ankle[0])
#                     shoulder_dist = abs(l_shoulder[0] - r_shoulder[0])

#                     if ankle_dist < shoulder_dist:
#                         result["message"] = "Jump Wider!"

#                     result["value"] = int(ankle_dist)

#     except:
#         result["message"] = "Detection Error"

#     return JSONResponse(result)


# # =========================================================
# # 📹 VIDEO UPLOAD API (WITH MISMATCH DETECTION)
# # =========================================================
# @app.post("/analyze/")
# async def analyze_video(
#     file: UploadFile = File(...),
#     exercise: str = Form(default="Push-Ups"),
# ):

#     temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
#     temp.write(await file.read())
#     temp.close()

#     cap = cv2.VideoCapture(temp.name)

#     states = {
#         "Push-Ups": {"counter": 0, "stage": "start"},
#         "Squats": {"counter": 0, "stage": "start"},
#         "Jumping Jacks": {"counter": 0, "stage": "start"},
#         "Plank": {"frames": 0}
#     }

#     frame_skip = 2
#     frame_count = 0

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps == 0:
#         fps = 30

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue

#         frame = cv2.resize(frame, (640, 480))

#         try:
#             results = model(frame, conf=0.4, verbose=False)

#             if results and results[0].keypoints is not None:
#                 kp = results[0].keypoints.xy[0].cpu().numpy()

#                 if len(kp) < 17:
#                     continue

#                 l_shoulder = kp[5]
#                 r_shoulder = kp[6]
#                 l_elbow = kp[7]
#                 l_wrist = kp[9]
#                 l_hip = kp[11]
#                 l_knee = kp[13]
#                 l_ankle = kp[15]
#                 r_ankle = kp[16]

#                 # PUSH-UP
#                 pu_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
#                 if pu_angle < 110:
#                     states["Push-Ups"]["stage"] = "down"
#                 if pu_angle > 150 and states["Push-Ups"]["stage"] == "down":
#                     states["Push-Ups"]["counter"] += 1

#                 # SQUATS
#                 sq_angle = calculate_angle(l_hip, l_knee, l_ankle)
#                 if sq_angle < 100:
#                     states["Squats"]["stage"] = "down"
#                 elif sq_angle > 160 and states["Squats"]["stage"] == "down":
#                     states["Squats"]["counter"] += 1

#                 # JUMPING JACKS
#                 ankle_distance = abs(l_ankle[0] - r_ankle[0])
#                 shoulder_distance = abs(l_shoulder[0] - r_shoulder[0])
#                 if ankle_distance > shoulder_distance * 1.5:
#                     states["Jumping Jacks"]["stage"] = "open"
#                 elif ankle_distance < shoulder_distance and states["Jumping Jacks"]["stage"] == "open":
#                     states["Jumping Jacks"]["counter"] += 1

#                 # PLANK (STRICT)
#                 pl_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
#                 if 165 < pl_angle < 195:
#                     states["Plank"]["frames"] += frame_skip

#         except:
#             continue

#     cap.release()

#     final_results = {
#         "Push-Ups": states["Push-Ups"]["counter"],
#         "Jumping Jacks": states["Jumping Jacks"]["counter"],
#         "Squats": states["Squats"]["counter"],
#         "Plank": int(states["Plank"]["frames"] / fps)
#     }

#     detected = max(final_results, key=final_results.get)

#     if detected != exercise:
#         return JSONResponse(
#             status_code=400,
#             content={
#                 "error": "exercise video didn't match",
#                 "selected": exercise,
#                 "detected": detected
#             }
#         )

#     if exercise == "Plank":
#         return {"plank_time_seconds": final_results["Plank"]}

#     return {"total_reps": final_results[exercise]}