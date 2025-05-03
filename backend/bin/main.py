from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import base64
from collections import deque

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ReactアプリのURL
    allow_credentials=True,
    allow_methods=["*"],   # 全HTTPメソッドを許可する（これ超重要）
    allow_headers=["*"],   # 全ヘッダーを許可する（これも超重要）
)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(await file.read())
    tmp_file.close()

    cap = cv2.VideoCapture(tmp_file.name)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    left_contact = False
    right_contact = False
    left_monitoring = False
    right_monitoring = False

    prev_left_heel_y = None
    prev_right_heel_y = None

    left_window = deque(maxlen=5)
    right_window = deque(maxlen=5)
    window_size = 5
    violation_threshold = 3

    movement_threshold = 0.1
    prev_landmarks = None

    violation_frame = None
    violation_side = None
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            if prev_landmarks is not None:
                movements = []
                for side in ('LEFT','RIGHT'):
                    for part in ('HIP','KNEE','ANKLE','HEEL'):
                        idx = getattr(mp_pose.PoseLandmark, f"{side}_{part}")
                        dx = landmarks[idx].x - prev_landmarks[idx].x
                        dy = landmarks[idx].y - prev_landmarks[idx].y
                        dz = landmarks[idx].z - prev_landmarks[idx].z
                        movements.append(np.linalg.norm([dx, dy, dz]))
                if max(movements) > movement_threshold:
                    prev_landmarks = landmarks
                    frame_index += 1
                    continue
            prev_landmarks = landmarks

            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            l_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]

            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            r_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

            left_heel_speed = abs(l_heel.y - prev_left_heel_y) if prev_left_heel_y is not None else 1.0
            right_heel_speed = abs(r_heel.y - prev_right_heel_y) if prev_right_heel_y is not None else 1.0

            if not left_contact:
                if left_heel_speed < 0.005 and l_heel.y > l_ankle.y + 0.02:
                    left_contact = True
                    left_monitoring = True
            elif left_monitoring:
                if l_ankle.y > 0.5:
                    knee_angle = calculate_3d_angle(l_hip, l_knee, l_ankle)
                    horiz_alignment = abs(l_hip.x - l_ankle.x) < 0.05
                    line_angle = calculate_3d_angle(l_hip, l_knee, l_ankle)
                    vert_alignment = line_angle > 178
                    violation_flag = (knee_angle < 178) and (line_angle < 178) and horiz_alignment
                    left_window.append(violation_flag)

                    if len(left_window) == window_size and sum(left_window) >= violation_threshold:
                        violation_frame = frame.copy()
                        violation_side = "left"
                        break

                if horiz_alignment and vert_alignment:
                    left_contact = False
                    left_monitoring = False
                    left_window.clear()

            if not right_contact:
                if right_heel_speed < 0.005 and r_heel.y > r_ankle.y + 0.02:
                    right_contact = True
                    right_monitoring = True
            elif right_monitoring:
                if r_ankle.y > 0.5:
                    knee_angle = calculate_3d_angle(r_hip, r_knee, r_ankle)
                    horiz_alignment = abs(r_hip.x - r_ankle.x) < 0.05
                    line_angle = calculate_3d_angle(r_hip, r_knee, r_ankle)
                    vert_alignment = line_angle > 178
                    violation_flag = (knee_angle < 178) and (line_angle < 178) and horiz_alignment
                    right_window.append(violation_flag)

                    if len(right_window) == window_size and sum(right_window) >= violation_threshold:
                        violation_frame = frame.copy()
                        violation_side = "right"
                        break

                if horiz_alignment and vert_alignment:
                    right_contact = False
                    right_monitoring = False
                    right_window.clear()

            prev_left_heel_y = l_heel.y
            prev_right_heel_y = r_heel.y

        frame_index += 1

    cap.release()
    os.remove(tmp_file.name)
    
    if violation_frame is not None:
        _, buffer = cv2.imencode('.jpg', violation_frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        return {
            "message": f"{violation_side} foot detected knee bend violation.",
            "frame": frame_index,
            "image_base64": img_b64
        }

    return {"message": "Knee extension is compliant."}

def calculate_3d_angle(a, b, c):
    p1 = np.array([a.x, a.y, a.z])
    p2 = np.array([b.x, b.y, b.z])
    p3 = np.array([c.x, c.y, c.z])
    v1 = p1 - p2
    v2 = p3 - p2
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

