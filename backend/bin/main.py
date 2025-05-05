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

def is_strict_side_view(landmarks, mp_pose, tol_deg=10):
    # 肩と腰で体幹平面を定義
    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    # 信頼度チェック
    if ls.visibility < 0.7 or rs.visibility < 0.7 or lh.visibility < 0.7:
        return False
    # 3Dベクトル化
    v1 = np.array([rs.x - ls.x, rs.y - ls.y, rs.z - ls.z])
    v2 = np.array([lh.x - ls.x, lh.y - ls.y, lh.z - ls.z])
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return False
    normal /= norm
    # カメラ視線ベクトルをZ軸とみなす
    cam_axis = np.array([0, 0, 1])
    angle = np.degrees(np.arccos(np.clip(np.dot(normal, cam_axis), -1.0, 1.0)))
    return abs(angle - 90.0) <= tol_deg

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    # 動画を一時保存
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(await file.read())
    tmp_file.close()

    cap = cv2.VideoCapture(tmp_file.name)
    mp_pose = __import__('mediapipe').solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    # パラメータ
    window_size = 3
    side_window = deque(maxlen=window_size)
    heel_speed_thresh = 0.005
    heel_height_offset = 0.02
    ankle_height_thresh = 0.5
    horizontal_thresh = 0.05
    knee_angle_thresh = 178.0
    tol_side_deg = 10

    prev_left_heel_y = None
    prev_right_heel_y = None
    frame_index = 0
    violation_frame = None
    violation_side = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        frame_index += 1
        if not result.pose_landmarks:
            continue

        lm = result.pose_landmarks.landmark
        # サイドビュー判定
        side_ok = is_strict_side_view(lm, mp_pose, tol_deg=tol_side_deg)
        side_window.append(side_ok)
        if len(side_window) < window_size or not all(side_window):
            continue

        # ランドマーク取得
        l_hip   = lm[mp_pose.PoseLandmark.LEFT_HIP]
        l_knee  = lm[mp_pose.PoseLandmark.LEFT_KNEE]
        l_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        l_heel  = lm[mp_pose.PoseLandmark.LEFT_HEEL]
        r_hip   = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        r_knee  = lm[mp_pose.PoseLandmark.RIGHT_KNEE]
        r_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]
        r_heel  = lm[mp_pose.PoseLandmark.RIGHT_HEEL]

        # かかと速度計算
        left_speed  = abs(l_heel.y - prev_left_heel_y)  if prev_left_heel_y is not None else 1.0
        right_speed = abs(r_heel.y - prev_right_heel_y) if prev_right_heel_y is not None else 1.0

        # 左足接地判定＆膝チェック
        if left_speed < heel_speed_thresh and (l_heel.y > l_ankle.y + heel_height_offset):
            if l_ankle.y > ankle_height_thresh and abs(l_hip.x - l_ankle.x) < horizontal_thresh:
                angle = calculate_3d_angle(l_hip, l_knee, l_ankle)
                if angle < knee_angle_thresh:
                    violation_frame = frame.copy()
                    violation_side = "left"
                    break

        # 右足接地判定＆膝チェック
        if right_speed < heel_speed_thresh and (r_heel.y > r_ankle.y + heel_height_offset):
            if r_ankle.y > ankle_height_thresh and abs(r_hip.x - r_ankle.x) < horizontal_thresh:
                angle = calculate_3d_angle(r_hip, r_knee, r_ankle)
                if angle < knee_angle_thresh:
                    violation_frame = frame.copy()
                    violation_side = "right"
                    break

        prev_left_heel_y  = l_heel.y
        prev_right_heel_y = r_heel.y

    cap.release()
    os.remove(tmp_file.name)

    if violation_frame is not None:
        _, buf = cv2.imencode('.jpg', violation_frame)
        img_b64 = base64.b64encode(buf).decode('utf-8')
        return {
            "message": f"{violation_side} foot knee bend violation at strict side view.",
            "frame": frame_index,
            "image_base64": img_b64
        }

    return {"message": "Knee extension compliant at strict side views only."}


def calculate_3d_angle(a, b, c):
    p1 = np.array([a.x, a.y, a.z])
    p2 = np.array([b.x, b.y, b.z])
    p3 = np.array([c.x, c.y, c.z])
    v1 = p1 - p2
    v2 = p3 - p2
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
