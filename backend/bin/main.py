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

# MediaPipe 初期化
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=False)
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Head pose モデルポイント (mm 単位の参考モデル)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye left corner
    (43.3, 32.7, -26.0),    # Right eye right corner
    (-28.9, -28.9, -24.1),  # Left mouth corner
    (28.9, -28.9, -24.1)    # Right mouth corner
], dtype='double')

def get_head_yaw(frame, face_landmarks):
    h, w = frame.shape[:2]
    image_points = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),       # Nose tip
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),   # Chin
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),     # Left eye corner
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),   # Right eye corner
        (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),     # Left mouth
        (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)    # Right mouth
    ], dtype='double')
    # カメラ行列 (焦点距離は幅を代用)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype='double')
    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None
    # 回転ベクトルを行列に変換してオイラー角取得
    rmat, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    yaw = euler[1][0]
    return yaw

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(await file.read())
    tmp_file.close()

    cap = cv2.VideoCapture(tmp_file.name)

    # 判定パラメータ
    window_size = 3
    side_window = deque(maxlen=window_size)
    heel_speed_thresh = 0.002      # かかと速度閾値を厳しく
    heel_height_offset = 0.03      # かかと高さオフセットを増加
    ankle_height_thresh = 0.6      # 足首高さを厳しく
    horizontal_thresh = 0.03       # X差閾値を厳しく
    knee_angle_thresh = 175.0      # 膝角度閾値を厳しく
    yaw_side_tol = 5.0             # Yaw の許容角度を狭める    # Yaw が 90°±tol の場合のみサイド

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

        # Head pose 推定
        face_res = face_mesh.process(frame_rgb)
        if not face_res.multi_face_landmarks:
            frame_index += 1
            continue
        yaw = get_head_yaw(frame, face_res.multi_face_landmarks[0])
        if yaw is None or abs(abs(yaw) - 90.0) > yaw_side_tol:
            frame_index += 1
            continue

        # Pose 推定
        res = pose.process(frame_rgb)
        if not res.pose_landmarks:
            frame_index += 1
            continue
        lm = res.pose_landmarks.landmark

        # ランドマーク取得
        l_hip, l_knee, l_ankle, l_heel = (
            lm[mp_pose.PoseLandmark.LEFT_HIP],
            lm[mp_pose.PoseLandmark.LEFT_KNEE],
            lm[mp_pose.PoseLandmark.LEFT_ANKLE],
            lm[mp_pose.PoseLandmark.LEFT_HEEL]
        )
        r_hip, r_knee, r_ankle, r_heel = (
            lm[mp_pose.PoseLandmark.RIGHT_HIP],
            lm[mp_pose.PoseLandmark.RIGHT_KNEE],
            lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
            lm[mp_pose.PoseLandmark.RIGHT_HEEL]
        )

        # かかと速度
        left_speed = abs(l_heel.y - prev_left_heel_y)  if prev_left_heel_y is not None else 1.0
        right_speed = abs(r_heel.y - prev_right_heel_y) if prev_right_heel_y is not None else 1.0

        # 左足判定
        if left_speed < heel_speed_thresh and (l_heel.y > l_ankle.y + heel_height_offset):
            if l_ankle.y > ankle_height_thresh and abs(l_hip.x - l_ankle.x) < horizontal_thresh:
                angle = calculate_3d_angle(l_hip, l_knee, l_ankle)
                if angle < knee_angle_thresh:
                    violation_frame = frame.copy()
                    violation_side = "left"
                    break

        # 右足判定
        if right_speed < heel_speed_thresh and (r_heel.y > r_ankle.y + heel_height_offset):
            if r_ankle.y > ankle_height_thresh and abs(r_hip.x - r_ankle.x) < horizontal_thresh:
                angle = calculate_3d_angle(r_hip, r_knee, r_ankle)
                if angle < knee_angle_thresh:
                    violation_frame = frame.copy()
                    violation_side = "right"
                    break

        prev_left_heel_y  = l_heel.y
        prev_right_heel_y = r_heel.y
        frame_index += 1

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
