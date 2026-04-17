import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import winsound

# -----------------------------
# CONFIGURATION
# -----------------------------
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 20

MOUTH_AR_THRESH = 0.6

COUNTER = 0
MOUTH_COUNTER = 0
ALARM_ON = False

EAR_HISTORY = []
MAX_HISTORY = 5

HEAD_THRESHOLD = 60  # FIXED (important)

# -----------------------------
# MEDIAPIPE INIT
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

# FIXED mouth landmarks
MOUTH = [13, 14, 78, 308, 82, 312, 87, 317]

frame_count = 0

# -----------------------------
# FUNCTIONS
# -----------------------------
def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_MAR(mouth):
    A = dist.euclidean(mouth[0], mouth[1])
    B = dist.euclidean(mouth[4], mouth[5])
    C = dist.euclidean(mouth[2], mouth[3])
    return (A + B) / (2.0 * C)

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye, right_eye, mouth_points = [], [], []

            # -----------------------------
            # EYES
            # -----------------------------
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))

            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))

            # -----------------------------
            # MOUTH
            # -----------------------------
            for idx in MOUTH:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                mouth_points.append((x, y))

            # -----------------------------
            # EAR
            # -----------------------------
            EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0

            EAR_HISTORY.append(EAR)
            if len(EAR_HISTORY) > MAX_HISTORY:
                EAR_HISTORY.pop(0)

            avg_EAR = sum(EAR_HISTORY) / len(EAR_HISTORY)

            # -----------------------------
            # MAR
            # -----------------------------
            MAR = calculate_MAR(mouth_points)

            # -----------------------------
            # DISPLAY (TOP LEFT)
            # -----------------------------
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"MAR: {MAR:.2f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # -----------------------------
            # DROWSINESS
            # -----------------------------
            if avg_EAR < EYE_AR_THRESH:
                COUNTER += 1
            else:
                COUNTER = 0

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (150, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                ALARM_ON = True
            else:
                ALARM_ON = False

            # -----------------------------
            # YAWNING (IMPROVED)
            # -----------------------------
            if MAR > MOUTH_AR_THRESH:
                MOUTH_COUNTER += 1
            else:
                if MOUTH_COUNTER > 5:
                    cv2.putText(frame, "YAWNING DETECTED", (380, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                MOUTH_COUNTER = 0

            # -----------------------------
            # HEAD POSE (FIXED)
            # -----------------------------
            nose = face_landmarks.landmark[NOSE_TIP]
            nose_x = int(nose.x * w)

            center_x = w // 2

            if nose_x < center_x - HEAD_THRESHOLD:
                direction = "LEFT"
            elif nose_x > center_x + HEAD_THRESHOLD:
                direction = "RIGHT"
            else:
                direction = "FORWARD"

            cv2.putText(frame, f"Looking {direction}", (380, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # -----------------------------
    # ALARM
    # -----------------------------
    if ALARM_ON:
        winsound.Beep(1000, 200)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()