import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import winsound

# -----------------------------
# CONFIGURATION
# -----------------------------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20

COUNTER = 0
ALARM_ON = False

# -----------------------------
# INITIALIZE MEDIAPIPE
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False
)

# -----------------------------
# START CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

frame_count = 0

# -----------------------------
# EAR FUNCTION
# -----------------------------
def calculate_EAR(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
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

            left_eye = []
            right_eye = []

            # LEFT EYE
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # RIGHT EYE
            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # EAR calculation
            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)
            EAR = (left_EAR + right_EAR) / 2.0

            # Display EAR
            cv2.putText(frame, f"EAR: {EAR:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # DROWSINESS DETECTION
            if EAR < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    ALARM_ON = True
            else:
                COUNTER = 0
                ALARM_ON = False

    # CONTINUOUS ALARM (NON-BLOCKING STYLE)
    if ALARM_ON:
        winsound.Beep(1000, 200)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()