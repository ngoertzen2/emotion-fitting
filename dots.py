import cv2
import numpy as np
import mediapipe as mp
import random
from scipy.interpolate import splprep, splev
from sklearn.linear_model import LinearRegression

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=2,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Facial feature index groups (based on MediaPipe's 468-landmark model)
landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                        296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                        380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

face_outline =  [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377,
                400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297]

jaw_idx = [93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]


# Start webcam feed
cap = cv2.VideoCapture(0)
display_num = 0

trained = 0
t = False
emName = []
X = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            color = (random.randint(1, 255),random.randint(1, 255),random.randint(1, 255))
            # Convert all landmarks to pixel coordinates
            pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark])
            

            # Draw small circles at all detected landmarks
            match display_num:
                case 0:
                    for i, (px, py) in enumerate(pts):
                        cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                        cv2.putText(frame, str(i), (px + 3, py - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
                case 1:
                    jaw_pts = pts[jaw_idx]
                    for i in jaw_idx:
                        cv2.circle(frame, (pts[i][0], pts[i][1]), 2, (0, 255, 0), -1)

                    x = jaw_pts[:, 0]
                    y = jaw_pts[:, 1]
                    degree = 4
                    coeffs = np.polyfit(x, y, degree)
                    poly_y = np.polyval(coeffs, x)

                    for i in range(len(x) - 1):
                        cv2.line(frame, (int(x[i]), int(poly_y[i])),
                                 (int(x[i + 1]), int(poly_y[i + 1])), (255, 0, 0), 2)
                case 2:
                    for i, (px, py) in enumerate(pts):
                        if ( i == 1 or i == 10 or i == 152):
                            cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)
                            cv2.putText(frame, str(i), (px + 3, py - 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

                case _:
                    for i in face_outline:
                        cv2.circle(frame, (pts[i][0], pts[i][1]), 2, (0, 0, 255), -1)

    cv2.imshow("Dots", frame)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        display_num = display_num + 1 if display_num < 3 else 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
