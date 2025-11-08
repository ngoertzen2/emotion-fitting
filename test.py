import cv2
import numpy as np
import mediapipe as mp
from sklearn.linear_model import LinearRegression

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Landmark indices for features
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
EYEBROW_TOP, EYE_TOP = 70, 105

# Storage for training samples
X, y = [], []
trained = False

def extract_features(pts):
    """Compute geometric facial features."""
    mouth_open = np.linalg.norm(pts[MOUTH_TOP] - pts[MOUTH_BOTTOM])
    smile_width = np.linalg.norm(pts[MOUTH_LEFT] - pts[MOUTH_RIGHT])
    eyebrow_raise = np.linalg.norm(pts[EYEBROW_TOP] - pts[EYE_TOP])
    return [mouth_open, smile_width, eyebrow_raise]

# Start webcam
cap = cv2.VideoCapture(0)
model = LinearRegression()

print("Press 'r' to record a sample (it will ask for emotion label).")
print("Press 't' to train model.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark])

            # Draw points for visualization
            for idx in [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT, EYEBROW_TOP, EYE_TOP]:
                cv2.circle(frame, (pts[idx][0], pts[idx][1]), 2, (0, 255, 0), -1)

            # Extract features
            features = extract_features(pts)

            # Predict if trained
            if trained:
                pred = model.predict([features])[0]
                emotion = "Happy" if pred > 0.5 else "Neutral"
                cv2.putText(frame, f"Predicted: {emotion}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Emotion Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # Record new sample
    if key == ord('r') and results.multi_face_landmarks:
        label = input("Enter label (0=neutral, 1=happy): ")
        features = extract_features(pts)
        X.append(features)
        y.append(float(label))
        print(f"Recorded sample: {features} -> {label}")

    # Train model
    elif key == ord('t') and len(X) >= 2:
        model.fit(X, y)
        trained = True
        print("Model trained using least squares regression!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
