import cv2, os, sys, pickle, numpy as np, mediapipe as mp
from sklearn.linear_model import LinearRegression

filename = input("Enter filename for data: ")

if os.path.exists(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    points = data.get("points", [])
    values = data.get("values", [])
    print(f"Loaded {len(values)} samples from {filename}")
else:
    print("File not found. Exiting.")
    sys.exit(0)
if len(values) >= 2 :
    model = LinearRegression()
    model.fit(points, values)
else: 
    print("Not enough data. Exiting.")
    sys.exit(0)

emotion_images = {
    "happy": cv2.imread("happy.png"),
    "neutral": cv2.imread("neutral.png"),
    "sad": cv2.imread("sad.png"),
}

# Resize them to a consistent display size
for key in emotion_images:
    if emotion_images[key] is not None:
        emotion_images[key] = cv2.resize(emotion_images[key], (300, 300))

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)


    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert all landmarks to pixel coordinates
            pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark])

            # Draw box arround the face
            x_min, y_min = np.min(pts[:, 0]), np.min(pts[:, 1])
            x_max, y_max = np.max(pts[:, 0]), np.max(pts[:, 1])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            face_height = y_max - y_min
            
            #Calculate emotion level using linear regression(least squares)
            X_test = [np.linalg.norm(p - pts[152]) for p in pts]

            y_pred = model.predict([X_test])[0]

            cv2.putText(frame, f"Emotion: {y_pred:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if y_pred > 0.66:
                emotion_img = emotion_images["happy"]
            elif y_pred > 0.0:
                emotion_img = emotion_images["neutral"]
            else:
                emotion_img = emotion_images["sad"]

            # Display emotion image in another window
            if emotion_img is not None:
                cv2.imshow("Emotion Display", emotion_img)

    cv2.imshow("Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
