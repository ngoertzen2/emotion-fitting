import cv2, os, pickle, numpy as np, mediapipe as mp

F_HEIGHT = 350

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Start webcam feed
cap = cv2.VideoCapture(0)

points = []
values = []

trained = False

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
            
            face_height = y_max - y_min
            diff = abs(face_height - F_HEIGHT)
            ratio = min(diff / 100, 1.0)
            color = (0, int((1 - ratio) * 255), int(ratio * 255))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            nose3d = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
            chin3d = np.array([face_landmarks.landmark[152].x, face_landmarks.landmark[152].y, face_landmarks.landmark[152].z])
            forehead3d = np.array([face_landmarks.landmark[10].x, face_landmarks.landmark[10].y, face_landmarks.landmark[10].z])

            # Vector from chin to forehead
            vector = forehead3d - chin3d
            vector /= np.linalg.norm(vector)

            pitch = np.degrees(np.arcsin(-vector[2]))

            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            
            

    cv2.imshow("Dots", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if len(values) > 0:
            filename = input("Enter filename to save data (q to exit without saving): ")

            if filename == 'q':
                break

            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    old_data = pickle.load(f)
                old_points = old_data.get('points', [])
                old_values = old_data.get('values', [])

                print(f"Loaded {len(old_values)} existing samples from {filename}.")
                all_points = old_points + points
                all_values = old_values + values
            else:
                all_points = points
                all_values = values

            with open(filename, 'wb') as f:
                pickle.dump({'points': all_points, 'values': all_values}, f)

            print(f"Saved total of {len(all_values)} samples to {filename}.")
        break

    if cv2.waitKey(1) & 0xFF == ord('c'):
        print("Capturing:")

        list = []
        for p in pts:
            list.append(np.linalg.norm(p - pts[152]))  
        points.append(list)
        values.append(input("Enter emotion value between 0 (Neutral) and 1 (Happy): "))

        print("Captured. " + str(len(values)) + " samples captured")

cap.release()
cv2.destroyAllWindows()
