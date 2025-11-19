import cv2, os, sys, pickle, numpy as np, mediapipe as mp

F_HEIGHT = 350

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
    # Convert to NumPy arrays
    # shape (n_samples, n_features)
    X = np.array(points, dtype=float)   
    # shape (n_samples,)
    y = np.array(values, dtype=float)   

    n, d = X.shape

    # Build design matrix X (with bias column),
    # X is n x (p+1): first column all 1's, then features x_{ij}
    X_design = np.hstack([np.ones((n, 1)), X])
    d2 = d + 1  # = p+1

    # Compute X^T X  
    XtX = np.zeros((d2, d2))
    # row index
    for i in range(d2):  
        # column index          
        for j in range(d2):        
            total = 0.0
            # loop over samples
            for k in range(n):     
                total += X_design[k][i] * X_design[k][j]
            XtX[i][j] = total

    # Compute X^T y using nested loops
    Xty = np.zeros(d2)
    for i in range(d2):
        total = 0.0
        for k in range(n):
            total += X_design[k][i] * y[k]
        Xty[i] = total

    try:
    # Ideal theoretical case: (X^T X) θ = X^T y
        theta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
    # If X^T X is singular, fall back to Moore–Penrose pseudoinverse
        theta = np.linalg.pinv(X_design) @ y

    # b0 (intercept)
    bias = theta[0]    

    # b1..bp (slopes)  
    weights = theta[1:]  
else: 
    print("Not enough data. Exiting.")
    sys.exit(0)

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

    bar_width = 40
    x1 = w - 10  
    x2 = w - (10 + bar_width)
    mid_y = int(h - (1 / 3) * h)

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
            
            #Calculate emotion level using linear regression(least squares)
            X_test = [np.linalg.norm(p - pts[152]) for p in pts]  # feature vector

            # y_pred = bias + sum_j w_j * x_j   (explicit math version)
            y_pred = bias
            for w_i, x_i in zip(weights, X_test):
                y_pred += w_i * x_i
            y_pred = float(y_pred)

            #Display calculated value as bar on the right side
            y_clamped = max(-1, min(2, y_pred))

            norm = (y_clamped + 1) / 3

            bar_height = int(norm * h)   
            y1 = h - bar_height # top of bar
            y2 = h # bottom of bar

            #Color bar based on value
            red = int((1 - norm) * 255)
            green = int(norm * 255)
            blue  = 0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (blue, green, red), -1)
            
            cv2.putText(frame, f"Emotion: {y_pred:.2f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #Outline for emotion value bar
    cv2.line(frame, (x1, mid_y), (x2, mid_y), (255, 255, 255), 1)
    cv2.putText(frame, "0", (x1 - (bar_width + 26), mid_y + 8),
    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, 0), (x2, h), (255, 255, 255), 2)

    cv2.imshow("Dots", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
