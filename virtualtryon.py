import cv2
import mediapipe as mp

# Initialize mediapipe face mesh detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Load sunglasses image with alpha channel
sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)  # Must have alpha channel

# Start video capture from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye coordinates (landmark indices)
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            h, w, _ = frame.shape
            x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
            x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

            # Compute sunglasses width and resize
            glasses_width = int(1.3 * abs(x2 - x1))
            glasses_height = int(sunglasses.shape[0] * (glasses_width / sunglasses.shape[1]))
            resized_glasses = cv2.resize(sunglasses, (glasses_width, glasses_height))

            # Compute top-left position
            x = int((x1 + x2) / 2 - glasses_width / 2)
            y = int((y1 + y2) / 2 - glasses_height / 2)

            # Overlay sunglasses with alpha blending
            for i in range(resized_glasses.shape[0]):
                for j in range(resized_glasses.shape[1]):
                    if y + i >= frame.shape[0] or x + j >= frame.shape[1] or y + i < 0 or x + j < 0:
                        continue
                    alpha = resized_glasses[i, j, 3] / 255.0
                    if alpha > 0:
                        for c in range(3):
                            frame[y + i, x + j, c] = int(
                                alpha * resized_glasses[i, j, c] + (1 - alpha) * frame[y + i, x + j, c])

    # Show the frame
    cv2.imshow("Live Try-On", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

