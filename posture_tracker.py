import cv2
import mediapipe as mp
# from feedback import send_posture_alert  # ✅ Import the alert function

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Pose landmark indexes
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
NOSE = mp_pose.PoseLandmark.NOSE.value

# Slouch detection state
slouch_triggered = False

# Start webcam
cap = cv2.VideoCapture(0)

print("Starting posture detection... Press 'q' to quit.")
i = 0

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Run pose detection
    results = pose.process(image)

    # Convert back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

        # ✅ Get shoulder and nose positions
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        # print("left shoulder", left_shoulder)
        # print("right shoulder", right_shoulder)
        # print("landmarks", landmarks)
        nose = landmarks[NOSE]

        # ✅ Calculate average shoulder x and y
        avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        # ✅ Determine slouching if nose is far in front of shoulders (in x-axis)
        slouching = abs(nose.x - avg_shoulder_x) > 0.04  # adjust threshold as needed
        if slouching:
            print(i, "YOU ARE SLOUCHING")
            i += 1
            

        if slouching and not slouch_triggered:
            # send_posture_alert()
            slouch_triggered = True  # Avoid spamming notifications
        elif not slouching:
            slouch_triggered = False  # Reset if user sits up again

    # Display the result
    cv2.imshow("Posture Tracker", image)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()