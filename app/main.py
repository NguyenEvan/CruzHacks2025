import cv2
from pose.detector import init_pose, process_frame, draw_landmarks
import mediapipe as mp
from math import degrees, acos

from gemini.feedback import analyze_posture_with_gemini

# Pose landmark indexes
mp_pose = mp.solutions.pose

# Pose landmark indexes
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
NOSE = mp_pose.PoseLandmark.NOSE.value
LEFT_EAR = mp_pose.PoseLandmark.LEFT_EAR.value
RIGHT_EAR = mp_pose.PoseLandmark.RIGHT_EAR.value



def detect_slouch(landmarks):
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    nose = landmarks[NOSE]
    avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    slouching = abs(nose.x - avg_shoulder_x) > 0.04
    slouching_features = None
    return slouching
    if slouching:
        slouching_features = detect_slouch_gemini(landmarks=landmarks)
    
    if slouching_features and slouching_features["is_slouching"]:
        return True
    return False


def detect_slouch_gemini(landmarks):
    """
    Determine if the user's nose is significantly forward of the shoulders.
    """
    # avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    # slouching = abs(nose.x - avg_shoulder_x) > 0.04

    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    nose = landmarks[NOSE]
    left_ear = landmarks[LEFT_EAR]
    right_ear = landmarks[RIGHT_EAR]

    # Calculate shoulder angle
    shoulder_vector = (right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y)
    neck_point = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2 - 0.1) # Approximate neck
    vector1 = (left_shoulder.x - neck_point[0], left_shoulder.y - neck_point[1])
    vector2 = (right_shoulder.x - neck_point[0], right_shoulder.y - neck_point[1])
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = (vector1[0]**2 + vector1[1]**2)**0.5
    magnitude2 = (vector2[0]**2 + vector2[1]**2)**0.5
    shoulder_angle_rad = acos(dot_product / (magnitude1 * magnitude2)) if (magnitude1 * magnitude2) > 0 else 0
    shoulder_angle_deg = degrees(shoulder_angle_rad)

    # Calculate head position relative to shoulders (forward lean)
    head_forward_displacement = nose.z - (left_shoulder.z + right_shoulder.z) / 2

    # Calculate ear-shoulder horizontal distance (proxy for forward head)
    left_ear_shoulder_x = left_ear.x - left_shoulder.x
    right_ear_shoulder_x = right_ear.x - right_shoulder.x

    features = {
        "shoulder_angle": shoulder_angle_deg,
        "head_forward_displacement": head_forward_displacement,
        "left_ear_shoulder_x": left_ear_shoulder_x,
        "right_ear_shoulder_x": right_ear_shoulder_x,
        # Add more features as needed
    }

    slouch_features = analyze_posture_with_gemini(features=features)
    
    return slouch_features

    avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    slouching = abs(nose.x - avg_shoulder_x) > 0.04
    return slouching

def run_posture_detection():
    pose_model = init_pose()
    cap = cv2.VideoCapture(0)
    slouch_triggered = False
    i = 0

    print("Starting posture detection... Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        processed_frame, results = process_frame(frame, pose_model)

        if results.pose_landmarks:
            draw_landmarks(processed_frame, results.pose_landmarks)

            landmarks = results.pose_landmarks.landmark
            slouching = detect_slouch(landmarks)

            if slouching:
                print(i, "YOU ARE SLOUCHING")
                i += 1

            if slouching and not slouch_triggered:
                # send_posture_alert()
                slouch_triggered = True
            elif not slouching:
                slouch_triggered = False

        cv2.imshow("Posture Tracker", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    #making features
    
    run_posture_detection()

