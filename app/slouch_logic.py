# app/slouch_logic.py
import mediapipe as mp

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
    # print("left shoulder x", left_shoulder.x)
    # print("right shoulder x", right_shoulder.x)
    z_face_values = []
    for i in range(0, 11):
        z_face_values.append(landmarks[i].z)
    average_z = sum(z_face_values) / len(z_face_values)

    head_forward_displacement = nose.z - (left_shoulder.z + right_shoulder.z) / 2
    # print("head forward displacement", head_forward_displacement)


    z_to_should_displace = average_z - (left_shoulder.z + right_shoulder.z) / 2 
    #print ("average z to shoulder didsplacement", z_to_should_displace)
    slouching_z = z_to_should_displace < -0.80 #adjust based on testing

    slouching_1 = head_forward_displacement < -1.00  # adjust based on testing
    slouching_2 = abs(nose.x - avg_shoulder_x) > 0.04
    # print("slouching 2", abs(nose.x - avg_shoulder_x))
    #print("slouching 1", head_forward_displacement)
    slouching_features = None
    return z_to_should_displace
    if slouching:
        slouching_features = detect_slouch_gemini(landmarks=landmarks)
    
    if slouching_features and slouching_features["is_slouching"]:
        return True
    return False
