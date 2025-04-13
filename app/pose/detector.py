import cv2
import mediapipe as mp

# Initialize MediaPipe pose components globally
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def init_pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5):
    """
    Initialize the MediaPipe Pose model.
    """
    return mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        enable_segmentation=enable_segmentation,
        min_detection_confidence=min_detection_confidence
    )

def process_frame(frame, pose_model):
    """
    Process a frame to extract pose landmarks using MediaPipe.
    Returns:
        - Annotated image
        - Pose landmarks result object
    """
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = rgb.shape
    rgb.flags.writeable = False
    results = pose_model.process(rgb)
    rgb.flags.writeable = True

    # Convert back to BGR
    output_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    return output_frame, results

def draw_landmarks(image, landmarks):
    """
    Draw pose landmarks and connections on the image.
    """
    if landmarks:
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
