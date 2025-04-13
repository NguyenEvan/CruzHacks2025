from pose.detector import process_frame, draw_landmarks
from slouch_logic import detect_slouch
from gemini.feedback import send_posture_alert, analyze_posture_with_gemini
from collections import deque


MAX_SCORE = 200
slouch_score = 0
slouch_triggered = False

def process_frame_with_posture(frame, pose_model, frame_buffer, landmarks_callback=None):
    global slouch_score, slouch_triggered
    processed_frame, results = process_frame(frame, pose_model)

    if results.pose_landmarks:
        draw_landmarks(processed_frame, results.pose_landmarks)
        landmarks = results.pose_landmarks.landmark
        z_score = detect_slouch(landmarks)
        slouching = z_score < -0.85
        # print("score", z_score)
        if slouching:
            slouch_score = min(slouch_score + 1, MAX_SCORE)
        else:
            slouch_score = max(slouch_score - 1, 0)

        report = ""

        if slouch_score > 0.75 * MAX_SCORE:
            print("⚠️ Slouching detected!")
            #placeholder
            #get lowest z_average from frame buffer
            min_z = float('inf')
            min_img = None
            for z_score, img in reversed(frame_buffer):
                # print("z_score type:", type(z_score))
                if z_score < min_z:
                    min_z = z_score 
                    min_img = img
            
            report = analyze_posture_with_gemini(min_img)
            print(report)
            #done doing gemini report

            frame_buffer = deque()

            send_posture_alert()
            slouch_score = 0

        if slouching and not slouch_triggered:
            slouch_triggered = True
        elif not slouching:
            slouch_triggered = False

    return processed_frame, z_score, report
