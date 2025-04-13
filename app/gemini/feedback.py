
import platform
import subprocess

import cv2
import mediapipe as mp
import time
from math import degrees, acos


from google import genai

# # Initialize Gemini API
# gen.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your actual API key
# model = gen.GenerativeModel('gemini-pro')

# client = genai.Client(api_key="AIzaSyAXFxDw2-tKAbE0CzICyr_r3Wp2lbPaU6M")

# response = client.models.generate_content(
#     model="gemini-2.0-flash", contents="Explain how AI works in a few words"
# )
# print(response.text)


def analyze_posture_with_gemini(features):
    """Sends posture features to Gemini API for analysis."""
    if not features:
        return None
    prompt = f"""Analyze the following body posture features to determine if the person is likely slouching. 

    Features:
    - Shoulder Angle (degrees): {features.get('shoulder_angle', 'N/A')}
    - Head Forward Displacement (relative z): {features.get('head_forward_displacement', 'N/A')}
    - Trunk Inclination (degrees from vertical): {features.get('trunk_inclination', 'N/A')}
    - Left Ear-Shoulder Horizontal Distance (relative x): {features.get('left_ear_shoulder_x', 'N/A')}
    - Right Ear-Shoulder Horizontal Distance (relative x): {features.get('right_ear_shoulder_x', 'N/A')}

    Use the above features to compute a slouching score between 0 (no slouch) and 1 (strong slouch).
    Then, determine if the person is slouching (score >= 0.75 means slouching).

    Respond ONLY with the final result in the EXACT following format:
    [slouching score: <number between 0 and 1>, is_slouching: <true or false>]

    Use reasoning to come up with the score, but Do NOT output any reasoning, explanation, or any additional text.
    """


    client = genai.Client(api_key="AIzaSyAXFxDw2-tKAbE0CzICyr_r3Wp2lbPaU6M")

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        if response.text:
            response_text = response.text
            print(f"Gemini Response: {response_text}")
            # Implement more sophisticated parsing based on Gemini's response format
            slouching_score = None
            is_slouching = False
            if "slouching score:" in response_text.lower():
                try:
                    start_index = len("[slouching score: ")
                    end_index = response_text.index(',')
                    score_str = float(response_text[start_index: end_index])     
                except ValueError:
                    print("Could not parse slouching score.")
            if "is_slouching: true" in response_text.lower():
                is_slouching = True
            elif "is_slouching: false" in response_text.lower():
                is_slouching = False

            
            return {"score": slouching_score, "is_slouching": is_slouching}
        else:
            print("Gemini API response was empty.")
            return None
    except Exception as e:
        print(f"Error communicating with Gemini API: {e}")
        return None

def send_posture_alert():
    message = "You're slouching! Sit up straight for better posture 🧍"
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        subprocess.run([
            "osascript", "-e",
            f'display notification "{message}" with title "Posture Alert"'
        ])
    
    elif system == "Linux":
        subprocess.run([
            "notify-send", "Posture Alert", message
        ])
    
    elif system == "Windows":
        print(f"Notification: {message}")
