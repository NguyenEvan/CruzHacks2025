import platform
import subprocess
from win10toast_click import ToastNotifier 
import cv2
import mediapipe as mp
import time
from math import degrees, acos
# import google.generativeai as genai
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

import PIL.Image


def analyze_posture_with_gemini(pil_image):
    load_dotenv()  # Load variables from .env file
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"""Analyze the person's body posture and give some feedback 
    on what's specifically wrong with the posture and 3 bullet points of actional steps to improve the posture. 
    Be specific in your response, limit it to 4-5 sentences. 

    #     Features You May Want to Observe:
    #     - Shoulder Angle 
    #     - Head Forward Displacement 
    #     - Trunk Inclination (degrees from vertical)
    #     - Left Ear-Shoulder Horizontal Distance (relative x)
    #     - Right Ear-Shoulder Horizontal Distance (relative x): 
    #     - Anything else you observe!

    #     Use reasoning to come up with actionable steps the person can take to fix their posture.
    #         - e.g. Move your shoulders back, tilt your head up, stop tilting your head to the side etc.
    #     Write your diagnosis as if you are addressing the person in the picture. 
    #     Don't include markdown formatting, such as asterisks.
    #     Use bullet points.
    #     """
    # image = PIL.Image.open('Screenshot 2024-02-20 220151.png')
    # print("image type", image)

    # print ("pil image type", type(pil_image))
    # print("pil image", pil_image)


    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, pil_image])

    print(response.text)

    return response.text

def send_posture_alert():
    message = "You're slouching! Sit up straight for better posture 🧍"
    title = "Posture Alert"
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
        toaster = ToastNotifier()
        toaster.show_toast(
            title,
            message,
            duration=5,         # seconds
            threaded=True       # allows notification while your main loop runs
        )

       