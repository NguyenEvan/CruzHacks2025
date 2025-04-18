# CruzHacks2025

## By: Wilson Xie, Evan Nguyen, Austin Lien

## 💻 How to Run

To launch Slouching Slugs, make sure you have Python 3.10+ and the required packages installed.

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/slouching-slugs.git
cd slouching-slugs
pip install -r requirements.txt
python ./app/gui_app.py
```


## Inspiration

As STEM students, we spend countless hours hunched over screens, often ignoring our posture until the pain kicks in. Yet, we easily notice when others slouch! We realized that if we had a real-time reminder to sit straight, we could prevent long-term damage. That’s why we built Slouching Slugs—an AI-powered posture guard that nudges you before bad habits occur.

## What it does

We utilize the built-in webcam or an external one and send a live feed to our program that detects if the user is slouching or not, and if so, a notification is sent to the user's desktop as well as a pop-up with advice on how they can improve 

## How we built it

We only used Python to code this, as it offers the most diverse range of libraries we can utilize. Some important libraries we used were mediapipe to help map out the important points on the body, such as the shoulders, nose, and eyes. Then, we created our own program to determine if the user is slouching or not utilizing those points. We then take a snapshot of the slouch and send that to gemini AI to provide advice to the user on what and how to improve. 

## Challenges we ran into

A big challenge we encountered was adjusting the sensitivity of the slouch detector; at times, it was too sensitive, and just a slight movement would cause it to send a notification. 

## Accomplishments that we're proud of

We are most proud of being able to fully commit ourselves to this project and achieve what we have done today within 48 hours. As this was our first hackathon, we came in with low expectations; however, those were easily exceeded. 

## What we learned

We learned how to utilize more technologies, such as the variety of Python libraries we used, and gained more experience with AI and LLMS through Gemini. Beyond tech, we also better understood how our postures can affect us in the long run and key points to look out for. Alongside how we will approach the next hackathon with better time management and organization and not be afraid to branch out and go all-in on an ambitious project. 

## What's next for Slouching Slugs

We plan on scaling this more through creating our own convolutional neural network and training it to our standards by having actual people be subjects and documenting their posture to provide our program with much better accuracy. Also, we would want to improve our user interface and include more features and customization for our users. 