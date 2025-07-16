import requests

def call_predict_video_api():
    print("1")
    video_path = r"C:\Users\MichalBenShalom\OneDrive - Infraedge LTD\Desktop\הורדות\אישי\תמונות\VID-20230910-WA0006.mp4"
    url = "http://localhost:8000/predict_video"
    try:
        print("2")
        with open(video_path, "rb") as f:
            print("3")
            files = {"file": (video_path, f, "video/mp4")}
            print("4")
            response = requests.post(url, files=files)
            print("5")
            print("Status Code:", response.status_code)

            print("Response JSON:", response.json())
    except Exception as e:
        print("Raw Response Text:", response.text)
        print("Failed to parse JSON:", e)

if __name__ == "__main__":
    call_predict_video_api()