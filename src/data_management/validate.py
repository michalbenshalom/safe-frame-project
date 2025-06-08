import cv2
import tempfile
import os

def validate_video_file_single(file_name, file_bytes, allowed_extensions=[".mp4", ".avi", ".mov", ".mkv"]):
    """
    מקבל קובץ וידאו (כ־bytes) + שם קובץ, ומוודא שהוא:
    - בסיומת תקפה
    - נפתח בהצלחה עם OpenCV
    - יש בו לפחות פריים אחד
    """
    if not any(file_name.lower().endswith(ext) for ext in allowed_extensions):
        return False, "Unsupported file extension"

    try:
        # כתיבה זמנית של הקובץ לדיסק
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            os.remove(tmp_path)
            return False, "Cannot open video"

        ret, frame = cap.read()
        cap.release()
        os.remove(tmp_path)

        if not ret or frame is None:
            return False, "No frames found"

        return True, "Valid"

    except Exception as e:
        return False, f"Error: {str(e)}"
