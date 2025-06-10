import cv2
import tempfile
import os

def validate_video_file_single(file_name, file_bytes, allowed_extensions=[".mp4", ".avi", ".mov", ".mkv"]):
   
    """
    Validates a single video file by checking its extension, ensuring it can be opened with OpenCV, and verifying that it contains at least one frame.
    Parameters:
        file_name (str): The name of the video file.
        file_bytes (bytes): The content of the video file as bytes.
        allowed_extensions (list, optional): List of allowed video file extensions. Defaults to [".mp4", ".avi", ".mov", ".mkv"].
    Returns:
        tuple: (bool, str) where the boolean indicates if the file is valid, and the string provides a message describing the result.
    The function performs the following checks:
    - Verifies that the file extension is among the allowed types.
    - Writes the file bytes to a temporary file on disk.
    - Attempts to open the video file using OpenCV.
    - Checks that the video contains at least one frame.
    - Cleans up the temporary file after validation.
    Returns a tuple indicating whether the file is valid and a message describing the validation result.
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
