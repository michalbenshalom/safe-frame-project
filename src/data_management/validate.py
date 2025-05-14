import cv2
import numpy as np
import tempfile
import os
import json

def validate_video_files(files_dict: dict, allowed_extensions: list[str] = [".mp4", ".avi", ".mov", ".mkv"]) -> dict:
    """
    Validates video files extracted from ZIPs.
    Saves result report to 'video_validation_results.json'.
    """
    results = {}

    for zip_name, inner_files in files_dict.items():
        for file_name, content in inner_files.items():
            full_name = f"{zip_name}/{file_name}"

            # Check if file extension is valid
            if not any(file_name.lower().endswith(ext) for ext in allowed_extensions):
                results[full_name] = "Not a supported video file"
                continue

            try:
                # Save video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name

                # Try to open with OpenCV
                cap = cv2.VideoCapture(temp_file_path)

                if not cap.isOpened():
                    results[full_name] = " File could not be opened with OpenCV"
                else:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        results[full_name] = "No frames found in video file"
                    else:
                        results[full_name] = " Valid"
                
                cap.release()
                os.remove(temp_file_path)

            except Exception as e:
                results[full_name] = f"Error during validation: {str(e)}"

    # Save results to JSON file
    with open("video_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(" Results saved to video_validation_results.json")

    return results
