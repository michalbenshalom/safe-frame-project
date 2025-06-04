from src.data_management.s3_manager import stream_s3_videos
from src.data_management.validate import validate_video_file_single

for folder, name, content in stream_s3_videos():
    is_valid, msg = validate_video_file_single(name, content)
    print(f"{folder}/{name} → {msg}")
    break  
