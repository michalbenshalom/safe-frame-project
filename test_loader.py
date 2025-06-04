from src.data_management.s3_manager import stream_s3_videos
from src.data_management.loader import video_to_frames

for folder, name, content in stream_s3_videos():
    frames = video_to_frames(content, fps_interval=1)
    print(f"{folder}/{name} → {len(frames)} frames extracted")
    break
