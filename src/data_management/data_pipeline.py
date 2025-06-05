from data_management.s3_manager import stream_s3_videos
from data_management.loader import video_to_frames
from data_management.validate import validate_video_file_single

def process_and_validate_videos():
    """
    Pipeline to download videos, extract frames, and validate them.
    """

    # עיבוד קבצים
    for folder, name, content in stream_s3_videos():
        results = validate_video_file_single(name, content)

        frames = video_to_frames(content, fps_interval=1)
        
        print(f"{folder}/{name} → {len(frames)} frames extracted")


    return results