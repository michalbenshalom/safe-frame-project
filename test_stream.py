from src.data_management.s3_manager import stream_s3_videos

for folder, name, content in stream_s3_videos():
    print(f"📁 {folder} | 🎥 {name} | {len(content)} bytes")
    break  # רק לראות שזה עובד
