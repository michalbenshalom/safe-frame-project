from tkinter import Image
from fastapi import File, UploadFile
from loguru import logger
from matplotlib import transforms
import torch

from config import MODEL_TYPE
from data_management import s3_manager
from data_management.loader import video_to_frames
from utils.ModelsTypes import MODEL_WRAPPERS


async def predict_from_video_file(file: UploadFile = File(...)):
    try:
        # קרא את קובץ הווידאו
        video_bytes = await file.read()
        
        # חילוץ פריימים
        frames = video_to_frames(video_bytes, fps_interval=1)
        if not frames:
            return {"error": "No frames extracted from video."}
        model_wrapper = MODEL_WRAPPERS[MODEL_TYPE]()  
        
        # טען את המודל
        try:
            filename = model_wrapper.get_best_model_filename()
            s3_path = f"Models/{MODEL_TYPE}/{filename}"
            s3_manager.load_model(model_wrapper.model, s3_path)
        except Exception as e:
            logger.warning(f"Failed to load existing model. Training a new one. Error: {e}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_wrapper.model = model_wrapper.model.to(device)
        model_wrapper.model.eval()

        # חיזוי לכל פריים
        preds = []
        for _, frame in frames:
            pred = predict_single_frame(frame, model_wrapper.model, model_wrapper, device)
            preds.append(pred)

        avg_pred = float(sum(preds) / len(preds))
        return {
            "num_frames": len(preds),
            "avg_prediction": avg_pred,
            "frame_predictions": preds
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
    

def predict_single_frame(image, model_wrapper, device):
    """
    מקבל פריים (np.ndarray), מחזיר הסתברות של סיווג כ־1 (float).
    """
    # המרה ל־PIL והכנה לפי הגדרות המודל
    img_pil = Image.fromarray(image)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_wrapper.forward_pass(tensor)
        pred = torch.sigmoid(output).item()
    return pred
