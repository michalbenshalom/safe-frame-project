from config import CONFIG
from data.dataset_loader import load_dataset
from train.trainer import train_model
from train.evaluation import evaluate_model
from models.vit_model import get_vit_model
from models.resnet_model import get_resnet_model
# from models.blip_model import get_blip_model
# from models.llava_model import get_llava_model  

MODEL_REGISTRY = {
    "vit": get_vit_model,
    "resnet": get_resnet_model,
    # "blip": get_blip_model,
    # "llava": get_llava_model
}

def model_pipeline():
    train_loader, val_loader = load_dataset(CONFIG["dataset_path"], CONFIG["batch_size"])

    for model_name in CONFIG["model_names"]:
        print(f"=== Training {model_name} ===")
        model_fn = MODEL_REGISTRY[model_name]
        hf_name = CONFIG["model_hf_names"][model_name]
        model = model_fn(CONFIG["num_classes"], hf_name)
        #do train flag
        train_model(model, train_loader, val_loader, CONFIG, model_name)
        #do eval flag
        #evaluate_model(model, val_loader, CONFIG)

