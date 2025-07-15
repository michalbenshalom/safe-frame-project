RELOAD_DATASET = False  
MODEL_TYPE = "VIT"      #  "VIT"  או "ResNet"
USE_EXISTING_MODEL = False
CONFIG = {
    "val_size" : 0.005,  # 0.1
    "test_size": 0.99,   # 0.3
    "batch_size": 32, 
    "epochs": 2,
    "learning_rate": 2e-5,
    "model_names": ["resnet", "vit", "blip", "llava"],
    "dataset_path": "/content/drive/MyDrive/ml_project/safe-frame-project/data/processed",
    "num_classes": 1,
    "early_stopping_patience": 3,
    "save_dir": "C:/Users/MichalBenShalom/OneDrive - Infraedge LTD/Desktop/הורדות/course/Project/trained_model/",  
    "checkpoint_dir": "./checkpoints/",
    "model_hf_names": {
        "vit": "google/vit-base-patch16-224-in21k",
        "resnet": "microsoft/resnet-50",
        "blip": "Salesforce/blip-image-classification-base",
        "llava": "liuhaotian/llava-v1.5-7b"
    },
    "loss_type": "bce",  # או cross_entropy
    "loss_params": {
        "pos_weight": None  # לדוגמה: torch.tensor([2.0])
    },
}