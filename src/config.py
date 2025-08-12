RELOAD_DATASET = False  
USE_EXISTING_MODEL = False
CONFIG = {
    "model_type": "VIT",
    "val_size" : 0.15,  # 0.1
    "test_size": 0.15,   # 0.3
    "batch_size": 32, 
    "epochs": 2,
    "limit_dataset":7000,
    "dataset_percent": 15,
    "learning_rate": 2e-5,
    "model_names": ["resnet", "vit"],
    "dataset_path": "/content/drive/MyDrive/ml_project/safe-frame-project/data/processed",
    "num_classes": 1,  # שינוי ל-1 עבור binary classification
    "early_stopping_patience": 3,
    "continue_training": False,  # האם להמשיך אימון מודל קיים מ-S3
    "use_augmentation": False,  # האם להשתמש ב-data augmentation
    "train_strategy": "full",  # "full" או "top_layers" - אימון מלא או רק שכבות עליונות 
    "save_dir": "/content/drive/MyDrive/ml_project/safe-frame-project/trained_model/",  
    "checkpoint_dir": "/content/drive/MyDrive/ml_project/safe-frame-project/checkpoints/",
    "model_hf_names": {
        "vit": "google/vit-base-patch16-224-in21k",
        "resnet": "microsoft/resnet-50"
    },
    "loss_type": "focal",
    "loss_params": {
        "gamma": 2.0,
        "alpha": 0.25
    },
}