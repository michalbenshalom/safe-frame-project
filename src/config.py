

CONFIG = {
    "reload_dataset": True  ,
    "model_type" : "ViT" , #"ViT" or "ResNet"
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 2e-5,
    "model_names": ["resnet", "vit", "blip", "llava"],
    "dataset_path": "./data/images/",
    "num_classes": 10,
    "device": "cuda",

    "checkpoint_dir": "./checkpoints/",
    "model_hf_names": {
        "vit": "google/vit-base-patch16-224-in21k",
        "resnet": "microsoft/resnet-50",
        "blip": "Salesforce/blip-image-classification-base",
        "llava": "liuhaotian/llava-v1.5-7b"
    }
}