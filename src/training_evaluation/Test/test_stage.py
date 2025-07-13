
import torch
from torch.utils.data import DataLoader
from model_wrapper.models.model_wrapper_factory  import ModelWrapperFactory
from model_wrapper.models.base_model_wrapper import BaseModelWrapper
from training_evaluation.Evaluation.evaluation import evaluate_model
from src.config import MODEL_TYPE, CONFIG
from utils.ModelsTypes import MODEL_WRAPPERS
from torch.utils.data import Subset

def test_model(model, test_dataset=None, weights_path=None,):
   
    if MODEL_TYPE not in MODEL_WRAPPERS:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    # Load data
    test_dataset = Subset(test_dataset, list(range(100))) #michalbs
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model_wrapper = MODEL_WRAPPERS[MODEL_TYPE]() #ModelWrapperFactory.create(MODEL_TYPE, CONFIG)
    if not isinstance(model_wrapper, BaseModelWrapper):
        raise TypeError("ModelWrapper must inherit from BaseModelWrapper")

    model_wrapper.set_model(model)

    # Evaluate
    eval_result = evaluate_model(model_wrapper, test_loader, device)
    return eval_result


