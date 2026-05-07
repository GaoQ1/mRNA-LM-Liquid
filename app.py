import os
import torch
import numpy as np
import gc
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from transformers import Trainer, TrainingArguments
import argparse

from FullModel import FullModel
from dataload import *

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Parse command line arguments
parser = argparse.ArgumentParser(description='mRNA-LM Prediction API')
parser.add_argument('--device', type=int, default=1, help='CUDA device index to use')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind to')
parser.add_argument('--port', type=int, default=9020, help='Port to listen on')
parser.add_argument('--max_models', type=int, default=1, help='Maximum number of models to keep in memory per device')
args, _ = parser.parse_known_args()

# Define the CUDA device
CUDA_DEVICE = args.device
MAX_MODELS_PER_DEVICE = args.max_models
print(f"Using CUDA device: {CUDA_DEVICE}")
print(f"Maximum models per device: {MAX_MODELS_PER_DEVICE}")
if torch.cuda.is_available():
    torch.cuda.set_device(CUDA_DEVICE)
    print(f"Using GPU: {torch.cuda.get_device_name(CUDA_DEVICE)}")
else:
    print("CUDA is not available. Using CPU.")

# Initialize FastAPI app
app = FastAPI(
    title="mRNA-LM Prediction API",
    description="API for predicting using various mRNA models",
    version="1.0.0"
)

# Define model configurations
MODEL_CONFIGS = {
    "tr": {
        "path": "outputs/tr/finetuned_model.pt",
        "num_labels": 1,
        "class_weights": []
    },
    "halflife": {
        "path": "outputs/halflife/finetuned_model.pt",
        "num_labels": 1,
        "class_weights": []
    },
    "liver": {
        "path": "outputs/liver/finetuned_model.pt",
        "num_labels": 1,
        "class_weights": []
    },
    "hek293t": {
        "path": "outputs/hek293t/finetuned_model.pt",
        "num_labels": 1,
        "class_weights": []
    }
}

# Default model parameters
DEFAULT_PARAMS = {
    "lorar": 32,
    "lalpha": 32,
    "ldropout": 0.5,
    "head_dim": 768,
    "head_dropout": 0.5,
    "useCLIP": False,
    "temperature": 0.07,
    "coefficient": 0.2,
    "device": CUDA_DEVICE
}

# Dictionary to store loaded models and their last usage timestamp
loaded_models = {}
model_usage_order = []  # Track the order of model usage for LRU eviction

# Input data models
class SequenceInput(BaseModel):
    utr5: str = Field(..., description="5' UTR sequence")
    cds: str = Field(..., description="CDS sequence")
    utr3: str = Field(..., description="3' UTR sequence")
    
class BatchSequenceInput(BaseModel):
    sequences: List[SequenceInput] = Field(..., description="List of sequences to predict")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Optional model parameters to override defaults")

# Helper function to clear CUDA cache and force garbage collection
def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
# Helper function to unload models when memory limit is reached
def manage_model_memory(device_id, new_model_key):
    # Count models on the specific device
    device_models = [key for key in loaded_models.keys() if key.endswith(f"_device_{device_id}")]
    
    # If we're not at the limit, no need to unload
    if len(device_models) < MAX_MODELS_PER_DEVICE:
        return
    
    # If the new model is already loaded, just update its position in the usage order
    if new_model_key in loaded_models:
        if new_model_key in model_usage_order:
            model_usage_order.remove(new_model_key)
        model_usage_order.append(new_model_key)
        return
    
    # Find the least recently used model on this device
    lru_model = None
    for model in model_usage_order:
        if model.endswith(f"_device_{device_id}") and model != new_model_key:
            lru_model = model
            break
    
    # If we found a model to unload
    if lru_model:
        print(f"Unloading model {lru_model} to free GPU memory")
        # Remove from dictionaries
        if lru_model in loaded_models:
            # Delete the model and remove references
            del loaded_models[lru_model]
            model_usage_order.remove(lru_model)
            # Force garbage collection and clear CUDA cache
            clear_gpu_memory()
            print(f"Model {lru_model} unloaded, memory freed")

# Helper function to load model
def load_model(model_name, device_id=None):
    # Use the specified device_id or default to the global CUDA_DEVICE
    cuda_device = device_id if device_id is not None else CUDA_DEVICE
    
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Create a model key that includes the device to support the same model on different devices
    model_key = f"{model_name}_device_{cuda_device}"
    
    # Update model usage order (for LRU tracking)
    if model_key in model_usage_order:
        model_usage_order.remove(model_key)
    model_usage_order.append(model_key)
    
    # If model is already loaded, return it
    if model_key in loaded_models:
        print(f"Using already loaded model {model_key}")
        return loaded_models[model_key]
    
    # Before loading a new model, check if we need to unload others
    manage_model_memory(cuda_device, model_key)
    
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found at {model_path}")
    
    # Initialize model
    print(f"Initializing new model {model_key}")
    model = FullModel(
        config["num_labels"],
        config["class_weights"],
        DEFAULT_PARAMS["lorar"],
        DEFAULT_PARAMS["lalpha"],
        DEFAULT_PARAMS["ldropout"],
        DEFAULT_PARAMS["head_dim"],
        DEFAULT_PARAMS["head_dropout"],
        DEFAULT_PARAMS["useCLIP"],
        DEFAULT_PARAMS["temperature"],
        DEFAULT_PARAMS["coefficient"]
    )
    
    # Load model weights
    try:
        device_map = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} to {device_map}")
        model.load_state_dict(torch.load(model_path, map_location=device_map))
        model.eval()
        loaded_models[model_key] = model
        
        # Report memory usage after loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(cuda_device) / (1024 ** 3)
            max_allocated = torch.cuda.max_memory_allocated(cuda_device) / (1024 ** 3)
            print(f"GPU Memory: Current: {allocated:.2f} GB, Peak: {max_allocated:.2f} GB")
        
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Function to explicitly unload a model
def unload_model(model_name, device_id=None):
    cuda_device = device_id if device_id is not None else CUDA_DEVICE
    model_key = f"{model_name}_device_{cuda_device}"
    
    if model_key in loaded_models:
        print(f"Explicitly unloading model {model_key}")
        del loaded_models[model_key]
        if model_key in model_usage_order:
            model_usage_order.remove(model_key)
        clear_gpu_memory()
        return True
    return False

# Helper function to process a single sequence
def process_sequence(model, sequence, device_id=None):
    # Use the specified device_id or default to the global CUDA_DEVICE
    cuda_device = device_id if device_id is not None else CUDA_DEVICE
    
    # Tokenize the sequence
    utr5 = " ".join(mytok(sequence.utr5, 1, 1))
    cds = " ".join(mytok(sequence.cds, 3, 3))
    utr3 = " ".join(mytok(sequence.utr3, 1, 1))
    
    # Create a dataset with a single item
    ds = Dataset.from_list([{"5utr": utr5, "cds": cds, "3utr": utr3, "label": 0.0}])
    
    # Encode the dataset
    encoded_ds = ds.map(model.encode_string, batched=True)
    
    # Create trainer for prediction
    device_map = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    training_args = TrainingArguments(
        output_dir="./tmp_output",
        per_device_eval_batch_size=1,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
        dataloader_drop_last=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    
    # Make prediction
    predictions = trainer.predict(encoded_ds).predictions
    
    # Return the prediction
    if isinstance(predictions, np.ndarray) and predictions.size > 0:
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # For multi-class predictions, return probabilities
            return predictions[0].tolist()
        else:
            # For regression, return the scalar value
            return float(predictions[0])
    else:
        return None

# Helper function to process a batch of sequences
def process_batch(model, batch_input, device_id=None):
    # Use the specified device_id or default to the global CUDA_DEVICE
    cuda_device = device_id if device_id is not None else CUDA_DEVICE
    
    sequences = batch_input.sequences
    
    # Prepare batch data
    utr5_list = [" ".join(mytok(seq.utr5, 1, 1)) for seq in sequences]
    cds_list = [" ".join(mytok(seq.cds, 3, 3)) for seq in sequences]
    utr3_list = [" ".join(mytok(seq.utr3, 1, 1)) for seq in sequences]
    
    # Create dataset
    ds = Dataset.from_list([
        {"5utr": utr5, "cds": cds, "3utr": utr3, "label": 0.0} 
        for utr5, cds, utr3 in zip(utr5_list, cds_list, utr3_list)
    ])
    
    # Encode the dataset
    encoded_ds = ds.map(model.encode_string, batched=True)
    
    # Create trainer for prediction
    device_map = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    training_args = TrainingArguments(
        output_dir="./tmp_output",
        per_device_eval_batch_size=8,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
        dataloader_drop_last=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    
    # Make predictions
    predictions = trainer.predict(encoded_ds).predictions
    
    # Process and return predictions
    results = []
    for i, pred in enumerate(predictions):
        if len(pred.shape) > 0 and pred.shape[0] > 1:
            # For multi-class predictions, return probabilities
            results.append({
                "index": i,
                "prediction": pred.tolist(),
                "predicted_class": int(np.argmax(pred))
            })
        else:
            # For regression, return the scalar value
            results.append({
                "index": i,
                "prediction": float(pred)
            })
    
    return results

# API endpoints
@app.get("/")
async def root():
    cuda_info = f"CUDA device: {CUDA_DEVICE}" if torch.cuda.is_available() else "Running on CPU"
    memory_info = ""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(CUDA_DEVICE) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(CUDA_DEVICE) / (1024 ** 3)
        memory_info = f", Memory: Current: {allocated:.2f} GB, Peak: {max_allocated:.2f} GB"
    
    return {
        "message": "mRNA-LM Prediction API", 
        "available_models": list(MODEL_CONFIGS.keys()),
        "cuda_info": cuda_info + memory_info,
        "loaded_models": list(loaded_models.keys())
    }

@app.get("/models")
async def get_models():
    return {
        "available_models": list(MODEL_CONFIGS.keys()),
        "loaded_models": list(loaded_models.keys())
    }

@app.get("/memory")
async def get_memory_info():
    if not torch.cuda.is_available():
        return {"status": "CUDA not available"}
    
    memory_info = {}
    for device in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        memory_info[f"cuda:{device}"] = {
            "allocated_gb": f"{allocated:.2f}",
            "max_allocated_gb": f"{max_allocated:.2f}",
            "reserved_gb": f"{reserved:.2f}",
            "device_name": torch.cuda.get_device_name(device)
        }
    
    return {
        "memory_info": memory_info,
        "loaded_models": list(loaded_models.keys()),
        "model_usage_order": model_usage_order
    }

@app.post("/unload_model/{model_name}")
async def api_unload_model(
    model_name: str,
    device: Optional[int] = Query(None, description="CUDA device index where model is loaded")
):
    cuda_device = device if device is not None else CUDA_DEVICE
    success = unload_model(model_name, cuda_device)
    clear_gpu_memory()  # Force memory cleanup
    
    if success:
        return {"status": f"Model {model_name} on device {cuda_device} unloaded successfully"}
    else:
        return {"status": f"Model {model_name} was not loaded on device {cuda_device}"}

@app.post("/unload_all_models")
async def unload_all_models():
    model_count = len(loaded_models)
    loaded_models.clear()
    model_usage_order.clear()
    clear_gpu_memory()  # Force memory cleanup
    
    return {"status": f"All {model_count} models unloaded successfully"}

@app.post("/predict/{model_name}")
async def predict(
    model_name: str, 
    sequence: SequenceInput,
    device: Optional[int] = Query(None, description="CUDA device index to use for prediction"),
    unload_after: bool = Query(False, description="Whether to unload the model after prediction")
):
    # Select the device to use
    cuda_device = device if device is not None else CUDA_DEVICE
    
    # Load the model
    model = load_model(model_name, cuda_device)
    
    # Make the prediction
    result = process_sequence(model, sequence, cuda_device)
    
    # Unload the model if requested
    if unload_after:
        unload_model(model_name, cuda_device)
    
    return {
        "model": model_name, 
        "prediction": result
    }

@app.post("/predict_batch/{model_name}")
async def predict_batch(
    model_name: str, 
    batch_input: BatchSequenceInput,
    device: Optional[int] = Query(None, description="CUDA device index to use for prediction"),
    unload_after: bool = Query(False, description="Whether to unload the model after prediction")
):
    # Override default parameters if provided
    if batch_input.model_params:
        for key, value in batch_input.model_params.items():
            if key in DEFAULT_PARAMS:
                DEFAULT_PARAMS[key] = value
    
    # Use the device from query parameter if provided
    cuda_device = device if device is not None else CUDA_DEVICE
    
    # Load the model
    model = load_model(model_name, cuda_device)
    
    # Make predictions
    results = process_batch(model, batch_input, cuda_device)
    
    # Unload the model if requested
    if unload_after:
        unload_model(model_name, cuda_device)
    
    return {
        "model": model_name, 
        "predictions": results
    }

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host=args.host, port=args.port, reload=True)
