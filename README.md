# mRNA-LM Prediction API

This is a FastAPI-based web service that provides prediction interfaces for various mRNA models trained using the mRNA-LM framework.

## Available Models

The API provides access to the following trained models:

- **tr**: Translation rate prediction model
- **halflife**: mRNA half-life prediction model
- **liver**: Liver transcript expression prediction model
- **5class**: 5-class protein expression classification model

## API Endpoints

### GET `/`

Returns a welcome message and a list of available models.

### GET `/models`

Returns a list of available models.

### POST `/predict/{model_name}`

Makes a prediction for a single sequence using the specified model.

**Path Parameters:**
- `model_name`: Name of the model to use (tr, halflife, liver, 5class)

**Request Body:**
```json
{
  "utr5": "ACGUACGUACGU...",
  "cds": "AUGCAUGCAUGC...",
  "utr3": "ACGUACGUACGU..."
}
```

**Response:**
```json
{
  "model": "model_name",
  "prediction": 0.75  // or array of class probabilities for 5class model
}
```

### POST `/predict_batch/{model_name}`

Makes predictions for multiple sequences using the specified model.

**Path Parameters:**
- `model_name`: Name of the model to use (tr, halflife, liver, 5class)

**Request Body:**
```json
{
  "sequences": [
    {
      "utr5": "ACGUACGUACGU...",
      "cds": "AUGCAUGCAUGC...",
      "utr3": "ACGUACGUACGU..."
    },
    {
      "utr5": "ACGUACGUACGU...",
      "cds": "AUGCAUGCAUGC...",
      "utr3": "ACGUACGUACGU..."
    }
  ],
  "model_params": {
    "lorar": 32,
    "lalpha": 32,
    "ldropout": 0.5,
    "head_dim": 768,
    "head_dropout": 0.5,
    "useCLIP": false,
    "temperature": 0.07,
    "coefficient": 0.2,
    "device": 0
  }
}
```

The `model_params` field is optional and can be used to override the default model parameters.

**Response:**
```json
{
  "model": "model_name",
  "predictions": [
    {
      "index": 0,
      "prediction": 0.75  // or array of class probabilities for 5class model
    },
    {
      "index": 1,
      "prediction": 0.82
    }
  ]
}
```

## Running the API

To run the API, use one of the following methods:

### Using Python directly:

```bash
python app.py --device 1 --host 0.0.0.0 --port 8000 --max_models 1
```

Command line options:
- `--device`: CUDA device index to use (default: 0)
- `--host`: Host address to bind to (default: 0.0.0.0)
- `--port`: Port to listen on (default: 8000)
- `--max_models`: Maximum number of models to keep in memory per device (default: 1)

### Using the provided shell script:

```bash
./run_api.sh --device 1 --host 0.0.0.0 --port 8000 --max_models 1
```

This will start the API server on `http://0.0.0.0:8000` using CUDA device 1 and keeping a maximum of 1 model in memory.

You can also access the interactive API documentation at `http://localhost:8000/docs`.

### Specifying CUDA device in API calls

You can specify which CUDA device to use for a specific API call by adding a query parameter:

```
http://localhost:8000/predict/tr?device=1
```

This will run the prediction on CUDA device 1, regardless of the default device specified when starting the server.

### Memory Management

The API includes memory management features to help prevent GPU out-of-memory errors:

1. **Automatic Model Unloading**: When the number of models loaded on a GPU exceeds the `max_models` setting, the least recently used model will be automatically unloaded.

2. **Manual Model Unloading**: You can explicitly unload models using the `/unload_model/{model_name}` endpoint.

3. **Auto-unload after prediction**: Add `unload_after=true` to prediction endpoints to automatically unload the model after prediction:
   ```
   http://localhost:8000/predict/tr?unload_after=true
   ```

4. **Memory Information**: Get current GPU memory usage information with the `/memory` endpoint.

5. **Unload All Models**: Clear all loaded models with the `/unload_all_models` endpoint.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- PyTorch
- Transformers
- Datasets
- NumPy
- Pandas
- PEFT (Parameter-Efficient Fine-Tuning)

## Model Paths

The API expects the trained models to be located at:

- `models/tr/finetuned_model.pt`
- `models/halflife/finetuned_model.pt`
- `models/liver/finetuned_model.pt`
- `models/5class/finetuned_model.pt`

Make sure these files exist before running the API.

## Pre-trained models
The pre-trained models are available for download here: [CodonBERT](https://cdn.prod.accelerator.sanofi/llm/CodonBERT.zip), [5UTRBERT](https://cdn.prod.accelerator.sanofi/llm/mrna_5utr_model.zip), and [3UTRBERT](https://cdn.prod.accelerator.sanofi/llm/mrna_3utr_model.zip). 

## Finetune the mRNA-LM model 
```python finetune_all.py --task halflife ```
### command-line arguments:
- --task, -t     (str,   default=""):  **Required.** Task. Specify the task to be performed by the model.
- --output, -o   (str,   default=""):  **Required.** Output folder. Specify the output folder to save models. 
- --batch, -b    (int,   default=64):    Batch size. Specify the size of each batch.
- --lr           (float, default=1e-5):  Learning rate. Modify this to change the learning rate. The comment mentions 2e-5 as an alternative.
- --device, -d   (int,   default=0):     GPU device. Specify the GPU ID to run the model.
- --lorar        (int,   default=32):    Lora rank. Adjust this to control the rank of Lora.
- --lalpha       (int,   default=32):    Lora alpha. Adjust this to control the alpha parameter of Lora.
- --ldropout     (float, default=0.5):   Lora dropout. Adjust this to control the dropout rate of Lora.
- --head_dim     (int,   default=768):   Production head dimension. Modify this to change the dimension of the production head.
- --head_dropout (float, default=0.5):   Production head dropout. Modify this to change the dropout rate of the production head.
- --useCLIP      (bool,  default=False): Whether to use CLIP. Adjust this to turn the CLIP loss on or off.
- --temperature  (float, default=0.07):  Temperature parameter in CLIP loss. Modify this to change the temperature for the CLIP loss.
- --coefficient  (float, default=0.2):   Adjustment coefficient for CLIP loss. Modify this to change the coefficient for the CLIP loss.
