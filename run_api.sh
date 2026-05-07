#!/bin/bash

# Default values
DEVICE=1
HOST="0.0.0.0"
PORT=9020
MAX_MODELS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --device|-d)
      DEVICE="$2"
      shift 2
      ;;
    --host|-h)
      HOST="$2"
      shift 2
      ;;
    --port|-p)
      PORT="$2"
      shift 2
      ;;
    --max_models|-m)
      MAX_MODELS="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --device, -d <number>     CUDA device index to use (default: 0)"
      echo "  --host, -h <address>      Host address to bind to (default: 0.0.0.0)"
      echo "  --port, -p <number>       Port to listen on (default: 8000)"
      echo "  --max_models, -m <number> Maximum number of models to keep in memory per device (default: 1)"
      echo "  --help                    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

echo "Starting API with the following settings:"
echo "- CUDA Device: $DEVICE"
echo "- Host: $HOST"
echo "- Port: $PORT"
echo "- Max Models per Device: $MAX_MODELS"


# Check if required packages are installed
echo "Checking required packages..."
python -c "import fastapi, uvicorn, torch, transformers, datasets" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Check if model files exist
echo "Checking model files..."
MODEL_PATHS=(
    "outputs/tr/finetuned_model.pt"
    "outputs/halflife/finetuned_model.pt"
    "outputs/liver/finetuned_model.pt"
    "outputs/hek293t/finetuned_model.pt"
)

MISSING_MODELS=0
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Warning: Model file $MODEL_PATH not found."
        MISSING_MODELS=1
    fi
done

if [ $MISSING_MODELS -eq 1 ]; then
    echo "Some model files are missing. The API will still start, but some endpoints may not work."
    echo "Please make sure the model files are in the correct locations."
    echo -n "Continue anyway? (y/n) "
    read REPLY
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create tmp_output directory if it doesn't exist
mkdir -p tmp_output

# Run the API server
echo "Starting API server..."
python app.py --device $DEVICE --host $HOST --port $PORT --max_models $MAX_MODELS 