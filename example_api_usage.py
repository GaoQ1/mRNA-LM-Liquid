import requests
import json
import argparse
import time

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test mRNA-LM Prediction API')
parser.add_argument('--host', type=str, default='localhost', help='API host address')
parser.add_argument('--port', type=int, default=8000, help='API port')
parser.add_argument('--device', type=int, default=None, help='CUDA device to use for predictions')
parser.add_argument('--test-memory', action='store_true', help='Test memory management features')
args = parser.parse_args()

# API base URL
BASE_URL = f"http://{args.host}:{args.port}"

def test_api_endpoints():
    """Test all API endpoints"""
    
    # Test root endpoint
    response = requests.get(f"{BASE_URL}/")
    print("Root endpoint response:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # Test models endpoint
    response = requests.get(f"{BASE_URL}/models")
    print("Models endpoint response:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # Test memory endpoint
    response = requests.get(f"{BASE_URL}/memory")
    print("Memory endpoint response:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # Test single prediction endpoint
    model_name = "tr"  # Change to any available model
    sequence_data = {
        "utr5": "ACGUACGUACGU",
        "cds": "AUGCAUGCAUGC",
        "utr3": "ACGUACGUACGU"
    }
    
    # Add device parameter to URL if specified
    url = f"{BASE_URL}/predict/{model_name}"
    params = {}
    if args.device is not None:
        params["device"] = args.device
    
    response = requests.post(
        url,
        json=sequence_data,
        params=params
    )
    
    print(f"Single prediction ({model_name}) response:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # Test batch prediction endpoint
    model_name = "liver"  # Change to any available model
    batch_data = {
        "sequences": [
            {
                "utr5": "ACGUACGUACGU",
                "cds": "AUGCAUGCAUGC",
                "utr3": "ACGUACGUACGU"
            },
            {
                "utr5": "UACGUACGUACG",
                "cds": "CAUGCAUGCAUG",
                "utr3": "UACGUACGUACG"
            }
        ],
        "model_params": {
            "head_dropout": 0.3  # Override default parameter
        }
    }
    
    # Add device parameter to URL if specified
    url = f"{BASE_URL}/predict_batch/{model_name}"
    params = {}
    if args.device is not None:
        params["device"] = args.device
    
    response = requests.post(
        url,
        json=batch_data,
        params=params
    )
    
    print(f"Batch prediction ({model_name}) response:")
    print(json.dumps(response.json(), indent=2))

def test_memory_management():
    """Test memory management features"""
    print("\n===== Testing Memory Management Features =====\n")
    
    # 1. Check initial memory state
    response = requests.get(f"{BASE_URL}/memory")
    print("Initial memory state:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 2. Load first model with auto-unload disabled
    model_name1 = "tr"
    sequence_data = {
        "utr5": "ACGUACGUACGU",
        "cds": "AUGCAUGCAUGC",
        "utr3": "ACGUACGUACGU"
    }
    
    print(f"Loading and using {model_name1} model...")
    params = {"unload_after": False}
    if args.device is not None:
        params["device"] = args.device
    
    response = requests.post(
        f"{BASE_URL}/predict/{model_name1}",
        json=sequence_data,
        params=params
    )
    print(f"Prediction with {model_name1} completed:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 3. Check memory after first model
    response = requests.get(f"{BASE_URL}/memory")
    print("Memory state after loading first model:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 4. Load second model with auto-unload disabled (should trigger LRU eviction if max_models=1)
    model_name2 = "liver"
    print(f"Loading and using {model_name2} model...")
    response = requests.post(
        f"{BASE_URL}/predict/{model_name2}",
        json=sequence_data,
        params=params
    )
    print(f"Prediction with {model_name2} completed:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 5. Check memory after second model
    response = requests.get(f"{BASE_URL}/memory")
    print("Memory state after loading second model (may have triggered LRU unloading):")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 6. Try a third model with auto-unload enabled
    model_name3 = "halflife"
    print(f"Loading and using {model_name3} model with auto-unload...")
    params["unload_after"] = True
    response = requests.post(
        f"{BASE_URL}/predict/{model_name3}",
        json=sequence_data,
        params=params
    )
    print(f"Prediction with {model_name3} completed and model unloaded:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 7. Check memory after auto-unload
    response = requests.get(f"{BASE_URL}/memory")
    print("Memory state after auto-unloading third model:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 8. Explicitly unload a model
    print("Explicitly unloading model...")
    unload_params = {}
    if args.device is not None:
        unload_params["device"] = args.device
    response = requests.post(
        f"{BASE_URL}/unload_model/{model_name2}",
        params=unload_params
    )
    print("Unload response:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 9. Check memory after explicit unload
    response = requests.get(f"{BASE_URL}/memory")
    print("Memory state after explicit unload:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 10. Unload all models
    print("Unloading all models...")
    response = requests.post(f"{BASE_URL}/unload_all_models")
    print("Unload all response:")
    print(json.dumps(response.json(), indent=2))
    print("\n" + "-"*50 + "\n")
    
    # 11. Final memory check
    response = requests.get(f"{BASE_URL}/memory")
    print("Final memory state:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print(f"Testing API at {BASE_URL}" + (f" using CUDA device {args.device}" if args.device is not None else ""))
    
    if args.test_memory:
        test_memory_management()
    else:
        test_api_endpoints() 