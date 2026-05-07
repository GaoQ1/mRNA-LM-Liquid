import os
import argparse
import torch
import numpy as np
import pandas as pd

from transformers import Trainer, TrainingArguments
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score

from FullModel import FullModel
from dataload import *

os.environ["TOKENIZERS_PARALLELISM"] = "true"

######### Arguments Processing
parser = argparse.ArgumentParser(description='FullModel Prediction')

parser.add_argument('--task',   '-t', required=True, type=str, default="", help='task')
parser.add_argument('--model_path', '-m', required=True, type=str, default="./outputs/tr/finetuned_model.pt", help='path to the trained model')
parser.add_argument('--output', '-o', required=True, type=str, default="", help='output dir') 

parser.add_argument('--lorar',    type=int, default=32, help='Lora rank')
parser.add_argument('--lalpha',   type=int, default=32, help='Lora alpha')
parser.add_argument('--ldropout', type=int, default=0.5, help='Lora dropout')

parser.add_argument('--head_dim', type=int, default=768, help='production head dimension')
parser.add_argument('--head_dropout', type=float, default=0.5, help='production head dropout')

parser.add_argument('--device', '-d', type=int, default=0, help='device')
parser.add_argument('--batch',  '-b', type=int, default=64, help='batch size')

parser.add_argument('--useCLIP',      '-clip', type=bool, default=False, help='use CLIP')
parser.add_argument('--temperature',  '-temp', type=float, default=0.07, help='temperature')
parser.add_argument('--coefficient',  '-coeff', type=float, default=0.2, help='coefficient')

args = parser.parse_args()

########### GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.device

num_labels = 1
class_weights = []

########### loading pretrained model and downstream task model
model = FullModel(
    num_labels, 
    class_weights, 
    args.lorar, 
    args.lalpha, 
    args.ldropout, 
    args.head_dim, 
    args.head_dropout,
    args.useCLIP, 
    args.temperature, 
    args.coefficient
)

# Load the trained model
print(f"Loading model from {args.model_path}")
model.load_state_dict(torch.load(args.model_path))
model.eval()

########### loading dataset and dataloader
csv_path = None
if args.task == "604":
    ds_test = build_604_dataset()
    csv_path = "data/实验数据/604.csv"
elif args.task == "auro_rsv":
    ds_test = build_auro_rsv_dataset()
    csv_path = "data/实验数据/auro_rsv.csv"
elif args.task == "zheda_homo":
    ds_test = build_zheda_homo_dataset()
    csv_path = "data/实验数据/zheda_homo.csv"
elif args.task == "zheda_mouse":
    ds_test = build_zheda_mouse_dataset()
    csv_path = "data/实验数据/zheda_mouse.csv"

test_loader = ds_test.map(model.encode_string, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if num_labels == 1:
        logits = logits.flatten()
        labels = labels.flatten()

        try:
            pearson_corr = pearsonr(logits, labels)[0].item()
            spearman_corr = spearmanr(logits, labels)[0].item()
            return {
                "pearson": pearson_corr,
                "spearmanr": spearman_corr,
            }
        except:
            return {"pearson":0.0, "spearmanr":0.0}
    else:
        predictions = np.argmax(logits, axis=-1)
        logits = softmax(logits, axis=1)
        
        f1 = f1_score(predictions, labels, average="macro")
        auroc = roc_auc_score(labels, logits, average="macro", multi_class='ovr')
        
        return {"f1": f1, "auroc": auroc}

######### Trainer Settings for Prediction
training_args = TrainingArguments(
    output_dir=args.output,
    per_device_eval_batch_size=8,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics
)

# Prediction on test set
print("Making predictions on test set...")
predictions, labels, metrics = trainer.predict(test_loader)
print("Test metrics:", metrics)

# Save predictions to CSV
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Save predictions as numpy arrays (for compatibility)
np.save(os.path.join(args.output, "predictions.npy"), predictions)
np.save(os.path.join(args.output, "labels.npy"), labels)

# Read original CSV and add predictions
if csv_path:
    df = pd.read_csv(csv_path)
    df['te'] = predictions.flatten()  # Add predictions as new column 'te'
    
    # Create result directory if it doesn't exist
    result_dir = os.path.join(args.output, "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Save to new CSV file
    output_csv = os.path.join(result_dir, f"{args.task}_with_predictions.csv")
    df.to_csv(output_csv, index=False)
    print(f"Results saved to CSV file: {output_csv}")

