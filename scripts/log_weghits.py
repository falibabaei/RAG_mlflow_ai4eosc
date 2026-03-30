import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import mlflow

bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float32
)
repo_id = 'microsoft/Phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(
   repo_id, device_map="auto", quantization_config=bnb_config
)
 
with mlflow.start_run():
    model.save_pretrained("Phi-3-mini-4k-instruct-4bit")

    mlflow.log_artifacts(
        "Phi-3-mini-4k-instruct-4bit",
        artifact_path="model"
    )