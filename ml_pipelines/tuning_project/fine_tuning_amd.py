import os
import torch
import mlflow
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

OUTPUT_DIR = "/app/results"

def main(args):
    print(f"--- Avvio del job di fine-tuning su GPU AMD (ROCm) ---")
    print(f"Modello di partenza: {args.model_name}")
    print(f"Server MLflow: {args.mlflow_uri}")

    # --- Setup di MLflow usando il parametro ---
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("Fine-Tuning Chatbot AMD")

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("gpu_type", "AMD/ROCm")
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("data_path", args.data_path)

        # --- Caricamento Dataset ---
        dataset = load_dataset("json", data_files=args.data_path, split="train")

        def format_prompt(example):
            return f"<s>[INST] {example['domanda']} [/INST] {example['risposta']}</s>"

        # --- Caricamento Modello e Tokenizer usando il parametro ---
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # ... (il resto della logica di training rimane identico) ...
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
        
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            logging_steps=10,
            bf16=True,
        )
        
        trainer = SFTTrainer(model=model, train_dataset=dataset, peft_config=lora_config, formatting_func=format_prompt, max_seq_length=512, tokenizer=tokenizer, args=training_args)

        trainer.train()
        
        trainer.save_model(os.path.join(OUTPUT_DIR, "final-adapter"))
        mlflow.log_artifact(local_path=os.path.join(OUTPUT_DIR, "final-adapter"), artifact_path="lora-adapter")
        
        print("--- Job di fine-tuning completato con successo! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Aggiungiamo i nuovi argomenti
    parser.add_argument("--model_name", type=str, required=True, help="Nome del modello base da Hugging Face")
    parser.add_argument("--mlflow_uri", type=str, required=True, help="URI del server di tracking MLflow")
    
    # Argomenti precedenti
    parser.add_argument("--data_path", type=str, required=True, help="Percorso del file dataset.jsonl")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate per il training")
    parser.add_argument("--epochs", type=int, required=True, help="Numero di epoche di training")
    
    args = parser.parse_args()
    main(args)
