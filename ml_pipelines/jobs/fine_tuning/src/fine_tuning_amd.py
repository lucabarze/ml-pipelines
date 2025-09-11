import os
import torch
import mlflow
from mlflow import start_span
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
from pathlib import Path
import tempfile
from mlflow.artifacts import download_artifacts
import pandas as pd 
import mlflow.pyfunc as pyfunc

class LoraArtifactOnly(pyfunc.PythonModel):
    def load_context(self, context): 
        pass
    def predict(self, context, model_input):
        raise RuntimeError("Solo contenitore per LoRA; usare vLLM per il serving.")

def _resolve_dataset_path(args) -> str:
    """
    Ritorna un path locale pronto per load_dataset.
    Regole:
    - se data_uri è 'latest:<EXP>', prende la run con tag latest=true su <EXP>
    - se data_uri inizia con runs:/, mlflow-artifacts:/ o s3://, scarica/risolve
    - altrimenti usa data_path locale
    """
    if args.data_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        if args.data_uri.startswith("latest:"):
            exp_name = args.data_uri.split("latest:", 1)[1]
            exp = mlflow.get_experiment_by_name(exp_name)
            assert exp, f"Esperimento '{exp_name}' inesistente"
            df = mlflow.search_runs(
                [exp.experiment_id],
                filter_string="tags.latest = 'true'",
                order_by=["start_time DESC"],
                max_results=1
            )
            assert len(df), f"Nessuna run con latest=true in '{exp_name}'"
            run_id = df.iloc[0]["run_id"]
            uri = f"runs:/{run_id}/datasets/dataset.jsonl"
            return download_artifacts(uri, dst_path=tempfile.mkdtemp(prefix="ds_"))
        if args.data_uri.startswith(("runs:/", "mlflow-artifacts:/", "s3://", "http://", "https://")):
            return download_artifacts(args.data_uri, dst_path=tempfile.mkdtemp(prefix="ds_"))
        # fallback: trattalo come file locale
        return args.data_uri
    # nessuna data_uri: usa data_path locale (comportamento attuale)
    return args.data_path



OUTPUT_DIR = "/lvol/barzel/results"
out_dir = Path(OUTPUT_DIR) / "final-adapter"

def main(args):
    print(f"--- Avvio del job di fine-tuning su GPU AMD (ROCm) ---")
    print(f"Modello di partenza: {args.model_name}")
    print(f"Server MLflow: {args.mlflow_uri}")

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("Fine-Tuning Chatbot AMD")

    with mlflow.start_run() as run:
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("gpu_type", "AMD/ROCm")
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("data_path", args.data_path)

        import platform, transformers, peft, trl
        mlflow.set_tags({
            "dataset_uri": args.data_uri or args.data_path,
            "run_purpose": "ft-lora-demo",
            "lora_target_modules": "qkv_proj,o_proj",
            "device": "rocm" if torch.version.hip else ("cuda" if torch.cuda.is_available() else "cpu"),
            "host": platform.node(),
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "trl_version": trl.__version__,
            "torch_version": torch.__version__,
        })

        with start_span("prepare_dataset") as s:
            ds_local_path = _resolve_dataset_path(args)
            mlflow.log_param("dataset_source", getattr(args, "data_uri", None) or args.data_path)
            dataset = load_dataset("json", data_files=ds_local_path, split="train").select(range(1000))
            s.set_attributes({"num_rows": len(dataset)})


        def format_prompt(example):
            q = example.get("question")
            a = example.get("answer")
        
            if isinstance(q, list):
                return [f"<s>[INST] {qq} [/INST] {aa}</s>" for qq, aa in zip(q, a)]
        
            return [f"<s>[INST] {q} [/INST] {a}</s>"]

        with start_span("load_model") as s:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            s.set_attributes({"base_model": args.model_name})


        with start_span("train") as s:
            lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["qkv_proj", "o_proj"])
        
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
        
        trainer.save_model(out_dir.as_posix())
        pyfunc.log_model(
          artifact_path="lora_model",
          python_model=LoraArtifactOnly(),
          artifacts={
              "adapter": out_dir.as_posix(),
              "tokenizer": out_dir.as_posix() 
          }
        )
        try:
            mlflow.log_artifact(ds_local_path, artifact_path="input")
        except Exception:
            pass
        
        print("--- Job di fine-tuning completato con successo! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Nome del modello base da Hugging Face")
    parser.add_argument("--mlflow_uri", type=str, required=True, help="URI del server di tracking MLflow")
    parser.add_argument("--data_path", type=str, required=True, help="Percorso del file dataset.jsonl")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate per il training")
    parser.add_argument("--epochs", type=int, required=True, help="Numero di epoche di training")
    parser.add_argument("--data_uri", type=str, required=False,
                    help="URI dell'artifact (runs:/..., mlflow-artifacts:/..., s3://..., http...) oppure 'latest:Datasets'")
    
    args = parser.parse_args()
    main(args)
