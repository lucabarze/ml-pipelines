import os
import mlflow
import argparse
from mlflow.tracking import MlflowClient

def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(args.experiment)
    if exp:
        old = mlflow.search_runs([exp.experiment_id], "tags.latest = 'true'", max_results=1)
        if len(old):
            prev_run_id = old.iloc[0]["run_id"]
            client.set_tag(prev_run_id, "latest", "false")

    if not os.path.exists(args.local_file):
        raise FileNotFoundError(f"File non trovato: {args.local_file}")

    with mlflow.start_run(run_name="dataset_upload") as run:
        mlflow.log_artifact(args.local_file, artifact_path="datasets")
        mlflow.set_tag("latest", "true")

        artifact_subpath = "datasets/dataset.jsonl"
        print("OK. run_id:", run.info.run_id)
        print("Artifact URI (specifico):", f"runs:/{run.info.run_id}/{artifact_subpath}")
        print("Artifact URI (stabile):  ", f"latest:{args.experiment}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mlflow_uri", default="http://mlflow.apps.eni.lajoie.de")
    p.add_argument("--experiment", default="Datasets")
    p.add_argument("--local_file", default="dataset.jsonl")
    args = p.parse_args()
    main(args)
