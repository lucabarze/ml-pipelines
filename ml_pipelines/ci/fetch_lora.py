import os
import shutil
import argparse
from mlflow.artifacts import download_artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlflow-uri", required=True, help="URI del server MLflow (es: http://mlflow.apps.eni.lajoie.de)")
    ap.add_argument("--model", required=True, help="Nome del modello registrato in MLflow")
    ap.add_argument("--alias", default="champion", help="Alias del modello (es: champion, production)")
    ap.add_argument("--out", required=True, help="Cartella di destinazione dove copiare l'adapter")
    args = ap.parse_args()

    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri
    uri = f"models:/{args.model}@{args.alias}"

    print(f"[MLflow] downloading {uri} ...")
    tmp = download_artifacts(uri)

    src = os.path.join(tmp, "artifacts", "final-adapter")
    if not os.path.isdir(src):
        raise RuntimeError(f"Adapter path non trovato: {src}")

    dst = os.path.join(args.out, "final-adapter")
    os.makedirs(dst, exist_ok=True)

    for fn in ["adapter_config.json", "adapter_model.safetensors"]:
        fsrc = os.path.join(src, fn)
        if not os.path.isfile(fsrc):
            raise RuntimeError(f"File mancante nell'adapter: {fsrc}")
        shutil.copy(fsrc, dst)

    print(f"[DONE] adapter ready at: {dst}")

if __name__ == "__main__":
    main()
