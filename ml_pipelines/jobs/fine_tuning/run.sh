python src/fine_tuning_amd.py \
	--model_name "microsoft/Phi-3-mini-4k-instruct"\
       	--mlflow_uri http://mlflow.apps.eni.lajoie.de/\
	--data_uri "latest:Datasets"\
	--lr 0.0005\
	--epochs 1\
	--data_path ""
