# QA Generation Pipeline Test for New Framework

This document describes a pipeline test designed to evaluate and select a new framework for use on High-Performance Computing (HPC) environments. It is divided into two main parts: the preprocessing of documents into a vector store, and the subsequent generation of question-answer pairs from that store. In `requirements.txt` are reported all necessary packages to be install

## 1. Preprocessing Pipeline: From Document to Vector Store

This section provides a detailed guide to `preprocessing_main.py`, a comprehensive, multi-stage data processing pipeline built using the Haystack framework.

### 1.1. Overview

The primary purpose of `preprocessing_main.py` is to convert a raw Markdown document into a structured, queryable vector database and optionally generate a dataset for model fine-tuning or evaluation.

The pipeline performs the following key operations:
1.  **Parses** a Markdown file into structured raw JSON.
2.  **Chunks** the structured content into smaller, semantically meaningful pieces based on a configurable template.
3.  **Generates vector embeddings** for each chunk using a Sentence Transformer model.
4.  **Indexes** the chunks and their embeddings into a **Qdrant** vector store.
5.  **(Optional)** Creates a **prompting dataset** by retrieving relevant and distractor documents for a given set of questions.

### 1.2. Pipeline Steps Explained

The script executes a series of sequential steps, each building on the output of the previous one.

#### Step 1: Markdown to Raw JSON Conversion
* **Component**: `MarkdownToRawJson`
* **Purpose**: To parse the input Markdown file (`--file_md`) and convert its content into a list of structured JSON objects. Each object typically represents a section or a paragraph from the original document.
* **Metadata Injection**: After parsing, the script injects a hardcoded `document_metadata` dictionary into each raw chunk. This is a crucial step because the subsequent chunking template relies on this metadata to build the final chunk structure. **You should customize these placeholder values to match your document's actual metadata.**

#### Step 2: Smart Chunking with `HaystackChunkizer`
* **Component**: `HaystackChunkizer`
* **Purpose**: To break down the raw JSON objects from Step 1 into smaller, more focused chunks suitable for retrieval. This is not a simple fixed-size chunker; it uses an NLP model and a configurable template to create semantically coherent chunks.
* **Configuration**: The chunking behavior is controlled by a YAML file specified via the `--chunking_option` argument. This file defines parameters like minimum/maximum words per chunk, the NLP model to use (e.g., for sentence splitting), and the final output template for each chunk.

#### Step 3: Populating the Qdrant Vector Store
* **Component**: `HaystackQdrantManager`
* **Purpose**: To create vector embeddings for the chunks and store them in a Qdrant collection for efficient similarity search.
* **Process**:
    * The `HaystackQdrantManager` is initialized with the Qdrant location, collection name, and the specified embedding model.
    * It converts each chunk from Step 2 into a Haystack `Document` object.
    * It then uses the specified Sentence Transformer model (`--embedding_model`) to generate a vector embedding for each `Document`.
    * Finally, it writes the documents and their corresponding embeddings to the Qdrant collection. The script checks if the collection already contains documents and will skip indexing if it's not empty to prevent duplication.

#### Step 4 (Optional): Generating a Prompting Dataset
* **Component**: `PromptingDatasetGenerator`
* **Purpose**: To create a structured dataset that can be used for fine-tuning a language model on a RAG (Retrieval-Augmented Generation) task.
* **Trigger**: This step only runs if you provide a path to a JSON file of questions via the `--prompt_questions_file` argument.
* **Process**: For each question in the input file, the generator queries the Qdrant vector store to find:
    * **Relevant Documents**: The top `k` documents that are most similar to the question (`--retriever_top_k`).
    * **Distractor Documents**: Documents that are generally relevant but not the best answer, used to teach the model to discern fine-grained differences (`--distractor_top_k`).
* **Output**: The final dataset is saved as a JSON file, with each entry containing a question along with its retrieved relevant and distractor contexts.

### 1.3. How to Run the Pipeline

The script is executed from the command line. You must provide arguments to specify the input files, output locations, and configuration for each step.

#### Command-Line Arguments
* `--file_md`: Path to the input Markdown file. **(Required)**
* `--output_json`: Path to save the intermediate raw JSON from Step 1. **(Required)**
* `--chunking_option`: Path to the YAML configuration file for the chunker. **(Required)**
* `--chunk_json`: Path to save the final chunked JSON from Step 2. **(Required)**
* `--qdrant_location`: Local file path to store the Qdrant database (e.g., `qdrant_store`). Use this for local persistence.
* `--qdrant_host` / `--qdrant_port`: Alternatively, provide the host and port for a running Qdrant server.
* `--qdrant_collection_name`: The name of the collection to create/use in Qdrant.
* `--embedding_model`: The name or path of the Sentence Transformer model to use for embeddings.
* `--vector_size`: The dimension of the embeddings. This **must** match the output dimension of the chosen `--embedding_model`.
* `--prompt_questions_file`: (Optional) Path to a JSON file containing questions to generate the prompting dataset.
* `--output_prompting_dataset`: (Optional) Path to save the generated prompting dataset.

#### Example Command
This command runs the full pipeline, including the optional prompting dataset generation.

```bash
# Set telemetry to false to disable Haystack's anonymous usage tracking
HAYSTACK_TELEMETRY_ENABLED=False python ./src/preprocessing_main.py \
    --file_md "./data/enciclopedia_idrocarburi/ENCICLOPEDIA DEGLI IDROCARBURI Eni-Treccani 2005 - Volume III.md" \
    --output_json "./data/output/raw_output.json" \
    --chunking_option "./models/chunking/config.yaml" \
    --chunk_json "./data/output/chunked_output.json" \
    --qdrant_location "./data/qdrant_store" \
    --qdrant_collection_name "eni_docs_collection" \
    --embedding_model "./data/gte-multilingual-base" \
    --vector_size 768 \
    --prompt_questions_file "./data/questions.json" \
    --output_prompting_dataset "./data/output/prompting_dataset.json"
```

## 2. Question Generation Pipeline

### 2.1. Overview
The primary goal of this pipeline is to automatically generate high-quality question-answer pairs from a given set of text documents (chunks). This is achieved by leveraging a large language model (LLM) through the `deepeval` library's `Synthesizer`. The process is designed to be run on a High-Performance Computing (HPC) cluster using Singularity containers to ensure a consistent and reproducible environment.

### 2.2. Key Components
#### src - Core Logic

* **qa_gen_single_node.py**: This is the main Python script responsible for the QA generation.
    * It uses the deepeval library's Synthesizer to create QA pairs.
    * It supports custom Hugging Face models.
    * It is built as a command-line tool using click, accepting arguments for the model, configuration files, and output path.
    * It includes a monkey-patch to ensure compatibility with the deepeval library.

#### hpc - HPC Integration

* **question_generation_concretizing.sh**: This is a PBS job submission script.
    * It defines the required HPC resources (GPUs, CPUs, walltime).
    * It sets up all necessary environment variables, including paths for models, data, caches, and outputs.
    * It configures the specific model (Mixtral-8x7B-Instruct-v0.1) and configuration files to be used for a run.
    * It orchestrates the execution of the Python script within a Singularity container.
* **run_qgen.sh**: This is a helper script that is executed inside the Singularity container to launch the Python application. This keeps the main job script cleaner.

#### models - Configuration Files

This directory likely holds YAML/JSON configuration files that control the behavior of the model and the QA generation process.
* **gen_conf.yaml**: Contains parameters for the language model's generation process (e.g., temperature, max_new_tokens).
* **synthesizer_config.yml**: Contains parameters for the deepeval synthesizer (e.g., max_goldens_per_context, evolution strategies).

### 2.3. Core Script: `qa_gen_single_node.py`

This is the main script that orchestrates the QA pair generation. It uses the `click` library to handle command-line arguments, making it easy to configure and run.

#### How It Works

1.  **Initialization**: The script starts by patching a class from the `deepeval` library to prevent potential errors.
2.  **Configuration Loading**: It loads model and synthesizer configurations from YAML files.
3.  **Model Loading**: A custom Hugging Face model is instantiated using the `CustomHuggingFaceModel` class. This includes loading a quantized version of the model to optimize for memory and performance.
4.  **Synthesizer Setup**: The `deepeval.Synthesizer` is configured with the custom model and specific parameters for `Filtration`, `Evolution`, and `Styling` to control the quality and nature of the generated QAs.
5.  **Generation Loop**: The script iterates through a list of predefined text chunks. For each chunk, it calls `synthesizer.generate_goldens_from_contexts()` to produce the QA pairs (referred to as "goldens").
6.  **Saving Results**: All generated QA pairs are collected and saved to a specified output file.

#### Command-Line Arguments

The script accepts the following arguments:

* `BASE_MODEL_ID`: (Required) The name or path to the Hugging Face model to be used for generation.
* `MODEL_CONFIG_FILE`: (Required) Path to the YAML file containing the model's generation parameters.
* `SYNTHESIZER_CONFIG_FILE`: (Required) Path to the YAML file with the `deepeval` synthesizer configurations.
* `OUTPUT_PATH`: (Required) The path where the generated QA pairs will be saved.
* `-m`, `--multiple-gpus`: (Optional Flag) A flag to indicate that multiple GPUs should be used. *Note: The current script version does not have multi-GPU logic implemented.*

#### Example Usage (from command line)

```bash
python qa_gen_single_node.py \
    'Mixtral-8x7B-Instruct-v0.1' \
    'configs/gen_conf.yaml' \
    'configs/synthesizer_config.yml' \
    'output/generated_qa.txt'
```

### 2.4. Execution Script: `question_generation_concretizing.sh`

This shell script is the primary entry point for running the entire pipeline. It is designed for execution on a PBS-managed HPC cluster and uses Singularity to run the Python script in a controlled containerized environment.

#### Key Responsibilities

* **PBS Configuration**: The script begins with `#PBS` directives that specify job requirements for the cluster's scheduler, such as the number of GPUs, CPUs, and walltime.
* **Environment Setup**: It sets up crucial environment variables and paths required for the job, including paths for output, checkpoints, and model storage.
* **Parameter Definition**: It defines which model and configuration files to use for the run.
* **Cache and Container Setup**: It configures paths for various caches (`torch`, `huggingface`) and defines the Singularity container image to be used.
* **Directory Creation**: Ensures that all necessary directories for output and caching exist before the job starts.
* **Singularity Execution**: It uses `singularity exec` to run the `qa_gen_single_node.py` script inside the specified container. It dynamically constructs the `python` command with all the necessary arguments and passes it to the container for execution.

To run the pipeline, you would typically submit this script to the PBS scheduler, for example: `qsub question_generation_concretizing.sh`.

### 2.5. Configuration Files

These YAML files allow you to easily modify the behavior of the model and the generation process without changing the code.

#### `gen_conf.yaml`

This file controls the text generation parameters for the Hugging Face model.

* `max_new_tokens`: The maximum number of tokens to generate for each answer.
* `temperature`: Controls the randomness of the output. Lower values make the model more deterministic.
* `top_k`: Restricts the model's choices to the `k` most likely next tokens.

#### `synthesizer_config.yml`

This file configures the `deepeval.Synthesizer`.

* `filtration_config`: Defines settings for filtering out low-quality generated pairs.
* `evolution_config`: Specifies how to "evolve" or transform the initial questions to create more complex or varied ones. The `CONCRETIZING` evolution, for example, makes questions more specific.
* `styling_config`: Provides a detailed scenario and task description to guide the LLM in generating questions and answers that are relevant to a specific domain (in this case, procedures at the ENI company).

### 2.6. Supporting Python Modules

These modules provide essential helper classes and functions for the main script.

#### `custom_models_hf.py`

* **`CustomHuggingFaceModel`**: A wrapper class that adapts a standard Hugging Face transformer model to be compatible with the `deepeval` library. It handles the model and tokenizer loading, including applying 4-bit quantization (`BitsAndBytesConfig`) to reduce the model's memory footprint. The `generate` method is the core of this class, which takes a prompt and a Pydantic schema to generate structured JSON output.

#### `processing.py`

* **`sanitize_string`**: A simple but important utility function that removes non-printable or invalid characters from a string. This is used to clean the model's output before parsing it as JSON, preventing errors.

#### `resolver.py`

* **`Resolver`**: A utility class that can dynamically import and instantiate Python classes from strings. It reads a configuration dictionary, finds a `__class__` key (e.g., `"deepeval.synthesizer.config.FiltrationConfig"`), and creates an object of that class, passing the other keys as arguments. This mechanism is currently not used in the main script but is available for more complex configuration scenarios.

# LLM Fine-Tuning and Evaluation Pipeline Test for New Framework

This guide describes how to test the fine-tuning pipeline using the provided Python scripts. Each script covers a different stage of the workflow, from prompting dataset creation to model evaluation.
In `requirements.txt` are reported all necessary packages to be install

---
## 1. Prompting Dataset Creation

**Script:** src/feature/make_prompting_dataset.py

**Purpose:** 
Generates a prompting dataset for fine-tuning by formatting raw data into system/user/assistant chat templates and splitting into train/eval/test sets.
To test the code, the prompting dataset was generated from the open source dataset downloaded from this link:
*https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/main/databricks-dolly-15k.jsonl*

**Usage Example:**

```sh
python src/feature/make_prompting_dataset.py <config-file> <data-path> <name-tokenizer> <output-path>
```
- `<config-file>`: YAML file with templates and split configuration (*models/prompting_dataset/config.yml*).
- `<data-path>`: Directory containing raw JSON data (*databricks-dolly-15k.jsonl*).
- `<name-tokenizer>`: Path to the tokenizer model (*meta-llama/Meta-Llama-3-8B-Instruct*)
- `<output-path>`: Directory to save processed datasets.

**Output:**  
Creates `train.jsonl`, `eval.jsonl`, and `test.json` in the output directory.

---

## 2. Model Fine-Tuning

**Script:** scr/fine_tuning/llm_fine_tuning.py

**Purpose:**  
This script fine-tunes a language model (LLM) using a prepared prompting dataset. It supports LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning and can run on multiple GPUs for distributed training. The script is configurable via several YAML files for data, model, trainer, and LoRA settings.

**Usage Example:**
This script is typically launched using the Accelerate library for distributed multi-GPU execution.
```sh
accelerate launch --config_file models/llm_fine_tuning/accelerate_config.yml src/fine_tuning/llm_fine_tuning.py <base-model-id> <data-path> <data-config-file> <model-config-file> <trainer-config-file> <lora-config-file> <output-path> <job-output-path> [--fine-tuned-model-id <path>] [-m]
```

#### Positional Arguments

- `<base-model-id>`  
  Path or HuggingFace identifier for the base LLM to be fine-tuned (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`).

- `<data-path>`  
  Directory containing the processed prompting dataset (only `train.jsonl` and `eval.jsonl`).

- `<data-config-file>`  
  YAML file specifying data loading and preprocessing options (*models/llm_fine_tuning/data_config.yml*).

- `<model-config-file>`  
  YAML file with model-specific configuration (*models/llm_fine_tuning/flashattn_model_noquant_config.yml*).

- `<trainer-config-file>`  
  YAML file with training hyperparameters (*models/llm_fine_tuning/sft_config.yml*).

- `<lora-config-file>`  
  YAML file with LoRA configuration (*models/llm_fine_tuning/peft_config.yml*).

- `<output-path>`  
  Directory where the fine-tuned model and checkpoints will be saved.

- `<job-output-path>`  
  Directory for logging job outputs, metrics, and artifacts.

#### Optional Arguments

- `--fine-tuned-model-id <path>`  
  If specified, resumes training from a previously fine-tuned model checkpoint.

- `-m`, `--multiple_gpus`  
  Enables distributed training across multiple GPUs using the Accelerate library.

### Output

- The LoRA adapter is saved in `<output-path>`.
- Training logs, metrics, and artifacts are saved in `<job-output-path>`.
- Checkpoints are created for resuming or evaluating training.

### Notes

- All configuration files must be valid YAML and match the expected schema.
- Ensure all dependencies in `requirements.txt` are installed.
- For multi-GPU training, hardware and Accelerate configuration must be set up.
- LoRA configuration enables efficient fine-tuning with reduced memory usage.
- This folder *src/fine_tuning_mlde_multi_nodo* contains the scripts we ran on MLDE to perform multi-node fine tuning.

---

## Adapter Model Inference

**Script:** src/models_inference/make_adapter_inference.py

**Purpose:**

This script runs inference using an adapter model (such as a LoRA fine-tuned model) on a test dataset. It supports multi-GPU execution and is highly configurable via YAML files for quantization and generation settings.

**Usage Example:**
This script is typically launched using the Accelerate library for distributed multi-GPU execution.
```sh
accelerate launch --config_file models/inference/accelerate_config_inference.yml src/models_inference/make_adapter_inference.py <model-to-evaluate> <base-model> <quantization-config> <generation-config> <batch-size> [--multiple_gpus] [--runtime_errors] <input_path> <output_path>
```

#### Positional Arguments

- `<model-to-evaluate>`  
  Path to the LoRA adapter checkpoint to be used for inference (the LoRA adapter checkpoint obtained from the fine-tuning step).

- `<base-model>`  
  Path or HuggingFace identifier for the base model architecture (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`).

- `<quantization-config>`  
  YAML file specifying quantization parameters for efficient inference (*models/inference/quatization_config.yml*).

- `<generation-config>`  
  YAML file with generation parameters (*models/inference/generation_config.yml*).

- `<batch-size>`  
  Integer specifying the batch size for inference (default is 4).

- `<input_path>`  
  Path to the input file or directory containing the test dataset (the previously generated `test.json` from src/feature/make_prompting_dataset.py).

- `<output_path>`  
  Path to the output file or directory where predictions will be saved.

#### Optional Flags

- `--multiple_gpus`  
  Enables distributed inference across multiple GPUs using the Accelerate library.

- `--runtime_errors`  
  If set, the script will catch and log runtime errors during inference, allowing the process to continue.

### Output

- Predictions are saved in `<output_path>` as a JSON or JSONL file.
- If `--runtime_errors` is enabled, errors are logged and problematic samples are skipped.
- Supports multi-GPU inference for faster processing on large datasets.

---

## Baseline Model Inference

**Script:** src/models_inference/make_baseline_inference.py

**Purpose:**

This script runs inference using a baseline (non-adapter) language model on a test dataset. It is designed for comparison against fine-tuned or adapter models. The script supports multi-GPU execution and is configurable via YAML files for quantization and generation settings.

**Usage Example:**

This script is typically launched using the Accelerate library for distributed multi-GPU execution.
```sh
accelerate launch --config_file models/inference/accelerate_config_inference.ym src/models_inference/make_baseline_inference.py --base_model <path> --quantization_config <path> --generation_config <path> --batch_size <int> [--multiple_gpus] [--runtime_errors] <input_path> <output_path>
```

#### Command-Line Arguments

- `--base_model <path>`  
  Path or HuggingFace identifier for the base model architecture (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`).

- `<quantization-config>`  
  YAML file specifying quantization parameters for efficient inference (*models/inference/quatization_config.yml*).

- `<generation-config>`  
  YAML file with generation parameters (*models/inference/generation_config.yml*).

- `<batch-size>`  
  Integer specifying the batch size for inference (default is 4).

- `<input_path>`  
  Path to the input file or directory containing the test dataset (the previously generated `test.json` from src/feature/make_prompting_dataset.py).

- `<output_path>`  
  Path to the output file or directory where predictions will be saved.

#### Optional Flags

- `--multiple_gpus`  
  Enables distributed inference across multiple GPUs using the Accelerate library.

- `--runtime_errors`  
  If set, the script will catch and log runtime errors during inference, allowing the process to continue.

### Output

- Predictions are saved in `<output_path>` as a JSON or JSONL file.
- If `--runtime_errors` is enabled, errors are logged and problematic samples are skipped.
- Supports multi-GPU inference for faster processing on large datasets.

---

## Evaluation

**Script:** src/models_evaluation/rag_evaluation.py  
**Dependencies:** Uses model wrapper classes from src/models_evaluation/wrapper.py.

**Purpose:**

This script evaluates the outputs of a language model using RAGAS and DeepEval metrics. It supports multi-GPU distributed evaluation and is configurable via command-line arguments for model paths and input/output directories. The script computes metrics such as response relevancy, answer correctness, and faithfulness, and saves the results in a JSON file for further analysis.

**Usage Example:**

This script is typically launched using the Accelerate library for distributed multi-GPU execution.
```sh
accelerate launch --config_file models/inference/accelerate_config_inference.yml src/models_evaluation/rag_evaluation.py --multiple_gpus --base_model <path> --embedding_model <path> <input_path> <output_path>
```

#### Command-Line Arguments

- `--multiple_gpus`  
  *(Flag, optional)* Enables distributed evaluation across multiple GPUs using the Accelerate library (it needs to be True).

- `--base_model <path>`  
  *(Required)* Path or HuggingFace identifier for the language model to use as evaluator (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`).

- `--embedding_model <path>`  
  *(Required)* Path or HuggingFace identifier for the embedding model used in RAGAS metrics (e.g., `Alibaba-NLP/gte-multilingual-base`).

- `<input_path>`  
  *(Required)* Path to the directory containing the input JSON files with model predictions and references (the output json file from inference script).

- `<output_path>`  
  *(Required)* Path to the directory where evaluation results will be saved as JSON files.

### Output

- Evaluation metrics (response relevancy, answer correctness, faithfulness, etc.) are saved in a JSON file in `<output_path>`.
- If multi-GPU is enabled, results are saved per process (e.g., `rag_evaluation_<timestamp>_<process_index>.json`).
- Logs are generated for each evaluation batch and error handling.

### Notes

- All model paths must be valid and compatible with the wrapper classes in wrapper.py.
- Ensure all dependencies in requirements.txt are installed.
- For multi-GPU evaluation, hardware and Accelerate configuration must be set up.
- The script uses RAGAS and DeepEval metrics via wrapper classes for robust evaluation.

---

## hpc - HPC Integration

The followings are PBS job submission scripts:
* **hpc/job/llm_fine_tuning.sh**
* **hpc/job/adapter_inference.sh**
* **hpc/job/baseline_inference.sh**
* **hpc/job/model_evaluation.sh**

They define the required HPC resources (GPUs, CPUs, walltime), set up all necessary environment variables, including paths for models, data, caches, and outputs, configure the specific model and configuration files to be used for a run, and orchestrate the execution of the Python script within a Singularity container.

* **hpc/scripts/run.sh**: This is a helper script that is executed inside the Singularity container to launch the Python application. This keeps the main job script cleaner.

---