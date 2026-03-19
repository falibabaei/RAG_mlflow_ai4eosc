# RAG Pipeline with MLflow for AI4EOSC Documentation

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **AI4EOSC project documentation** as the knowledge base. It enables context-aware question answering by combining document retrieval (FAISS + embeddings) with large language models.

---

##  Features

-  **Semantic Retrieval** using Sentence Transformers + FAISS  
-  **LLM-based Answer Generation** (via Groq / OpenAI-compatible API)  
-  **MLflow Integration** for:
  - Model logging
  - Evaluation
  - Prompt tracking
-  **Context-aware QA** over AI4EOSC documentation  
-  **Prompt versioning** with MLflow GenAI  

---

## Project Structure
RAG_mlflow_ai4eosc/
│
├── data/                # Stores input datasets and documents used for retrieval (knowledge base for RAG)
├── models/              # Contains saved or exported models, including MLflow artifacts and trained models
├── scripts/             # Core scripts for data preparation and MLflow model lifecycle management
│   ├── load_model_mlflow.py    # Loads a trained model from MLflow for inference or evaluation
│   ├── log_model_mlflow.py     # Logs models, parameters, and artifacts to MLflow tracking server
│   ├── prepare_data.py         # Preprocesses and prepares raw data for the RAG pipeline (cleaning, formatting, embedding input)
│
├── requirements.txt     # Lists all Python dependencies required to run the project
└── README.md            # Project documentation with setup instructions and usage details

##  Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
```
2.  Install dependencies
```bash
pip install requirment.text

```
3. Set environment variables
Create a `.env` file in the root directory:
```bash

export MLFLOW_TRACKING_URI=your_tracking_uri
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_password
GROQ_API_KEY=your_api_key_here
BASE_URL=https://api.groq.com/openai/v1
```

4. Prepare Data
This project uses the `.pdf` version of the AI4EOSC documentation.

```bash
python prepaer_data.py

``` 
5. Log RAG Model with MLflow
``` bash
python rag_log_model_mlflow.py
``` 
This:

- Builds FAISS index
- Wraps retrieval + generation into a pyfunc model
- Logs model and artifacts to MLflow

5. Run Inference
``` bash
python use_rag_model.py
``` 
Example query:
``` bash
"how to add my model in ai4eosc platform?"
``` 