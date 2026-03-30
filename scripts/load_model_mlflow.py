import os
from dotenv import load_dotenv
import mlflow
import pandas as pd
from mlflow.genai.scorers import Correctness
# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_URI = "mlflow-artifacts:/17/models/m-b0c1e6e13a3e4d2a9cc68a153fdec845/artifacts"
INPUT_QUERIES = [
    "how to use LLM on the ai4eosc platform?",
    
    "how to deploy my model in ai4eosc platform?"
]
eval_dataset = [
    {
        "inputs": {"query": "how to use LLM on the ai4eosc platform?"},
        "expectations": {
            "expected_response": "The response should explain how to use LLM on the ai4eosc platform from the documentation."
        },
    },
]
print(f"Using API Key: {API_KEY[:4]}...{API_KEY[-4:]}")  # only partial display for security

# Load the MLflow model
model = mlflow.pyfunc.load_model(MODEL_URI)
            
# Prepare input DataFrame
input_data = pd.DataFrame({
    "queries": INPUT_QUERIES,
    "model_name": ["llama-3.3-70b-versatile"] * len(INPUT_QUERIES)
})

def model_wrapper(query):
    df = pd.DataFrame({
        "queries": [query],
        "model_name": ["llama-3.3-70b-versatile"]
    })

    return model.predict(
        df,
        params={
            "api_key": API_KEY,
            "base_url": BASE_URL
        }
    )
# Start MLflow run for inference logging
with mlflow.start_run(run_name="rag_inference"):
    
    logged_prompts = []
    logged_outputs = []

    for i, query in enumerate(input_data["queries"]):
        
        # Prepare single-query DataFrame
        single_input = pd.DataFrame({
            "queries": [query],
            "model_name": [input_data["model_name"].iloc[i]]
        })
        
        # Run the model
        prediction_df = model.predict(
            single_input,
            params={"api_key": API_KEY, "base_url": BASE_URL}
        )

        evaluation_result =mlflow.genai.evaluate(
            data=eval_dataset,
             scorers=[Correctness(model='groq:/llama-3.3-70b-versatile')],
            predict_fn= model_wrapper
        )
        # Rebuild the prompt for logging (or modify predict() to return it)
        #prompt_text = f"Question: {query}"

        # Log prompt and output to MLflow
        #prompt_file = f"prompt_{i}.txt"
       # output_file = f"output_{i}.txt"
       # mlflow.log_text(prompt_text, prompt_file)
        #mlflow.log_text(prediction_df["answer"].iloc[0], output_file)

     #   logged_prompts.append(prompt_file)
      #  logged_outputs.append(output_file)

    # Optional: log all in a single JSON for easy retrieval
 #   mlflow.log_dict(
    #    [{"query": q, "prompt_file": p, "output_file": o}
     #    for q, p, o in zip(INPUT_QUERIES, logged_prompts, logged_outputs)],
    #    "inference_log.json"
  #  )

print("Predictions:")
print(prediction_df)