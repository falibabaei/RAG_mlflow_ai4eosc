import os
from dotenv import load_dotenv
import mlflow
import pandas as pd

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
MODEL_URI = "mlflow-artifacts:/5/models/m-04ae4a57d36d4f1b99ed7fe1d32aad86/artifacts"
INPUT_QUERIES = [
    "how to use LLM on the ai4eosc platform?",
    "how to deploy my model in ai4eosc platform?"
]

print(f"Using API Key: {API_KEY[:4]}...{API_KEY[-4:]}")  # only partial display for security

# Load the MLflow model
model = mlflow.pyfunc.load_model(MODEL_URI)
            
# Prepare input DataFrame
input_data = pd.DataFrame({
    "queries": INPUT_QUERIES,
    "model_name": ["llama-3.3-70b-versatile"] * len(INPUT_QUERIES)
})

def model_wrapper(df):
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

        evaluation_result = mlflow.evaluate(
            model=model_wrapper,
            data=single_input,
            predictions="answer",
            model_type='text'

    
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