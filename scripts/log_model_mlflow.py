import mlflow
from sentence_transformers import SentenceTransformer
import re
import pickle
import faiss
import os
from dotenv import load_dotenv
import openai
import pandas as pd
from mlflow.models import infer_signature


load_dotenv()
 
class PyfuncWithRetrieval(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow model for text generation with retrieval functionality.

    Extends the mlflow.pyfunc.PythonModel class and utilizes a pre-trained
    OpenAI transformer model for text generation based on an external vector
    database for retrieval of relevant context.
    """

    def __init__(self, docs, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the PyfuncWithRetrieval model.

        Args:
        """
        super().__init__()
        
        # Initialize the vector database client here (e.g., Pinecone, Weaviate, etc.)
        self.docs = None
        self.embeddings = None#SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.index = None
     
      
    def load_context(self, context):

        load_dotenv()
        
        # Load vector DB
        with open(context.artifacts["vector_db"], "rb") as f:
            self.vector_db = pickle.load(f)
        self.docs = self.vector_db
        self.embeddings = SentenceTransformer(self.embedding_model_name)
        self.index, self.embeddings_matrix = self._build_index(self.docs)
 
        

    def _build_index(self, docs):
        text_chunks = [doc["content"] for doc in docs]
        embs= self.embeddings.encode(text_chunks,batch_size=32, show_progress_bar=True, convert_to_numpy=True).astype("float32")
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs)
        return index ,embs

    def retrieve_relevant_chunks(self, query: str, top_k=5):
        """Retrieves the most relevant chunks of text from the vector database based on the input query.
        
        Args:
            query (str): The input query for retrieval.
            top_k (int): The number of top relevant chunks to retrieve. Defaults to 5.  
        Returns:
            List[str]: A list of the most relevant chunks of text based on the input query.
        """     
        query_embedding = self.embeddings.encode([query], convert_to_numpy=True).astype("float32")
        distance, indices = self.index.search(query_embedding, top_k)
        relevant_chunks = [self.docs[i] for i in indices[0]]
        return relevant_chunks[:top_k]

    def build_context(self, chunks):
        return "\n\n---\n\n".join(
            f"(Page {c['metadata'].get('page_number', '?')}) {c['content']}"
            for c in chunks
        )
    
    def predict(self, context, model_input, params=None) -> str:       
        """Generates text based on the input context and retrieved relevant information.        
        Args:
            context (str): The input context for text generation.
            model_input (pd.DataFrame): DataFrame with "query" column containing the queries to be answered.
        Returns:
            str: The generated text based on the input context and retrieved information.
        """
        outputs = []
        model_name = model_input["model_name"].iloc[0]
        if isinstance(model_input, pd.DataFrame):
            if "queries" in model_input.columns:
                queries = model_input["queries"].tolist()
            elif "query" in model_input.columns:
                queries = model_input["query"].tolist()
            else:
                raise ValueError('Column "query" or "queries" required')
        else:
            queries = [model_input]
        self.client = openai.OpenAI(
            api_key=params.get("api_key") if params and "api_key" in params else None,
            base_url=params.get("base_url") if params and "base_url" in params else None
        )
        for query in queries:
            retrieved_chunks = self.retrieve_relevant_chunks(query)
            for r in retrieved_chunks:
                print(r["content"])
            context = self.build_context(retrieved_chunks)
            prompt = f"""You are an assistant answering questions using ONLY the context below.

            Context:
            {context}

            Question: {query}

            Answer using the context as much as possible. If the answer is not explicitly in the context, 
            make an informed response and indicate uncertainty if needed, but try to provide helpful information.
            """

            completion= self.client.chat.completions.create(
                model=model_name,
                messages=[{'role': "system", "content": "You are a helpful assistant."},
                {'role': "user", "content": prompt}],
                temperature=0.5,
                max_tokens=700,)
            outputs.append(completion.choices[0].message.content)
        return pd.DataFrame({"query": queries, "answer": outputs})

if __name__ == "__main__":
    # Example usage
    input_example = pd.DataFrame({
    "queries": ["What is the ai4eosc project?"], 
    "model_name": ["llama-3.3-70b-versatile"]
})  
    params_example = {
    "api_key": "sk-...",  
    "base_url": "https://api.groq.com/openai/v1"

}
    chat_template = [
    {
        "role": "system",
        "content": """You are an assistant answering questions using ONLY the context below.

Context:
{{context}}

Question: {{query}}

Answer strictly from the context. If the answer is not in the context, say you don't know."""
    }
]
    with mlflow.start_run(run_name="rag_model_demo") as run:

        
        #prompt = mlflow.genai.register_prompt(
        #       name="RAG-prompt",
        #      template=chat_template,
        #      commit_message="Initial commit",
       # tags={
       #     "author": "f.alibabaee@gmail.com",
       #     "task": "question-answering",
       #     "language": "en",
       # },
#)
    #print(f"Created prompt '{prompt.name}' (version {prompt.version})")
    #log the model artifact 
        model_info = mlflow.pyfunc.log_model(
            artifact_path="rag_model",
            python_model=PyfuncWithRetrieval(docs=None),
            artifacts={"vector_db": "vector_db.pkl"},
            input_example=pd.DataFrame({"queries": ["What is the ai4eosc project?"], "model_name": ["llama-3.3-70b-versatile"]}),    
            signature=infer_signature(
            params= params_example,
            model_input=input_example,
            model_output=pd.DataFrame({"query": ["What is the ai4eosc project?"], "answer": ["The ai4eosc project is an initiative to..."]})
        ), 
             
            pip_requirements=["openai", "sentence-transformers", "faiss-cpu", "nltk",'mlflow']
        )
        print(f"Model logged with run_id: {run.info.run_id} and model_uri: {model_info.model_uri}")