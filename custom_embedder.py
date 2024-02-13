import os
from chromadb.api.types import (
    Documents,
    EmbeddingFunction
)
import requests

class CustomEmbedder(EmbeddingFunction):
    def __call__(self, input: Documents):
        rest_client = requests.Session()
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        EM_API_TOKEN = os.environ["EM_API_TOKEN"] #Set a API_TOKEN environment variable before running
        API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}" #"https://c9ejquzh6yum3xqf.us-east-1.aws.endpoints.huggingface.cloud/"
        response = rest_client.post(
            API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {EM_API_TOKEN}"}
        ).json()
        return response
    
    def embed_documents(self, texts):
        return self(texts)
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]