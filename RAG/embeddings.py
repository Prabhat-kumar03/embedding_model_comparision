from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
def generate_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        print(f"Unable to generate embeddings : {e}")