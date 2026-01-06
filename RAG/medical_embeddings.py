from langchain_huggingface import HuggingFaceEmbeddings

def generate_medical_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            model_kwargs={"device": "cpu"},  # change to "cuda" if needed
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings
    except Exception as e:
        print(f"Unable to generate embeddings: {e}")
        return None
