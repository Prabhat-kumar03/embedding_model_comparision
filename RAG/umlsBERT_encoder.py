from langchain_huggingface import HuggingFaceEmbeddings

def generate_umlsbert_embeddings():
    """
    Generates a HuggingFace-based UMLS-BERT embedding object
    ready to use with LangChain vector stores or for semantic search.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="pankajrajdeo/UMLS-ED-Bioformer-8L-V-1",
            model_kwargs={"device": "cpu"},  # change to "cuda" if GPU is available
            encode_kwargs={"normalize_embeddings": True}  # normalized vectors for better similarity
        )
        print("UMLS-BERT embeddings initialized successfully!")
        return embeddings
    except Exception as e:
        print(f"Unable to generate UMLS-BERT embeddings: {e}")
        return None