from langchain_huggingface import HuggingFaceEmbeddings

def generate_medembed_large_embeddings():
    """
    Generates a HuggingFace-based MedEmbed large embedding object
    ready to use with LangChain vector stores or for semantic search.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="abhinand/MedEmbed-large-v0.1",
            model_kwargs={"device": "cpu"},  # change to "cuda" if GPU is available
            encode_kwargs={"normalize_embeddings": True}  # normalized vectors for better similarity
        )
        print("MedEmbed-large embeddings initialized successfully!")
        return embeddings
    except Exception as e:
        print(f"Unable to generate MedEmbed embeddings: {e}")
        return None
