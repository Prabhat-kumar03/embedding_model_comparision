from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from extract_pdf import extract_pdf
from text_splitter import text_splitter

embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/embeddinggemma-300m-medical",
        model_kwargs={"device": "cpu"},  # change to "cuda" if GPU is available
        encode_kwargs={"normalize_embeddings": True}  # normalized vectors for better similarity
    )

def cosine_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    cosine_similarity = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )
    
    return 1 - cosine_similarity


pdf1_data = extract_pdf("patient1_blood_cancer.pdf")
pdf2_data = extract_pdf("patient2_oral_cancer.pdf")

chunks_array_1 = text_splitter(pdf1_data)
chunks_array_2 = text_splitter(pdf2_data)

pdf1_vectors = [embeddings.embed_query(chunk.page_content) for chunk in chunks_array_1]
pdf2_vectors = [embeddings.embed_query(chunk.page_content) for chunk in chunks_array_2]

pdf1_avg = np.mean(pdf1_vectors, axis=0)
pdf2_avg = np.mean(pdf2_vectors, axis=0)

distance = cosine_distance(pdf1_avg, pdf2_avg)
print("Cosine distance between PDF1 and PDF2:", distance)