import numpy as np
from langchain_community.vectorstores import FAISS
from extract_pdf import extract_pdf
from text_splitter import text_splitter

# import faiss
# from RAG.vector_store import get_vector_store, search_vector_store
# from RAG.embeddings import generate_embeddings
from embeddings import generate_embeddings
from vector_store import get_vector_store
def cosine_distance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    cosine_similarity = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )
    
    return 1 - cosine_similarity

pdf_data = extract_pdf("pdf.pdf")
chunks_array = text_splitter(pdf_data)
vector_store = get_vector_store()
vector_store.add_documents(documents=chunks_array)
vector_store.save_local("faiss_index")

query = "Why can cancer mortality be a misleading surrogate for cancer incidence?"
embeddings = generate_embeddings()
query_embedding = embeddings.embed_query(query)
vector_store = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
results = vector_store.similarity_search(query=query, k=5)
print(results)

for doc in results:
    doc_embedding = embeddings.embed_query(doc.page_content)
    distance = cosine_distance(query_embedding, doc_embedding)
    print("Cosine distance:", distance)
    print("-" * 50)