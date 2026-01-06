from RAG.process_pdf import process_pdf
from RAG.text_splitter import text_splitter
from RAG.vector_store import get_vector_store
from RAG.vector_store_2 import get_vector_store_2
from fastapi import UploadFile

async def upload_pdf(file: UploadFile):
    try:
        pdf_data = await process_pdf(file)
        chunks_array = text_splitter(pdf_data)
        print("chunks_array:", type(chunks_array), chunks_array)
        vector_store = get_vector_store()
        vector_store.add_documents(documents=chunks_array)
        vector_store.save_local("faiss_index")
        print("Pdf file uploaded successfully in vector_store.")
        vector_store_2 = get_vector_store_2()
        vector_store_2.add_documents(documents=chunks_array)
        vector_store_2.save_local("med_faiss_index")
        print("Pdf file uploaded successfully in vector_store.")
        return {"success": True}
    except Exception as e:
        print("Error occured while saving data :", e)
        return {"success": False}

if __name__ == "__main__":
    upload_pdf("pdf.pdf")