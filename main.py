from fastapi import FastAPI, UploadFile, BackgroundTasks ,Form,File
from transformers import pipeline
from RAG.save_data_to_vectordb import upload_pdf
from RAG.vector_store import search_vector_store
from RAG.vector_store_2 import search_vector_store_2
from fastapi.responses import JSONResponse
app = FastAPI()

generator = pipeline("text-generation",
                        model="microsoft/Phi-3-mini-4k-instruct",
                        device=-1) 
                        
@app.post("/query")
def medical_response_generator(prompt :str = Form(...)):
    """
    FastAPI endpoint: Takes a user query and returns AI-generated text.
    """
    output = generator(prompt, max_new_tokens=200)
    print(output[0]["generated_text"])
    return {"response": output[0]["generated_text"]}

@app.post("/upload-file")
def pdf_upload_handler(background_task: BackgroundTasks, file: UploadFile = File(...)):
    try:
        background_task.add_task(upload_pdf, file)
        return JSONResponse(
            status_code=200, content={"message": "File Uploaded Successfully"}
        )
    except Exception as e:
        print("Error occured while uploading file to Vector DB- ", e)
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )
        
@app.post("/get-answer-from-rag")
def get_answer_from_RAG( query : str = Form(...)):
    return search_vector_store(query)
@app.post("/get-answer-from-rag_2")
def get_answer_from_RAG( query : str = Form(...)):
    return search_vector_store_2(query)