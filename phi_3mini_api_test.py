from fastapi import FastAPI
from transformers import pipeline
app = FastAPI()

@app.get("/query")
def medical_response_generator(query):
    generator = pipeline("text-generation",
                        model="microsoft/Phi-3-mini-4k-instruct",
                        device=-1)  # -1 = CPU (or API automatically)
                        
    prompt = "Hi there, I am suffering from sneezing, cough and cold. At times I feel fever too. Whenever I take the medicine i get relief for a while but then after few hours it starts again. what could be the major problem."
    output = generator(prompt, max_new_tokens=100)
    return (output[0]['generated_text'])