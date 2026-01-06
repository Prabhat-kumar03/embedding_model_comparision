from transformers import pipeline

pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

prompt = "Hi there, You are an advanced AI able to answer the queries in a formal format.As there are two zero's in hundred, 3 in thousand, 4 in ten thousand, similarly Count the number of zero's in a billion."
output = pipe(prompt, max_new_tokens=100)
print(output[0]['generated_text'])