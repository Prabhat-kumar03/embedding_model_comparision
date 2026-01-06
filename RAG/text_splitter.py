from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 

def text_splitter(text : str):
    try:
        chunk_array = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        for chunk in text_splitter.split_text(text):
            print("----------- Chunk:",chunk)
            chunk_array.append(Document("".join(chunk)))
        print("chunks created succesfully")
        return chunk_array
    except Exception as e:
        print(f"Error while splitting text: {e}")