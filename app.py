from fastapi import FastAPI, UploadFile, File, Body
import shutil
from dotenv import load_dotenv
import time
from pydantic import BaseModel
import os
from langchain_openai import ChatOpenAI
from utils.pdf_loader import extract_text_from_pdf
from utils.chunker import chunk_text
from utils.vector_store import create_vector_store, save_vector_store,load_vector_store
import uuid

load_dotenv()

app = FastAPI(
	title ="AI Document Assistant",
	version ="1.0.0")
	
# Globals
UPLOAD_DIR = "data"
VECTORSTORE_BASE_DIR = "vectorstores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_BASE_DIR, exist_ok=True)

cache = {}

# Request model
class AskRequest(BaseModel):
    doc_id: str
    question: str
	
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text_from_pdf(file_path)

    if not text.strip():
        return {"error": "No text extracted from PDF"}

    chunks = chunk_text(text)

    vector_store = create_vector_store(chunks)

    doc_vector_path = os.path.join(VECTORSTORE_BASE_DIR, doc_id)
    save_vector_store(vector_store, doc_vector_path)

    

    return {
    	"doc_id": doc_id,
        "message": f"{file.filename} uploaded successfully",
        "text_length": len(text),
        "num_chunks": len(chunks),
        "vector_store_status": "saved"
    }


@app.post("/ask")
async def ask_question(request: AskRequest):
    start_time = time.time()

    doc_id = request.doc_id
    question = request.question

    cache_key = f"{doc_id}:{question}"

    # 1️⃣ Cache check
    if cache_key in cache:
        return cache[cache_key]

    doc_vector_path = os.path.join(VECTORSTORE_BASE_DIR, doc_id)

    if not os.path.exists(doc_vector_path):
        return {"error": "Invalid doc_id. Please upload document again."}

    try:
        vector_store = load_vector_store(doc_vector_path)

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        retrieval_start = time.time()
        docs = retriever.invoke(question)
        retrieval_time = time.time() - retrieval_start

        if not docs:
            response_data = {
                "doc_id": doc_id,
                "answer": "No relevant context found.",
                "sources": [],
                "retrieval_time_seconds": round(retrieval_time, 3),
                "llm_time_seconds": 0,
                "latency_seconds": round(time.time() - start_time, 3)
            }

            cache[cache_key] = response_data
            return response_data

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""

        llm_start = time.time()

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(prompt)

        llm_time = time.time() - llm_start
        total_latency = time.time() - start_time

        response_data = {
            "doc_id": doc_id,
            "answer": response.content,
            "sources": [doc.page_content[:300] for doc in docs],
            "retrieval_time_seconds": round(retrieval_time, 3),
            "llm_time_seconds": round(llm_time, 3),
            "latency_seconds": round(total_latency, 3)
        }

        # 2️⃣ Save to cache
        cache[cache_key] = response_data

        return response_data

    except Exception as e:
        return {"error": str(e)}

