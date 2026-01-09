# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="LFM2-350M Chat API", version="1.0")

# Load model once at startup
MODEL_DIR = "./model"
MODEL_FILE = "LFM2-350M-Q4_0.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Ensure it was downloaded during build.")

print("Loading LLM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("Model loaded.")

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant. Provide clear, concise, accurate responses."

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ]
        response = llm.create_chat_completion(messages=messages)
        reply = response["choices"][0]["message"]["content"]
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}
