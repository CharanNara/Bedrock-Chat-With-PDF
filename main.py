
import os
import json
import boto3
import logging
import markdown2
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS


class RAGConfig:
    def __init__(self):
        self.aws_region = os.getenv("AWS_REGION", "us-east-2")
        self.embedding_model_id = "amazon.titan-embed-text-v2:0"
        self.chat_model_id = "arn:aws:bedrock:us-east-2:285982080176:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.max_tokens = 1000
        self.top_k = 250
        self.temperature = 1.0
        self.top_p = 0.999
        self.vector_stores_dir = Path("vector_stores")
        self.uploads_dir = Path("uploads")

        self.vector_stores_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)

class PDFProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len
        )
        
    def extract_text_from_pdf(self, file_path: str):
        try:
            reader = PdfReader(file_path)
            raw_text = ""
            for page in reader.pages:
                raw_text += page.extract_text() + "\n"
            return raw_text
        except Exception as e:
            logging.error(f"Error reading PDF file {file_path}: {e}")
            raise HTTPException(status_code=400, detail=f"Error reading PDF file {file_path}: {e}")
    
    def split_text_into_chunks(self, text: str):
        try:    
            chunks = self.text_splitter.split_text(text)
            logging.info(f"Text split into {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logging.error(f"Error splitting text into chunks: {e}")
            raise HTTPException(status_code=500, detail=f"Error splitting text into chunks: {e}")

class EmbeddingService:
    def __init__(self, config: RAGConfig, client):
        self.config = config
        self.client = client
        self.embeddings = BedrockEmbeddings(
            model_id = self.config.embedding_model_id,
            client = self.client
        )

    def create_vector_store(self, chunks: List[str], store_name: str):
        try:
            vector_store = FAISS.from_texts(chunks, self.embeddings)
            store_path = self.config.vector_stores_dir / store_name
            vector_store.save_local(str(store_path))

            logging.info(f"Vector store saved at {store_path}")
            return vector_store
        except Exception as e:
            logging.error(f"Error creating vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating vector store: {e}")
    
    def load_vector_store(self, store_name: str):
        try:
            store_path = self.config.vector_stores_dir / store_name
            if not store_path.exists():
                raise HTTPException(status_code=404, detail=f"Vector store {store_name} not found.")
            
            vector_store = FAISS.load_local(str(store_path), embeddings=self.embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            logging.error(f"Error loading vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading vector store: {e}")
        

class ChatService:
    def __init__(self, config: RAGConfig, client):
        self.config = config
        self.client = client
        self.prompt_template = """
Use the following pieces of context to answer the question at the end to provide a concise and accurate answer.
At least summarize with 100 words with detailed explanation. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<Context>
{context}
</Context>

<Question>
{question}
</Question>

Answer in markdown format with detailed explanation.
"""
    def retrieve_relevant_chunks(self, vector_store: FAISS, query: str, k: int = 10):
        try:
            docs = vector_store.similarity_search(query, k = k)
            return docs
        except Exception as e:
            logging.error(f"Error retrieving relevant chunks: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving relevant chunks: {e}")
        
    def generate_answer(self, context: str, question: str):
        try:
            prompt = self.prompt_template.format(context=context, question=question)
            response = self.client.converse(
                modelId=self.config.chat_model_id,
                messages=[
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                inferenceConfig={
                    "maxTokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "topP": self.config.top_p
                },
                additionalModelRequestFields={
                    "top_k": self.config.top_k
                },
                performanceConfig={
                    "latency": "standard"
                }

            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")
        
class RAGService:
    def __init__(self):
        self.config = RAGConfig()
        self.client = boto3.client("bedrock-runtime", region_name=self.config.aws_region)
        self.pdf_prcoessor = PDFProcessor(self.config)
        self.embedding_service = EmbeddingService(self.config, self.client)
        self.chat_service = ChatService(self.config, self.client)
        
    def process_pdf(self, file_path: str):
        raw_text = self.pdf_prcoessor.extract_text_from_pdf(file_path)
        chunks = self.pdf_prcoessor.split_text_into_chunks(raw_text)
        store_name = f"faiss_{Path(file_path).stem}"
        vector_store = self.embedding_service.create_vector_store(chunks, store_name)
        
        return store_name, len(chunks)
    
    def chat_with_pdf(self, store_name: str, query: str):
        vector_store = self.embedding_service.load_vector_store(store_name)
        docs = self.chat_service.retrieve_relevant_chunks(vector_store, query, k=5)
        context = "\n".join([doc.page_content for doc in docs])

        answer = self.chat_service.generate_answer(context, query)
        return answer
    
app = FastAPI(
    title="RAG Chat with PDF",
    description="A RAG-based application for chatting with PDF documents using AWS Bedrock and Claude",
    version="1.0.0"
)
rag_service = RAGService()

@app.get("/")
async def root():
    return {
        "message": "Welcome to the RAG Chat with PDF API!",
        "endpoints": {
            "upload_pdf": "/upload-pdf/",
            "chat": "/chat/",
            "vector_stores": "/vector-stores/",
            "health": "/health/",
            "openapi": "/openapi.json",
            "redoc": "/redoc",
            "swagger_ui": "/docs",
            "docs": "/docs"
        }
    }   

class PDFUploadResponse(BaseModel):
    store_name: str = Field(..., description="Name of the created vector store")
    num_chunks: int = Field(..., description="Number of text chunks created from the PDF")



@app.post("/upload-pdf/", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    upload_path = rag_service.config.uploads_dir / file.filename
    with open(upload_path, "wb") as f:
        f.write(await file.read())
    
    store_name, num_chunks = rag_service.process_pdf(str(upload_path))
    return PDFUploadResponse(store_name=store_name, num_chunks=num_chunks)

class ChatRequest(BaseModel):
    store_name: str = Field(..., description="Name of the vector store to use")
    query: str = Field(..., description="User query to ask about the PDF content")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from the model") 


@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the uploaded PDF content"""
    answer = rag_service.chat_with_pdf(request.store_name, request.query)
    html = markdown2.markdown(answer)
    return HTMLResponse(content=html)


@app.get("/vector-stores/", response_model=List[str])
async def list_vector_stores():
    """List available vector stores"""
    stores = [p.name for p in rag_service.config.vector_stores_dir.iterdir() if p.is_dir()]
    return stores


@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)