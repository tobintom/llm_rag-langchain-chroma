"""
apprag.py
RAG pipeline:
- Load PDFs from a folder
- Split to chunks
- Embed with BAAI/bge-small-en-v1.5 via LangChain community BGE wrapper
- Persist vectors to ChromaDB
- Query via a ConversationalRetrievalChain using Ollama (llama3.2)
"""

from pathlib import Path
import os
import gradio as gr
# langchain & loaders
from langchain.document_loaders import PyPDFLoader   
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage

# embeddings (LangChain community BGE wrapper)
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# vector store (Chroma)
from langchain.vectorstores import Chroma

# Ollama LLM wrapper
from langchain_ollama import ChatOllama  # conversational wrapper 

# utilities
from langchain.schema import Document

def index_pdfs_to_chroma(pdf_dir: str, persist_dir: str, bge_device: str = "cpu"):
    pdf_dir = Path(pdf_dir)
    assert pdf_dir.exists(), f"{pdf_dir} not found"

    # 1) load all PDFs
    docs = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        print("Loading", pdf_path.name)
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        docs.extend(pages)

    # 2) split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    print("Splitting into chunks...")
    docs = text_splitter.split_documents(docs)

    # 3) create BGE embeddings (use CPU or cuda)
    # model_name can be "BAAI/bge-small-en-v1.5"
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": bge_device}
    encode_kwargs = {"normalize_embeddings": True}
    print("Creating HuggingFaceBgeEmbeddings (this will load the model)...")
    bge = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # 4) create/persist Chroma DB
    print("Creating Chroma vectorstore (persisted)...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=bge,
        persist_directory=persist_dir,
        collection_name="pdfs_collection"
    )
    vectordb.persist()
    print("Indexing complete. persisted to", persist_dir)
    return vectordb

def make_qa_chain(vectordb, ollama_model_name="llama3.2:1b"):
    # create retriever
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
   
    llm = ChatOllama(model=ollama_model_name, base_url=None)  # base_url="http://localhost:11434" if needed
    # ConversationalRetrievalChain wraps the LLM + retriever
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
    return chain

PDF_DIR = "./docs"           # folder containing your PDFs
PERSIST_DIR = "./chroma_persist"
chat_history = []

# set to 'cuda' if you have GPU and BGE on GPU, else 'cpu'
BGE_DEVICE = "cpu"

# 1) Index PDFs to Chroma (first-run)
vectordb = index_pdfs_to_chroma(PDF_DIR, PERSIST_DIR, bge_device=BGE_DEVICE)

# 2) Build chain
qa_chain = make_qa_chain(vectordb, ollama_model_name="llama3.2")

def chat(question, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=question))

    result = qa_chain.invoke({"question": question,"chat_history":history_langchain_format})
    sources = result.get("source_documents", [])
    answer = result["answer"]
    print("\nAssistant:", answer, "\n")
    if sources:
        print("Top source chunks (text snippets):")
        for i, doc in enumerate(sources[:3], start=1):
            print(f"--- source {i} (metadata: {doc.metadata}):")
            snippet = doc.page_content
            print(snippet[:600].strip(), "\n---\n") 
    
    return answer

 
if __name__ == "__main__":
    view = gr.ChatInterface(chat,type="messages").launch(inbrowser=True)