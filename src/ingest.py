import os
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

DATA_PATH = "data/complaints.csv"
PERSIST_DIR = "chroma_db"
LOG_PATH = "logs/ingestion.log"


# configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def ingest():
    logging.info("Starting ingestion process.")
    print("Loading data...")
    loader = CSVLoader(file_path=DATA_PATH)
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents from {DATA_PATH}")

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")

    print(f"Created {len(chunks)} chunks. Generating embeddings and persisting to {PERSIST_DIR}...")
    embedding = OpenAIEmbeddings()

    chroma_path = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    if os.path.exists(chroma_path):
        print("Vector store exists — appending new documents...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding
        )
        vectorstore.add_documents(chunks)
    else:
        print("No vector store found — creating a new one...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=PERSIST_DIR
        )

    vectorstore.persist()
    logging.info("Successfully persisted vector store.")
    print("Ingestion complete and persisted.")

if __name__ == "__main__":
    ingest()