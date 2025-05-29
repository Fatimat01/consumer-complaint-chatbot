import os
import logging
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from utils import setup_logging, compute_sha256
from more_itertools import chunked as batched


DATA_PATH = "data/complaints.csv"
PERSIST_DIR = "chroma_db"
LOG_PATH = "logs/ingestion.log"
BATCH_SIZE = 200

# setup logging
os.makedirs("logs", exist_ok=True)
setup_logging(log_path=LOG_PATH)

# function to load existing hashes from a file
def load_existing_hashes(path):
    hash_file = os.path.join(path, "doc_hashes.txt")
    if not os.path.exists(hash_file):
        return set()
    with open(hash_file, "r") as f:
        return set(line.strip() for line in f)

# function to save new hashes to a file
def save_new_hashes(path, new_hashes):
    hash_file = os.path.join(path, "doc_hashes.txt")
    with open(hash_file, "a") as f:
        for h in new_hashes:
            f.write(f"{h}\n")

# main ingestion function
def ingest():
    logging.info("Starting ingestion process")
    loader = CSVLoader(file_path=DATA_PATH)
    all_docs = loader.load()
    logging.info(f"Loaded {len(all_docs)} documents from {DATA_PATH}")

    existing_hashes = load_existing_hashes(PERSIST_DIR)
    new_docs, new_hashes = [], set()

    for doc in all_docs:
        doc_hash = compute_sha256(doc.page_content)
        if doc_hash not in existing_hashes:
            doc.metadata["hash"] = doc_hash
            new_docs.append(doc)
            new_hashes.add(doc_hash)

    if not new_docs:
        logging.info("No new documents to ingest.")
        print("No new documents found.")
        return

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)

    all_chunks = []
    for i, doc_batch in enumerate(batched(new_docs, BATCH_SIZE)):
        logging.info(f"Processing batch {i + 1}")
        chunks = splitter.split_documents(doc_batch)
        vectorstore.add_documents(chunks)
        all_chunks.extend(chunks)
        logging.info(f"Added {len(chunks)} chunks to vectorstore.")

    save_new_hashes(PERSIST_DIR, new_hashes)
    logging.info(f"Ingestion complete. Total new chunks added: {len(all_chunks)}")
    print(f"Ingestion complete. Total new chunks added: {len(all_chunks)}")

if __name__ == "__main__":
    ingest()