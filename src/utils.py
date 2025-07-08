import logging
import hashlib

def setup_logging(log_path="logs/app.log"):
    """
    Sets up logging for the application.
    """
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="a",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

def compute_sha256(text: str) -> str:
    """
    Computes SHA256 hash of a string (used for deduplication of documents).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def format_source_metadata(doc) -> str:
    """
    Formats metadata from a document for display in the UI.
    """
    company = doc.metadata.get("Company", "Unknown")
    product = doc.metadata.get("Product", "Unknown")
    return f"**{company}** – {product}"

def count_tokens(text: str) -> int:
    """
    Approximate token count (can be useful for debugging or rate limit warnings).
    """
    return len(text.split())