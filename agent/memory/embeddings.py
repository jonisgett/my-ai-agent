"""
Local embeddings — FastEmbed (384-dim, ONNX). Zero API calls.

This module converts text into numerical vectors for similarity search.
Everything runs locally on your machine — no data leaves your computer.
"""

import struct

# We lazy load the model so the app starts fast
_embed_model = None

def _get_model():
    global _embed_model

    if _embed_model is not None:
        return _embed_model
    
    try:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("✅ FastEmbed loaded (384-dim, ONNX, fully local)")
        return _embed_model
    except ImportError:
        raise ImportError(
            "❌ FastEmbed is not installed! Memory search won't work without it.\n"
            "   Stop Agent!\n"
            "   Install it with: pip install fastembed\n"
            "   Restart Agent then try again."
        )

# Embed a single text string   
def embed_text(text: str) -> list[float]:
    """
    Convert a piece of text into a 384-dimensional vector.
    
    This is the core function of the embedding system. You give it
    any text, and it returns a list of 384 floating-point numbers
    that represent the "meaning" of that text.
    
    Parameters
    ----------
    text : str
        Any text you want to embed
        
    Returns
    -------
    list[float]
        A list of 384 numbers representing the text's meaning
        
    Example
    -------
    >>> vec = embed_text("I love programming in Python")
    >>> len(vec)
    384
    >>> type(vec[0])
    float
    """
    model = _get_model()
    
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()

# Process multiple texts at the same time    
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts at once (more efficient than one-by-one)."""
    model = _get_model()

    return [e.tolist() for e in model.embed(texts)]
    
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Measure how similar two vectors are.
    
    Returns a number between -1 and 1:
    -  1.0 = identical meaning
    -  0.0 = completely unrelated
    - -1.0 = opposite meaning
    
    The math: cos(θ) = (A · B) / (|A| × |B|)
    
    Don't worry if the math isn't clear — what matters is:
    higher number = more similar.
    """
    dot = sum(x * y for x, y in zip(a, b))
    # Find the magnitude of each vector
    norm_a = sum(x * x for x in a) ** .05
    norm_b = sum(x * x for x in b) ** .05

    # Avoid division by zero if either of our vectors are 0
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)

def serialize_vector(vec: list[float]) -> bytes:  
    """
    Convert a vector to bytes for storage in SQLite.
    
    SQLite doesn't have a "vector" column type, so we pack our
    list of floats into raw bytes using Python's struct module.
    
    Each float is 4 bytes, so a 384-dim vector = 1,536 bytes.
    """
    return struct.pack(f"{len(vec)}f", *vec)

def deserialize_vector(data: bytes) -> list[float]:
    """Convert bytes back to a vector (reverse of serialize_vector)."""
    n = len(data) // 4 # Each float is 4 bytes
    return list(struct.unpack(f"{n}f", data))
