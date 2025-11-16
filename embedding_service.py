from ray import serve
from sentence_transformers import SentenceTransformer


@serve.deployment(name="embedding_api")
class EmbeddingAPI:
    def __init__(self):
        print("🔹 Loading embedding model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def encode(self, body: dict):
        text = body.get("text", "")
        return {"embedding": self.model.encode(text).tolist()}


def app(config: dict | None = None):
    # Only bind internal deployment (no HTTP routes here)
    return EmbeddingAPI.bind()
