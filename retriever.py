import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EMB_DIR = BASE_DIR / "Data" / "embeddings"

class NCORetriever:

    def __init__(self):

        self.index = faiss.read_index(str(EMB_DIR / "nco2015.faiss"))

        with open(EMB_DIR / "nco2015_metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Loaded vectors:", self.index.ntotal)
        print("Loaded metadata:", len(self.metadata))

    def search(self, query: str, top_k: int = 5):

        q_emb = self.model.encode([query]).astype("float32")

        distances, indices = self.index.search(q_emb, top_k)

        results = []

        for idx, dist in zip(indices[0], distances[0]):

            job = self.metadata[idx]

            results.append({
                "nco_2015": job["nco_2015"],
                "title": job["title"],
                "description": job.get("description", "")[:500],
                "score": float(dist)
            })

        return results