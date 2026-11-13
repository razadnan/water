"""
Semantic Similarity Memory Store for Past Incident Retrieval.
Enables contextual recall of similar historical incidents based on:
- Zone proximity & pipe material
- Risk level and failure pattern (burst, joint failure, valve)
- TF-IDF cosine similarity over incident problem descriptions
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from database import db


class SemanticMemoryStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def recall_similar_incidents(self, zone: str, problem_desc: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves past incidents ranked by semantic similarity score,
        blending lexical problem match with zone alignment.
        """
        all_incidents = db.list_incidents(limit=50)
        if not all_incidents:
            return []

        # Corpus of problems and recommendations
        corpus = [f"{inc.get('problem', '')} {inc.get('recommendation', '')}" for inc in all_incidents]
        query = f"{zone} {problem_desc}"

        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus + [query])
            query_vec = tfidf_matrix[-1]
            doc_vecs = tfidf_matrix[:-1]
            sims = cosine_similarity(query_vec, doc_vecs).flatten()
        except ValueError:
            # Fallback if vocabulary is too sparse
            sims = np.zeros(len(all_incidents))

        ranked = []
        for idx, inc in enumerate(all_incidents):
            score = float(sims[idx]) if idx < len(sims) else 0.0
            # Zone match bonus
            if inc.get("zone", "").lower() == zone.lower():
                score += 0.25
            item = dict(inc)
            item["similarity_score"] = round(min(1.0, score), 3)
            ranked.append(item)

        ranked.sort(key=lambda x: x["similarity_score"], reverse=True)
        return ranked[:top_k]


def query_incident_memory(zone: str, problem_description: str) -> List[Dict[str, Any]]:
    store = SemanticMemoryStore()
    return store.recall_similar_incidents(zone, problem_description)
