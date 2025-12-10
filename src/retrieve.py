from typing import List, Any, Dict
from src.vectorstore import ChromaVectorStore
from src.embed import EmbeddingPipeline
from sentence_transformers.cross_encoder import CrossEncoder

class RAGRetriever:

    def __init__(self, vector_store: ChromaVectorStore, embedding_pipeline: EmbeddingPipeline, reranker: CrossEncoder):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 50, rerank: bool = True, top_n: int = 10, score_threshold: float = -1.0) -> List[Dict[str, Any]]:
        print(f'[INFO] Retrieving documents for query: {query}')
        print(f'Top K: {top_k}, Top N: {top_n if rerank else 'No Reranking'}, Score threshold: {score_threshold}')

        query_embedding = self.embedding_pipeline.embed_chunks([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if not results['documents'] or not results['documents'][0]:
                print('[INFO] No documents found')
                return []
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]

            if rerank:
                query_doc_pairs = [(query, doc) for doc in documents]
                scores = self.reranker.predict(sentences=query_doc_pairs)
                res = sorted(zip(ids, documents, metadatas, distances, scores), key=lambda x: x[4], reverse=True)[:top_n]
            else:
                norerank = [None]*len(ids)
                res = list(zip(ids, documents, metadatas, distances, norerank))[:top_n]

            for i, (doc_id, document, metadata, distance, rerank_score) in enumerate(res):
                similarity_score = 1 - distance

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': doc_id,
                        'content': document,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'rerank_score': rerank_score,
                        'rank': i + 1
                    })
            print(f'Retrieved {len(retrieved_docs)} documents (after filtering)')
            
            return retrieved_docs
        
        except Exception as e:
            print(f'Error during retrievel: {e}')
            return []