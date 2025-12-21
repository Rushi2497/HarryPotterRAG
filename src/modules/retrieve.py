from typing import List, Any, Dict
from src.modules.vectorstore import ChromaVectorStore
from src.modules.embed import EmbeddingPipeline
from voyageai import Client
from src.config_loader import load_config

config = load_config()

class RAGRetriever:

    def __init__(self, vector_store: ChromaVectorStore, embedding_pipeline: EmbeddingPipeline, reranker: Client):
        self.vector_store = vector_store
        self.embedding_pipeline = embedding_pipeline
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 50, rerank: bool = True, top_n: int = 10, score_threshold: float = -1.0) -> List[Dict[str, Any]]:
        print(f'[INFO] Retrieving documents for query: {query}')
        print(f'[PARAMETERS] Top K: {top_k}, Top N: {top_n if rerank else 'No Reranking'}, Score threshold: {score_threshold}')

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
                resp = self.reranker.rerank(query=query, documents=documents, model=config['RAG_MODELS']['RERANKER_MODEL']).results
                scores =[item.relevance_score for item in resp]
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
            print(f'[INFO] Retrieved {len(retrieved_docs)} documents (after filtering)')
            
            return retrieved_docs
        
        except Exception as e:
            print(f'[ERROR] Error during retrievel: {e}')
            return []