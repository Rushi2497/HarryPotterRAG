import os
import uuid
import chromadb
import numpy as np
from typing import List, Any, Dict
from src.embedding import EmbeddingPipeline
from sentence_transformers.cross_encoder import CrossEncoder

class ChromaVectorStore:
    
    def __init__(self, collection_name: str = 'HP_Books', persist_directory: str = './data/vector_store'):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_pipeline = EmbeddingPipeline()
        self._initialize_store()
    
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={'description':'Harry Potter book embeddings for RAG'}
            )
            print(f'Vector store initialized. Collection: {self.collection_name}')
            print(f'Existing documents in collection: {self.collection.count()}')

        except Exception as e:
            print(f'Error initializing vector store: {e}')
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        
        print(f'Adding {len(documents)} documents to vector store...')

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents,embeddings)):
            doc_id = f'doc_{uuid.uuid4().hex[:8]}_{i}'
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        total_docs = len(documents)
        max_batch_size=5000

        try:
            for batch_start in range(0, total_docs, max_batch_size):
                batch_end = min(batch_start + max_batch_size, total_docs)

                batch_ids = ids[batch_start:batch_end]
                batch_docs = documents_text[batch_start:batch_end]
                batch_meta = metadatas[batch_start:batch_end]
                batch_embs = embeddings_list[batch_start:batch_end]

                print(f" → Adding batch {batch_start} to {batch_end} ...")

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embs,
                    metadatas=batch_meta,
                    documents=batch_docs
                )

            print(f'Sucessfully added {len(documents)} documents to vector store')
            print(f'Total documents in collection: {self.collection.count()}')
        except Exception as e:
            print(f'Error adding documents to vector store: {e}')
            raise

    def retrieve(self, query: str, top_k: int = 50, rerank: bool = True, top_n: int = 10, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
            print(f'Retrieving documents for query: {query}')
            print(f'Top K: {top_k}, Top N: {top_n}, Score threshold: {score_threshold}')

            query_embedding = self.embedding_pipeline.embed_chunks([query])[0]

            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k
                )

                retrieved_docs = []

                if results['documents'] and results['documents'][0]:
                    documents = results['documents'][0]
                    metadatas = results['metadatas'][0]
                    distances = results['distances'][0]
                    ids = results['ids'][0]

                    if rerank:
                        reranker = CrossEncoder(model_name_or_path='cross-encoder/ms-marco-MiniLM-L6-v2')
                        query_doc_pairs = [(query, doc) for doc in documents]
                        scores = reranker.predict(sentences=query_doc_pairs)
                        res = [(id, docs, meta, dist, sc) for id, docs, meta, dist, sc in sorted(zip(ids, documents, metadatas, distances, scores), key=lambda x: x[4], reverse=True)[:top_n]]
                    else:
                        res = [(id, docs, meta, dist, None) for id, docs, meta, dist in zip(ids, documents, metadatas, distances)]

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
                else:
                    print('No documents found')
                
                return retrieved_docs
            
            except Exception as e:
                print(f'Error during retrievel: {e}')
                return []