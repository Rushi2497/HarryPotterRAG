import os
import uuid
import chromadb
import numpy as np
from typing import List, Any
from src.config_loader import load_config

config = load_config()

class ChromaVectorStore:
    
    def __init__(self, collection_name: str = config['DATA_PATHS']['COLLECTION_NAME'], persist_directory: str = config['DATA_PATHS']['VECTOR_STORE_DIR']):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
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
        max_batch_size=2000

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