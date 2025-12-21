import os
from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from voyageai import Client
import numpy as np
from src.config_loader import load_config

config = load_config()

class EmbeddingPipeline:
    def __init__(self, model_name: str = config['RAG_MODELS']['EMBEDDING_MODEL'], chunk_size: int = config['CHUNKING_PARAMS']['CHUNK_SIZE'], chunk_overlap: int = config['CHUNKING_PARAMS']['CHUNK_OVERLAP']):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
        print(f'[INFO] Loaded embedding model: {model_name}')
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:

        token_length_function = lambda text: self.model.count_tokens(texts=[text],model=config['RAG_MODELS']['TOKENIZER_MODEL'])

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = token_length_function,
            separators=['\n\n','\n',' ', '']
        )
        chunks = splitter.split_documents(documents)
        print(f'[INFO] Split {len(documents)} documents split into {len(chunks)} chunks.')
        return chunks
    
    def embed_chunks(self, chunks: List[Any], batch_size: int = 128) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        if not isinstance(chunks[0],str):
            texts = [chunk.page_content for chunk in chunks]
        else:
            texts = chunks

        total = len(texts)
        print(f'[INFO] Generating embeddings for {total} chunks in batches of {batch_size} ...')

        all_embeddings = []
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f'[DEBUG] Embedding batch {i // batch_size + 1}: items {i} to {i + len(batch_texts) - 1}')
            resp = self.model.embed(texts=batch_texts, model=config['RAG_MODELS']['EMBEDDING_MODEL'], output_dimension=1024)
            batch_embeddings = resp.embeddings
            all_embeddings.extend(batch_embeddings)

        embeddings_np = np.array(all_embeddings)
        print(f'[INFO] Generated embeddings with shape: {embeddings_np.shape}')
        return embeddings_np