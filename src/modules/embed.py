from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from transformers import AutoTokenizer
import numpy as np
from src.config_loader import load_config

config = load_config()

class EmbeddingPipeline:
    def __init__(self, model_name: str = config['RAG_MODELS']['EMBEDDING_MODEL'], chunk_size: int = config['CHUNKING_PARAMS']['CHUNK_SIZE'], chunk_overlap: int = config['CHUNKING_PARAMS']['CHUNK_OVERLAP']):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = OllamaEmbeddings(model=model_name, num_ctx=512)
        print(f'[INFO] Loaded embedding model: {model_name}')
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:

        tokenizer = AutoTokenizer.from_pretrained(config['RAG_MODELS']['TOKENIZER_MODEL'], local_files_only=True)
        token_length_function = lambda text: len(tokenizer.encode(text))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap = self.chunk_overlap,
            length_function = token_length_function,
            separators=['\n\n','\n',' ', '']
        )
        chunks = splitter.split_documents(documents)
        print(f'[INFO] Split {len(documents)} documents split into {len(chunks)} chunks.')
        return chunks
    
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        if not isinstance(chunks[0],str):
            texts = [chunk.page_content for chunk in chunks]
        else:
            texts = chunks
        
        print(f'[INFO] Generating embeddings for {len(texts)} chunks ...')
        embeddings = self.model.embed_documents(texts)
        embeddings_np = np.array(embeddings)
        print(f'[INFO] Generated embeddings with shape: {embeddings_np.shape}')
        return embeddings_np