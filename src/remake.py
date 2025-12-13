import numpy as np
from src.modules.load import load_documents
from src.modules.embed import EmbeddingPipeline
from src.modules.vectorstore import ChromaVectorStore
from src.config_loader import load_config

if __name__ == '__main__':
    
    config = load_config()

    documents = load_documents()

    embedding_pipeline = EmbeddingPipeline()
    chunks = embedding_pipeline.chunk_documents(documents)
    embeddings = embedding_pipeline.embed_chunks(chunks)

    np.savez('./data/embeddings_minilm_500_tokens',embeddings)
    embeddings = np.load('./data/embeddings_minilm_500_tokens.npz')['arr_0']

    vector_store = ChromaVectorStore()
    vector_store.client.delete_collection(name=config['DATA_PATHS']['COLLECTION_NAME'])

    vector_store = ChromaVectorStore(collection_name='HP_Books_Main')
    vector_store.add_documents(documents=chunks, embeddings=embeddings)

    print('[INFO] New embeddings and vector store ready for use !')