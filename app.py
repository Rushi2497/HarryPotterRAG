# import numpy as np
# from src.data_loader import load_documents
# from src.embedding import EmbeddingPipeline
# from src.vector_store import ChromaVectorStore
from src.search import RAGPipeline

if __name__ == '__main__':
    
    # documents = load_documents(data_dir='./data/pdfs')
    # embedding_pipeline = EmbeddingPipeline(chunk_size=500, chunk_overlap=100)
    # chunks = embedding_pipeline.chunk_documents(documents)
    # embeddings = embedding_pipeline.embed_chunks(chunks)
    # np.savez('./rag_tutorial/data/embeddings_onebook_1000',embeddings)
    # embeddings = np.load('./data/embeddings_minilm_500_tokens.npz')['arr_0']
    # vector_store = ChromaVectorStore()
    # vector_store.client.delete_collection(name='HP_Books')
    # vector_store.add_documents(documents=chunks, embeddings=embeddings)

    user_query = input("Write your query: ")

    answer = RAGPipeline().query(question=user_query,
                                 top_k=50,
                                 rerank=True,
                                 top_n=10,
                                 min_score=-1,
                                 with_citations=False,
                                 summarize=False)
    print(f'Question:',answer['question'])
    print(f'Answer:',answer['answer'])
    # print(f'Summary:',answer['summary'])
    # print(f'Sources:',answer['sources'])