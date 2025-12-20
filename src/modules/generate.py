import os
from dotenv import load_dotenv
from typing import Any, Dict
from src.modules.embed import EmbeddingPipeline
from src.modules.vectorstore import ChromaVectorStore
from sentence_transformers.cross_encoder import CrossEncoder
from src.modules.retrieve import RAGRetriever
from langchain_groq.chat_models import ChatGroq
from src.config_loader import load_config

config = load_config()

class RAGPipeline:

    def __init__(self, llm_model=config['RAG_MODELS']['LLM_MODEL']):
        self.embedding_pipeline = EmbeddingPipeline()
        self.vector_store = ChromaVectorStore()
        self.reranker = CrossEncoder(model_name_or_path=config['RAG_MODELS']['RERANKER_MODEL'], local_files_only=True)
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_pipeline=self.embedding_pipeline,
            reranker=self.reranker
        )
        _ = load_dotenv()
        self.llm = ChatGroq(model=llm_model, temperature=1, api_key=os.getenv("GROQ_API_KEY"))

    def query(self, question: str, top_k: int = 50, rerank: bool = True, top_n: int = 10, min_score: float = 0.0, with_citations: bool = False, summarize: bool = False) -> Dict[str, Any]:
        results = self.retriever.retrieve(question, top_k=top_k, rerank=rerank, top_n=top_n, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'similarity_score': doc['similarity_score'],
                'rerank_score': doc['rerank_score']
            } for doc in results]

            prompt = """Answer the following question by strictly using the provided context.\nQuestion: {question}\n\nContext:\n\n{context}\n\nAnswer:"""
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content

        if with_citations:
            citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if with_citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary
        }