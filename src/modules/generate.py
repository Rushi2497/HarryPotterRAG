import os
from typing import Any, Dict
from src.modules.embed import EmbeddingPipeline
from src.modules.vectorstore import ChromaVectorStore
from voyageai import Client
from src.modules.retrieve import RAGRetriever
from groq import Groq
from src.config_loader import load_config

config = load_config()

class RAGPipeline:

    def __init__(self, llm_model=config['RAG_MODELS']['LLM_MODEL']):
        self.embedding_pipeline = EmbeddingPipeline()
        self.vector_store = ChromaVectorStore()
        self.reranker = Client(api_key=os.getenv('VOYAGEAI_API_KEY'))
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_pipeline=self.embedding_pipeline,
            reranker=self.reranker
        )
        self.llm_model = llm_model
        self.llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._temperature = 1

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
            formatted = prompt.format(context=context, question=question)
            # Use groq client chat completions API directly
            resp = self.llm.chat.completions.create(model=self.llm_model, messages=[{"role": "user", "content": formatted}], temperature=self._temperature)
            answer = resp.choices[0].message.content

        if with_citations:
            citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if with_citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            resp = self.llm.chat.completions.create(model=self.llm_model, messages=[{"role": "user", "content": summary_prompt}], temperature=self._temperature)
            summary = resp.choices[0].message.content

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary
        }