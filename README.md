# HarryPotterRAG
A RAG application for queries based on the Harry Potter Series by J.K. Rowling

## Chunking Strategy:
* Recursive text splitter
* Token-based splitting
* 500 tokens chunk | 100 token overlap

## Retrieval Strategy:
* Stage1 - Bi-encoder retrieval (voyage-3-large)
* Stage2 - Reranking using cross encoder (voyage rerank-2.5)

## Generation:
openai/gpt-oss-120b used for generating responses