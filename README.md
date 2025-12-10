# HarryPotterRAG
A RAG application for queries based on the Harry Potter Series by J.K. Rowling

## Chunking Strategy:
* Recursive text splitter
* Token-based splitting
* 500 tokens chunk | 100 token overlap

## Retrieval Strategy:
* Stage1 - Bi-encoder retrieval (miniLM)
* Stage2 - Reranking using cross encoder (ms-marco-miniLM)

## Generation:
Mistral 7B used for generating responses