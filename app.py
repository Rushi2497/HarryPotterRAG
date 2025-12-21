from dotenv import load_dotenv
from src.modules.generate import RAGPipeline

load_dotenv()

if __name__ == '__main__':

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