
import pandas as pd
from data_embedder import Embedder
from search_engine import Engine
from answer_checker import check_similarity, check_llm

k = 1

def main():
    df = pd.read_parquet("hf://datasets/LGAI-EXAONE/Ko-LongRAG/data/test-00000-of-00001.parquet")
    embedder = Embedder()
    engine = Engine()

    results_log = []

    for i in range(0, 10, 1):
        print("~" * 80)
        print(f"TEST #{i}")

        persist_directory = f"chroma_dbs/test_{i}"
        context = df["context"][i]
        query = df["question"][i]
        answer = df["answer"][i]

        #embedder.generate_embeddings(context, persist_directory)
        results = engine.search_chroma(query, persist_directory, k)
        response = engine.query_llm(query, results[0].page_content)
        semantic_score = check_similarity(response, answer)
        llm_score = check_llm(response, answer)

        print(f"The AI Responsed with: {response}")
        print(f"The correct answer was: {answer}")
        print(f"The semantic similarity score is: {semantic_score}")
        print(f"llm review: {llm_score}")

        results_log.append(
            {
                "test_number": i,
                "question": query,
                "generated_answer": response,
                "actual_answer": answer,
                "semantic_score": semantic_score,
                "llm_review": llm_score
            }
        )
    
    results_df = pd.DataFrame(results_log)
    results_df.to_csv("test_results.csv", index=False)


if __name__ == "__main__":
    main()