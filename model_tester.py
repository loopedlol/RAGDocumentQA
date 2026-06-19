
import pandas as pd
from data_embedder import Embedder
from search_engine import Engine
from answer_checker import check_similarity, check_llm

import gc
import shutil
import time
from pathlib import Path

k = 3

def safe_delete_folder(folder_path: Path) -> None:
    gc.collect()
    time.sleep(0.2)

    if folder_path.exists():
        shutil.rmtree(folder_path, ignore_errors=True)

def main():
    df = pd.read_parquet("hf://datasets/LGAI-EXAONE/Ko-LongRAG/data/test-00000-of-00001.parquet")
    embedder = Embedder()
    engine = Engine()

    results_log = []

    for i in range(0, 100, 1):
        print("~" * 80)
        print(f"TEST #{i}")

        persist_directory = f"chroma_dbs/test_{i}"
        context = df["context"][i]
        query = df["question"][i]
        answer = df["answer"][i]

        embedder.generate_embeddings(context, persist_directory)
        results = engine.search_chroma(query, persist_directory, k)

        combined_context = ""
        for j, passage in enumerate(results):
            combined_context += f"\n\n[Passage Chunk {j+1}]\n{passage.page_content}"

        response = engine.query_llm(query, combined_context)
        semantic_score = check_similarity(response, answer)
        llm_score = check_llm(query, response, answer)

        print(f"The AI Responsed with: {response}")
        print(f"The correct answer was: {answer}")
        print(f"The semantic similarity score is: {semantic_score}")
        print(f"llm review: {llm_score}")

        before, word, after = llm_score.partition("INCORRECT")
        if word == "":
            before, word, after = llm_score.partition("CORRECT")
        
        results_log.append(
            {
                "test_number": i,
                "question": query,
                "generated_answer": response,
                "actual_answer": answer,
                "semantic_score": semantic_score,
                "llm_grade": word,
                "llm_reason": llm_score.partition("Reason:")[2].strip(),
                "found_context": combined_context
            }
        )

        safe_delete_folder(Path(persist_directory))
    
    results_df = pd.DataFrame(results_log)
    results_df.to_csv("test_results.csv", index=False)


if __name__ == "__main__":
    main()