# RAGTester

A compact evaluation harness for experimenting with retrieval-augmented generation (RAG) on long-form, multilingual question-answering data.

RAGTester builds temporary Chroma vector stores from each document, retrieves the most relevant chunks with multilingual E5 embeddings, generates an answer with an OpenAI model, and records both semantic and LLM-based evaluation results.

## What it does

For each example in the evaluation dataset, the project:

1. Splits the source context into article-aware overlapping chunks.
2. Embeds those chunks with `intfloat/multilingual-e5-small`.
3. Stores the embeddings in a temporary Chroma database.
4. Retrieves the most relevant chunk and its neighboring chunks.
5. Generates a grounded answer using only the retrieved context.
6. Compares the generated answer with the reference answer.
7. Saves the evaluation output to `test_results.csv`.
8. Deletes the temporary vector database before continuing.

## Architecture

```text
Ko-LongRAG dataset
        |
        v
 data_embedder.py
 chunking + embeddings
        |
        v
 temporary Chroma store
        |
        v
 search_engine.py
 retrieval + answer generation
        |
        v
 answer_checker.py
 semantic + LLM grading
        |
        v
 model_tester.py
 experiment orchestration + CSV output
```

## Project structure

```text
RAGTester/
├── answer_checker.py   # Semantic similarity and LLM-based grading
├── data_embedder.py    # Article splitting, chunking, and vector-store creation
├── model_tester.py     # Runs the evaluation loop and writes results
├── search_engine.py    # Retrieves context and generates grounded answers
├── requirements.txt    # Python dependencies
└── test_results.csv    # Example evaluation output
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/loopedlol/RAGTester.git
cd RAGTester
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Copy the example environment file:

```bash
cp .env.example .env
```

Then add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run the evaluation

```bash
python model_tester.py
```

The script currently evaluates the first 100 examples from the `LGAI-EXAONE/Ko-LongRAG` test split and writes the results to `test_results.csv`.

## Evaluation output

Each row in `test_results.csv` includes:

- the test number;
- the original question;
- the generated answer;
- the reference answer;
- embedding-based semantic similarity;
- an LLM-generated correctness grade;
- the grader's explanation;
- the retrieved context supplied to the answer model.

## Current design choices

- **Embedding model:** `intfloat/multilingual-e5-small`
- **Vector database:** Chroma
- **Answer model:** `gpt-4.1-mini`
- **Chunk size:** 400 characters
- **Chunk overlap:** 80 characters
- **Retrieved seed chunks:** 3
- **Context expansion:** each retrieved chunk is supplemented with its immediate neighbors

Temporary Chroma databases are created under `chroma_dbs/` and removed after each test case. This keeps stored artifacts small, but it also means embeddings are recomputed every time the experiment runs.

## Limitations

This repository is an experimental benchmark rather than a production RAG service.

- Evaluation currently uses a fixed range of dataset rows.
- Model and retrieval settings are defined directly in the source files.
- LLM grading is useful for qualitative evaluation but is not a perfect ground-truth metric.
- Rebuilding the vector store for every example prioritizes isolation over runtime efficiency.
- API usage may incur costs.

## Possible next steps

- Add command-line options for model, retrieval depth, and test range.
- Record aggregate accuracy and latency statistics.
- Separate configuration from implementation.
- Add automated tests for chunk boundaries and neighbor retrieval.
- Cache embeddings for repeatable large-scale experiments.

## Dataset attribution

The evaluation script loads the public `LGAI-EXAONE/Ko-LongRAG` dataset through Hugging Face. Review the dataset's own documentation and license before redistributing or using it beyond experimentation.
