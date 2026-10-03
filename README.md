<a id="ragdocumentqa"></a>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/readme/banner-dark.svg">
  <img src="docs/assets/readme/banner-light.svg" alt="RAGDocumentQA — Questions over long Korean documents." width="100%">
</picture>

I built this experiment to answer questions using information from long Korean documents. Instead of sending the entire document to the language model, it searches for relevant passages first. This is retrieval-augmented generation, or RAG.

<a id="pipeline"></a>
## What happens before the answer

The program splits each document into overlapping chunks and stores numerical representations of the text in Chroma for searching. It retrieves three matching chunks, adds their neighbors, and puts them back in document order before asking the model to answer.

Including neighboring chunks gives the model more context, but can bring in unrelated text too. The program saves that context alongside each answer so the retrieval step can be examined separately from the model's response.

**Built with:** Python, Chroma, multilingual E5 embeddings, and the OpenAI API.

<a id="results"></a>
## What the saved run shows

The [saved CSV](test_results.csv) contains 100 Ko-LongRAG questions. An automated grader labeled 93 answers correct and seven incorrect. These are the grader's labels, not independently verified accuracy.

Each row includes the question, retrieved passages, generated answer, reference answer, text-similarity score, and grading explanation. In test 84, a similarity score of about 0.991 still accompanies an incorrect grade—a useful reason to inspect the actual answer rather than trust one score.

The [results guide](docs/RESULTS.md) explains the saved failures and the limits of reproducing the run.

> [PLACEHOLDER — one annotated example showing the question, the retrieved passage containing the relevant fact, the generated answer, and the reference answer. Use an actual row from the saved CSV.]

<a id="start"></a>
## Run it

<details>
<summary>Setup, API costs, and saved-file warning</summary>

```bash
git clone https://github.com/loopedlol/RAGDocumentQA.git
cd RAGDocumentQA
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows, use `.venv\Scripts\activate` and copy `.env.example` to `.env` using your shell or file manager.

Set `OPENAI_API_KEY` in your local `.env`. Keep the key private.

**Running the script makes paid API calls and overwrites `test_results.csv`.** Copy the existing CSV somewhere else first to keep the saved run. Questions, retrieved text, and reference answers are sent to the API, so don't use private documents or material you aren't allowed to send.

```bash
python model_tester.py
```

The script downloads the dataset and model resources, then runs the first 100 questions. The number of questions and model settings are set in the source code.

</details>
