# RAGDocumentQA

I built this experiment to see how an AI could answer questions using information from long Korean documents. It first finds relevant passages, then gives them to a language model along with the question. This approach is called retrieval-augmented generation, or RAG.

<a id="pipeline"></a>
## How it works

The program splits documents into smaller pieces, searches for pieces related to the question, and uses them to generate an answer. It saves the passages, answer, reference answer, and automated feedback together so I can look into mistakes.

**Built with:** Python, Chroma, multilingual text embeddings, and the OpenAI API.

<a id="results"></a>
## Saved results

The repository includes a [saved run of 100 questions](test_results.csv) from the Ko-LongRAG dataset. You can read it without running the program or making API calls.

The answers were graded by another AI call, not independently checked by a person. Those grades are useful for exploring errors, but they aren't proof of accuracy. The [results guide](docs/RESULTS.md) explains what to look for and points out examples where the scores disagree.

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
