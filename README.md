# RAG Document QA

This is a small project I made while learning how **retrieval-augmented generation (RAG)** works.

The basic question I wanted to explore was simple:

> **Can an AI find the right information inside a long document and use it to answer a question correctly?**

A normal language model has to rely mostly on what it already knows. RAG works more like an **open-book test**: first the program searches through source material for useful information, and then it gives that information to the language model before asking it to answer.

For this project, I used long Korean documents from the Ko-LongRAG dataset and built a simple pipeline that searches the documents, generates an answer, and then checks how close that answer is to the provided correct answer.

## What the project does

In simple terms, the process looks like this:

```text
Long document
     ↓
Break it into smaller pieces
     ↓
Turn the pieces into searchable embeddings
     ↓
Find the pieces most related to the question
     ↓
Give those pieces to the language model
     ↓
Generate an answer
     ↓
Compare it with the correct answer
```

The goal was not to build a finished product. I mainly wanted to understand each part of the RAG process instead of only using a library that handled everything for me.

## My first experiment

I tested the system on the first **100 questions** from the `LGAI-EXAONE/Ko-LongRAG` test dataset.

The saved run in this repository produced:

- **100 questions tested**
- **93 answers graded correct**
- **7 answers graded incorrect**

The program also records a semantic similarity score for every answer and saves the passages that were retrieved, which made it much easier for me to look back at failures and figure out what may have gone wrong.

### Example

One test asked which state Al-Fashir is the capital of.

```text
Question: 알파시르는 어느 주의 주도인가요?
AI answer: 북다르푸르 주
Correct answer: 북다르푸르 주
Result: CORRECT
```

The important part is that the answer model was not supposed to know this on its own. The program first searched through the source material and found the passage containing the answer.

## What I learned

### 1. RAG is more than just asking an AI a question

Before this project, it was easy to think of RAG as simply "giving an AI documents." Building the pieces separately helped me understand that there are really multiple problems involved: splitting the document, representing its meaning, searching it, deciding how much context to include, prompting the model, and evaluating the final answer.

### 2. Retrieval quality matters a lot

Even a strong language model cannot give a grounded answer if the useful information never reaches it. This made me realize that improving a RAG system is not only about changing the language model. The search process can be just as important.

I currently retrieve the three most similar chunks and also include the chunks immediately before and after them. I added the neighboring chunks because useful information can be split across chunk boundaries, although this also means more unrelated text can sometimes enter the prompt.

### 3. Similar wording does not always mean a correct answer

One of the most interesting things I noticed was that **semantic similarity is not the same as factual correctness**.

For example, one answer in my saved results was graded incorrect even though its semantic similarity score was about **0.99**. The generated response was worded very similarly to the reference answer, but it changed an important fact.

That is why I ended up using two kinds of evaluation:

- an embedding-based similarity score;
- a second language-model check that decides whether the factual answer is actually correct.

Neither method is perfect, but using both gave me more information than relying on a single number.

### 4. Looking at failures is more useful than only looking at the final score

The incorrect examples were not all the same. Some answers missed an important detail, some returned the wrong number or entity, and some seemed to use related information without finding the exact fact the question asked for.

Because I save the retrieved context in `test_results.csv`, I can inspect whether a mistake came from the search step or from the answering step. That was one of the most useful parts of the experiment for me.

### 5. A simple implementation can be easier to learn from

This version rebuilds a temporary Chroma database for every test question and deletes it afterward. That is definitely not the fastest way to run a large benchmark, but it kept each experiment isolated and made the process easier for me to understand while I was learning.

If I continued developing this into a larger system, caching and reusing embeddings would be one of the first things I would improve.

## How it works technically

The project is split into four main Python files:

```text
data_embedder.py
    Splits the source text into chunks and creates embeddings.

search_engine.py
    Searches Chroma for relevant chunks and asks the language model
    to answer using only the retrieved information.

answer_checker.py
    Compares the generated answer with the reference answer using
    semantic similarity and a separate LLM grader.

model_tester.py
    Runs the experiment over the dataset and saves the results.
```

The current pipeline is:

```text
Ko-LongRAG dataset
        ↓
   data_embedder.py
        ↓
Temporary Chroma database
        ↓
   search_engine.py
        ↓
Generated answer
        ↓
  answer_checker.py
        ↓
   test_results.csv
```

## Current settings

These are the main settings I used for the saved experiment:

| Setting | Value |
| --- | --- |
| Embedding model | `intfloat/multilingual-e5-small` |
| Vector database | Chroma |
| Answer model | `gpt-4.1-mini` |
| Chunk size | 400 characters |
| Chunk overlap | 80 characters |
| Initial retrieved chunks | 3 |
| Extra context | Immediate neighboring chunks |
| Questions in saved run | 100 |

I chose these as reasonable starting values rather than claiming they are optimal. Testing different chunk sizes, retrieval counts, and models would be a useful next experiment.

## Running it yourself

### 1. Clone the repository

```bash
git clone https://github.com/loopedlol/RAG-Document-QA.git
cd RAG-Document-QA
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

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Add an OpenAI API key

Copy the example environment file:

```bash
cp .env.example .env
```

Then replace the placeholder in `.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the experiment

```bash
python model_tester.py
```

The program currently runs examples `0` through `99` and saves the detailed output in `test_results.csv`.

> Running the evaluation uses an OpenAI API and may cost money depending on the model and number of questions tested.

## What is saved in the results?

For every question, `test_results.csv` records:

- the question;
- the answer generated by the system;
- the reference answer from the dataset;
- the semantic similarity score;
- whether the LLM grader marked it correct or incorrect;
- the grader's short explanation;
- the exact retrieved context given to the answer model.

I kept the retrieved context because I wanted the results to be useful for debugging, not just for producing a final accuracy number.

## Limitations

This is still a learning project, and there are several things I would not treat as solved:

- I only tested a fixed set of 100 questions in the saved run.
- The retrieval and chunking settings were chosen manually and have not been systematically optimized.
- The LLM grader can make mistakes, so the 93/100 result should not be treated as perfect ground truth.
- Rebuilding the vector database for every example is slow and repetitive.
- The current script has most experiment settings written directly in the source code.
- I have not yet separated retrieval errors from answer-generation errors with a dedicated retrieval benchmark.

## What I would try next

If I continue this project, I would like to experiment with:

1. comparing several chunk sizes and overlap amounts;
2. changing how many passages are retrieved;
3. measuring whether the correct evidence was actually retrieved before grading the final answer;
4. caching embeddings so repeated experiments run much faster;
5. adding command-line settings instead of editing the Python files for every experiment;
6. comparing different embedding and answer models;
7. creating a small results summary automatically after each run.

## Dataset

The evaluation data comes from the public `LGAI-EXAONE/Ko-LongRAG` dataset, loaded through Hugging Face.

This repository only contains my experiment code and the saved evaluation output. Anyone reusing the dataset should also check its original documentation and license.
