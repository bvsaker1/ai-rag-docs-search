Project Goal
Build a modular Retrieval-Augmented Generation (RAG) system that answers questions over PDF documents. Your system must support multiple chunking strategies, embedding models, retrieval methods, and rerankers, making it easy to experiment with different configurations and rigorously evaluate their performance.
Core Challenge: RAG systems have many moving parts (chunking, embeddings, retrieval, reranking, generation). There's no one-size-fits-all configuration. Your task is to build a system where you can swap components, run experiments, and measure what actually works best for your documents.

The Problem Context
Researchers and students deal with hundreds of academic papers across conferences and journals. Finding specific results, methods, or conclusions buried inside dense PDFs is painfully slow. A simple keyword search misses semantic meaning. Reading entire papers end-to-end is impractical when you're surveying a field.
Your solution: Build an intelligent QA system that:
Ingests PDF documents and breaks them into searchable chunks
Retrieves the most relevant chunks for any question using semantic search
Generates accurate answers with proper citations to source material
Experiments with different configurations to optimize performance
Evaluates retrieval quality using standard IR metrics (Precision, Recall, MRR, NDCG)
Serves answers through both CLI and web interfaces

System Architecture Overview
Your RAG pipeline should follow this flow:
INGESTION PHASE:
PDF → Load → Preprocess → Chunk → Embed → Index → Save

QUERY PHASE:
Question → Retrieve top-K → Rerank (optional) → Generate Answer → Extract Citations

EVALUATION PHASE:
Test Queries → Run Experiments → Calculate Metrics → Compare Configurations
Core Components (All Must Be Swappable):
Document Processing 
PDF Loader: Extract text and metadata
Preprocessor: Clean and normalize text
Chunker: Split into semantic units (multiple strategies)
Embedding & Storage 
Embedder: Convert text to vectors (multiple models)
Vector Store: Index and search (FAISS/LanceDB/Turbopuffer)
Retrieval 
Dense Retriever: Embedding similarity search
BM25 Retriever: Keyword-based sparse retrieval
Hybrid Retriever: Weighted combination of dense + BM25
Reranking (Optional) 
Cohere Reranker: API-based reranking
Cross-Encoder: Local model reranking
Generation 
LLM Client: OpenAI or LiteLLM (multi-provider)
Prompt Templates: System prompts and formatting
Citation Extractor: Parse source references from answers
Evaluation 
Metrics Calculator: Precision, Recall, MRR, NDCG
LLM Judge: Quality assessment of generated answers
Experiment Tracker: Save and compare results

DO NOT BUILD anything before I give permission.  You are to help me build this.  I will give you sections to build one at a time. Always ask clarifying questions for instructions that are unclear or ill advised.

FIRST STEP (Ingestion Phase):
  - Build a pipeline that will include the 3 phases but only implement phase 1 and the pipeline with placeholders.
  - Make each main component an abstract component so that different configurations of chunking, embedding and retrieval can be combined.
  - Make separate parser, chunking, embedding, indexing and saving files for the ingestion phase.
  - Use the following chunking methods with 3 .env chunking methods available 
    - Fixed token based chunking with tiktoken where the default chunk size is 512 with 75 overlap but configurable in .env.
    - Sentence based chunking with 1 to 2 sentence overlap
    - Semantic chunking
  - Use the following 2 embedding models
    - all-MiniLM-L6-v2
    - all-mpnet-base-v2
  - Use BM25, Semantic with FAIS, and Hybrid retrieval methods.
  - Use Instructor with the following models to catch ill formatted data
    -Document:
      - id: UUID
      - content: str
      - metadata: {source, title, author, page_count, ...}

    - Chunk:
      - id: UUID
      - content: str
      - metadata: {document_id, source, page_number, start_char, end_char, chunk_index}
      - embedding: list[float] (optional)

    - RetrievalResult:
      - chunk: Chunk
      - score: float
      - retriever_type: enum (dense/bm25/hybrid)

    - QAResponse:
      - query: str
      - answer: str
      - citations: list[Citation]
      - chunks_used: list[Chunk]
      - confidence: float

    - Citation:
      - chunk_id: UUID
      - source: str (filename)
      - page_number: int
      - text_snippet: str
      - relevance_score: float
  - Use the same parsing, chunking, embedding, metric and logging file naming conventions and formats as mini project 3 in ../mini-project3
  - Generate tests for each component as well as integration tests
  - Parser Instructions
    - needs a parameter in .env to limit how many files are run with the default set to 50
    - Files are pulled from data/papers

SECOND_STEP (Query Phase)
  - Select 12 diverse queries from queries.json where 8 are abstractive and 4 are extractive.  Combine these with approximately 65% source of text, 25% source of text-image with 5% text-table.  Make the number and percents configurable in the .env file.
  - Embed the generated queries for each of the 15 configurations of chunking, embedding and retrieval methods.
  - Retrieve the recall@5 or top 5 matches.
  - Return the a generated top answer with citations that match the citations model.
  - Make a queries/query-set.json directory and file that copies each selected question from queries.json.
  - Make an answers/answer-<chunk>-<embed>-<retrieval>.json directory and file that contains the generated answer above with a name that describes the configuration. 

THIRD_STEP (Evaluation Phase)
  - Generate the metrics for recall@5, precision@5, MRR and NDCG@5
  - Run answers through an LLM as Judge (ie deepseek) and rate on a scale of 1-5 the following in a table:
    - Query ID
    - Relevance
    - Accuracy
    - Completeness
    - Citation Quality
    - Avg (of the 4 scores)
  - Using matplotlib, generate the following .png graphs in the Visualizations directory
    - Retrieval Metrics Comparison Heatmap: A matrix showing Recall@5, Precision@5, MRR, and NDCG@5 across all tested configurations. Use a sequential color scale (e.g., YlGnBu). Each cell must display the numeric value.
    - Configuration Dimension Impact: A grouped bar chart showing how each dimension (chunking strategy, embedding model, retrieval method) affects NDCG@5 when averaged across other dimensions. This reveals which dimension matters most.
    - Before/After Improvement Chart: A side-by-side bar chart comparing retrieval metrics before and after your configuration improvements. Include the delta value above each bar pair.
    - Generation Quality Radar Chart: A radar (spider) chart showing average LLM-as-Judge scores across Relevance, Accuracy, Completeness, and Citation Quality for your best configuration.
    - Query Latency Distribution: A histogram or box plot showing query latency across configurations. Helps identify configurations that are too slow for practical use.
    - Hybrid Fusion Weight Sweep (if applicable): A line chart showing NDCG@5 as a function of the fusion weight alpha, to justify your chosen alpha value.
    - Quality Standard: All visualizations must include clear titles, labeled axes, appropriate color scales, and legible font sizes. A chart that cannot be read at a glance is not a finished chart.

FOURTH_STEP (UI QUERY Phase)
  - Streamlit app with:
    - File upload for PDFs
    - Configuration sidebar (chunking, embedding, retrieval, top-k)
    - Question input
    - Answer display with citations
    - Source chunk viewer

<!-->
📂 Dataset: Open RAG Benchmark (arXiv Papers)
This project uses the Open RAG Benchmark by Vectara, a real-world dataset of 1,000 arXiv research papers with 3,045 question-answer pairs and ground truth relevance labels.
Source: License: CC-BY-NC-4.0
Dataset Structure
pdf/arxiv/
├── pdf_urls.json       # URLs to download the 1,000 arXiv PDFs
├── queries.json        # 3,045 questions (extractive + abstractive)
├── answers.json        # Ground truth answers for each query
├── qrels.json          # Relevance labels: maps each query → relevant doc + section
└── corpus/             # Pre-parsed paper text (fallback if PDFs unavailable)
    ├── {PAPER_ID}.json # Per-paper: title, abstract, authors, sections (text + tables + images)
    └── ...
Key Numbers
Component
Count
PDF documents
1,000 (400 relevant + 600 hard negatives)
QA pairs
3,045 (1,793 abstractive + 1,252 extractive)
Query types by source
1,914 text-only, 763 text-image, 148 text-table, 220 mixed
Ground Truth for Evaluation
The qrels.json file provides relevance labels in BEIR format, the standard used by major IR benchmarks. Each query maps to a specific paper ID and section index. This is exactly what you need to calculate Precision, Recall, MRR, and NDCG without manual labeling.
⚠️ Important: Downloading the PDFs
The dataset does not bundle the PDF files directly. Instead, pdf_urls.json contains arXiv URLs for each paper. You must download them yourself before running the ingestion pipeline.
Quick download script:
import json, requests, time
from pathlib import Path
﻿
pdf_dir = Path("data/papers")
pdf_dir.mkdir(parents=True, exist_ok=True)
﻿
with open("data/pdf_urls.json") as f:
    pdf_urls = json.load(f)
﻿
for paper_id, url in pdf_urls.items():
    dest = pdf_dir / f"{paper_id}.pdf"
    if dest.exists():
        continue
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    time.sleep(1)  # be polite to arXiv servers
    print(f"Downloaded {paper_id}")
Tips:
arXiv rate-limits aggressive downloads, so add a 1-second delay between requests
Start with a small subset (e.g., 50 papers) for development, scale up for final experiments
The corpus/ folder contains pre-parsed text as a fallback, but use the actual PDFs for your ingestion pipeline since that's the whole point of the project
﻿
📊 Success Metrics
Your RAG system will be evaluated on these quantitative benchmarks:
1. Retrieval Performance
Run experiments comparing different configurations and measure:
Metric
Description
Target
Recall@5
% of relevant chunks in top-5 results
>0.80
Precision@5
% of top-5 results that are relevant
>0.60
MRR
Mean Reciprocal Rank of first relevant result
>0.70
NDCG@5
Normalized Discounted Cumulative Gain
>0.75
2. Experiment Coverage
Test at least these configuration dimensions:
Chunking Strategies (minimum 3):
Fixed-size (token or character-based)
Recursive (multi-separator splitting)
Semantic (embedding-based breakpoints)
Sliding window (overlapping chunks)
Embedding Models (minimum 2):
Fast model (e.g., all-MiniLM-L6-v2, 384d)
Quality model (e.g., all-mpnet-base-v2, 768d)
Retrieval Methods (minimum 2):
Dense (embedding similarity)
BM25 (sparse keyword matching)
Hybrid (dense + BM25 fusion)
Reranking (optional but recommended):
Cohere Rerank API
Cross-encoder model
3. Generation Quality (LLM-as-Judge)
For generated answers, evaluate:
Relevance: Answer addresses the question (1-5 scale)
Accuracy: Information is factually correct (1-5 scale)
Completeness: Answer is thorough (1-5 scale)
Citation Quality: Proper source attribution (1-5 scale)
Target: Average score >4.0 across all criteria
4. System Functionality
Successfully ingest PDFs and create searchable indices
CLI tools work for ingestion, querying, and evaluation
Web UI (Streamlit) provides interactive QA experience
Experiment results saved with full configuration tracking
All components swappable through configuration files
﻿
🛠 Technical Requirements
Required Technology Stack
Python 3.10+ - Core language
PyMuPDF - PDF text extraction
SentenceTransformers - Embedding generation
FAISS - Vector similarity search (or LanceDB/Turbopuffer)
rank-bm25 - Sparse retrieval baseline
LiteLLM or OpenAI - LLM for answer generation
Pydantic - Data models and validation
Instructor - Structured LLM outputs for judge scoring and answer generation
Matplotlib / Seaborn - All experiment comparison charts and heatmaps
Streamlit - Web interface
Optional Enhancements
Plotly - Interactive visualizations in notebooks or dashboards
Cohere - Reranking API
Braintrust - Experiment tracking and evaluation across runs
Langfuse - Observability and tracing for LLM calls
pytest - Comprehensive testing
Pandas - Data manipulation and aggregation of experiment results
﻿
Data Models & Interfaces
Core Types (Use Pydantic):
Document:
  - id: UUID
  - content: str
  - metadata: {source, title, author, page_count, ...}
﻿
Chunk:
  - id: UUID
  - content: str
  - metadata: {document_id, source, page_number, start_char, end_char, chunk_index}
  - embedding: list[float] (optional)
﻿
RetrievalResult:
  - chunk: Chunk
  - score: float
  - retriever_type: enum (dense/bm25/hybrid)
﻿
QAResponse:
  - query: str
  - answer: str
  - citations: list[Citation]
  - chunks_used: list[Chunk]
  - confidence: float (optional)
﻿
Citation:
  - chunk_id: UUID
  - source: str (filename)
  - page_number: int (optional)
  - text_snippet: str
  - relevance_score: float (optional)
Abstract Base Classes:
All components must implement abstract interfaces:
BaseChunker: chunk(document) → list[Chunk]
BaseEmbedder: embed(texts) → np.ndarray
BaseVectorStore: add(), search(), save(), load()
BaseRetriever: retrieve(query, top_k) → list[RetrievalResult]
BaseReranker: rerank(query, results, top_k) → list[RerankResult]
BaseLLM: generate(prompt, system_prompt, temperature) → str
Why this matters: Swappable components let you experiment without rewriting code.
﻿
Question-answering tasks
Trade-off: Higher dimensions = better quality but slower search and more storage.
Your task: Test at least 2 models and measure the quality vs. speed trade-off.
Challenge 3: Hybrid Retrieval Score Fusion
Combining dense and BM25 scores requires normalization:
Problem: Dense scores are cosine similarity (0-1), BM25 scores are unbounded
﻿
Solution:
1. Normalize BM25 scores to [0, 1] range using min-max scaling
2. Combine: final_score = α * dense_score + (1-α) * bm25_score
3. Experiment with α ∈ {0.3, 0.5, 0.7} to find optimal weight
Why it matters: Without normalization, one retriever dominates and hybrid becomes useless.
Challenge 4: Evaluation with Ground Truth
To calculate Precision/Recall/MRR, you need ground truth labels:
Option 1: Use Provided Ground Truth (qrels.json) [Primary]
The dataset includes qrels.json mapping each query to its relevant document and section
Load these mappings and match them against your retrieved chunks
You'll need to map chunk IDs back to source document + section for comparison
Store as {query, relevant_chunk_ids} pairs derived from qrels
Challenge 5: Citation Extraction
Your LLM must cite sources. Two approaches:
Approach 1: Structured Format
Prompt: "Cite sources using [Source: filename, Page: X] format"
Parse: Regex to extract citations from answer
Approach 2: Index References
Prompt: "Context chunks are numbered [1], [2], [3]. Cite using [N]."
Parse: Extract [N] references, map back to chunks
Why it matters: Without citations, users can't verify answers or find source material.
﻿
📦 Deliverables
Your completed RAG system must produce:
1. Functional Components
PDF ingestion pipeline (CLI script)
Vector index creation and persistence
Query pipeline with retrieval + generation
Citation extraction from answers
Web UI for interactive QA
2. Experiment Results
Run experiments comparing configurations:
Minimum experiment grid:
3 chunking strategies × 2 embedding models × 2 retrieval methods = 12 configurations
Test on at least 10 diverse queries
Save results with full configuration metadata
Output format (JSON):
{
  "experiment_id": "exp_20260305_abc123",
  "config": {
    "chunking_strategy": "recursive",
    "chunk_size": 512,
    "embedding_model": "all-mpnet-base-v2",
    "retriever_type": "hybrid",
    "top_k": 5
  },
  "metrics": {
    "recall@5": 0.85,
    "precision@5": 0.72,
    "mrr": 0.78,
    "ndcg@5": 0.81
  },
  "query_results": [...]
}
3. Comparison Analysis
Generate comparison showing:
Which chunking strategy performed best?
Did hybrid retrieval beat dense-only?
Was reranking worth the latency cost?
Which embedding model gave best quality/speed trade-off?
4. CLI Tools
# Ingest PDFs
python scripts/ingest.py data/papers/ -o data/indices/arxiv_index
﻿
# Interactive QA
python scripts/serve.py -i data/indices/arxiv_index
﻿
# Run experiments (uses queries.json and qrels.json from the dataset)
python scripts/evaluate.py config/experiments/baseline.yaml \\\\\\\\
  -p data/papers/ -q data/queries.json -o experiments/results/
5. Web Interface
Streamlit app with:
File upload for PDFs
Configuration sidebar (chunking, embedding, retrieval, top-k)
Question input
Answer display with citations
Source chunk viewer
﻿
Visualization Requirements
All visualizations must be generated using Matplotlib, Seaborn, or Plotly. Every plot must be saved as a PNG file and included in your documentation.
Your experiment analysis must include these visualizations:
Retrieval Metrics Comparison Heatmap: A matrix showing Recall@5, Precision@5, MRR, and NDCG@5 across all tested configurations. Use a sequential color scale (e.g., YlGnBu). Each cell must display the numeric value.
Configuration Dimension Impact: A grouped bar chart showing how each dimension (chunking strategy, embedding model, retrieval method) affects NDCG@5 when averaged across other dimensions. This reveals which dimension matters most.
Before/After Improvement Chart: A side-by-side bar chart comparing retrieval metrics before and after your configuration improvements. Include the delta value above each bar pair.
Generation Quality Radar Chart: A radar (spider) chart showing average LLM-as-Judge scores across Relevance, Accuracy, Completeness, and Citation Quality for your best configuration.
Query Latency Distribution: A histogram or box plot showing query latency across configurations. Helps identify configurations that are too slow for practical use.
Hybrid Fusion Weight Sweep (if applicable): A line chart showing NDCG@5 as a function of the fusion weight alpha, to justify your chosen alpha value.
Quality Standard: All visualizations must include clear titles, labeled axes, appropriate color scales, and legible font sizes. A chart that cannot be read at a glance is not a finished chart.
﻿
🔄 Iteration Logs and Trace Requirements
Every configuration change must be traceable to specific experiment data. Do not make blind changes.
Iteration Log Format
For each improvement iteration, document the following:
Iteration: 2
Change: Switched from fixed-size (512 tokens) to recursive chunking (512 tokens, 100 overlap)
Reason: Fixed chunking split mid-paragraph in 34% of chunks, causing Recall@5 = 0.72
Metric Before: Recall@5 = 0.72, NDCG@5 = 0.68
Metric After: Recall@5 = 0.83, NDCG@5 = 0.77
Delta: Recall@5 +0.11, NDCG@5 +0.09
What Must Be Logged
Every chunking strategy, embedding model, or retrieval method change
The specific metric or observation that motivated the change
Before and after metric values with the delta
If a change made things worse, log that too and explain what you reverted or tried next
Trace Improvements
Every configuration decision in your final system must be traceable to a specific experiment result. Your final report should include a table like:
﻿
🎯 Evaluation Approach
Your system will be evaluated by running it fresh and examining the outputs. The evaluator will follow these steps:
Step 1. Run the baseline experiment grid
Run at least 12 configurations (3 chunking x 2 embedding x 2 retrieval) on a minimum of 10 diverse queries from queries.json. Record all retrieval metrics per configuration.
If Recall@5 < 0.80 across all configurations, your chunking or embedding strategy needs improvement. Try adjusting chunk size, overlap, or switching to a higher-quality embedding model.
Step 2. Evaluate generation quality with LLM-as-Judge
For each answer generated by your best retrieval configuration, score on 4 dimensions using a 1-5 scale:
Query ID
Relevance
Accuracy
Completeness
Citation Quality
Avg
Target: Average score > 4.0 across all criteria.
If Citation Quality is consistently low (< 3.5), your prompt template likely does not instruct the LLM to cite sources clearly. Revise the system prompt to include explicit citation format instructions and re-run.
If Completeness is low, check whether your top-K is too small (relevant context not retrieved) or your generation prompt is too restrictive.
Step 3. Identify the best configuration
Rank all configurations by NDCG@5 (primary) and MRR (secondary). The winning configuration must meet all four retrieval targets:
Metric
Target
If below target
Recall@5
> 0.80
Increase chunk overlap, try semantic chunking, or add hybrid retrieval
Precision@5
> 0.60
Reduce chunk size, add reranking, or filter low-confidence results
MRR
> 0.70
Improve embedding model quality or add reranking stage
NDCG@5
> 0.75
Combine improvements above; check if BM25 fusion weight needs tuning
Step 4. Run improvement iteration
After identifying weaknesses in Step 3, make targeted changes to your configuration. Log every change:
Change: Switched from fixed-512 to recursive-512 chunking
Reason: Recall@5 was 0.72 with fixed chunking; recursive preserves section boundaries
Result: Recall@5 improved to 0.83 (+0.11)
Re-run the full experiment grid and produce a before/after comparison table.
Step 5. Measure system performance
Metric
Value
Ingestion time (50 papers)
e.g., 4 min 32 sec
Avg query latency
e.g., 1.8 sec
Index size on disk
e.g., 245 MB
Peak memory usage
e.g., 1.2 GB
If query latency > 5 seconds, consider using a smaller embedding model or reducing top-K.
Step 6. Verify experiment reproducibility
Run the same configuration twice and confirm metrics are within 5% of each other. If results vary wildly, check for non-determinism in your retrieval or generation pipeline (e.g., LLM temperature > 0 for the judge).
🚀 Getting Started Hints
Recommended Development Order:
Download the dataset - Fetch PDFs from arXiv using the provided pdf_urls.json (start with ~50 papers)
Start with document processing - Get PDF loading and chunking working
Build vector store - Implement FAISS indexing and search
Add dense retrieval - Embed queries and retrieve similar chunks
Integrate LLM - Generate answers from retrieved context
Add citations - Parse source references from answers
Implement BM25 - Add sparse retrieval baseline
Build hybrid retrieval - Combine dense + BM25 with score fusion
Add evaluation - Implement metrics and experiment tracking (use provided qrels.json for ground truth)
Create CLI tools - Make system usable from command line
Build web UI - Add Streamlit interface for demos
Common Pitfalls to Avoid:
Don't skip preprocessing - Raw PDF text has formatting artifacts that hurt retrieval
Don't forget normalization - Embeddings should be L2-normalized for cosine similarity
Don't hardcode chunk size - Make it configurable for experimentation
Don't ignore edge cases - Handle empty documents, very short chunks, no results found
Don't skip persistence - Save indices to disk, don't rebuild every time
Don't forget error handling - LLM APIs fail, PDFs are corrupted, handle gracefully
Testing Strategy:
Unit tests: Each component (chunker, embedder, retriever) in isolation
Integration tests: Full pipeline from PDF to answer
Metric tests: Verify Precision/Recall calculations with known inputs
Edge case tests: Empty docs, malformed PDFs, very long queries
﻿
✅ Final Success Criteria
Before submitting, verify that your implementation meets all of the following:
Pipeline and Components
[ ] System ingests PDFs and creates searchable FAISS (or LanceDB/Turbopuffer) indices without crashing
[ ] At least 3 chunking strategies implemented and working (fixed-size, recursive, semantic or sliding window)
[ ] At least 2 embedding models tested (e.g., MiniLM-L6 and mpnet-base)
[ ] Dense, BM25, and Hybrid retrieval all functional and swappable via configuration
[ ] LLM generates answers with proper citations to source chunks
[ ] All components implement abstract base classes and are swappable without code changes
[ ] Pipeline handles malformed PDFs and LLM API errors without crashing
Experiments and Metrics
[ ] Experiments run comparing 12+ configurations (3 chunking x 2 embedding x 2 retrieval minimum)
[ ] Each experiment tested on at least 10 diverse queries from the dataset
[ ] Retrieval metrics (Precision@5, Recall@5, MRR, NDCG@5) calculated correctly using ground truth from qrels.json
[ ] Best configuration meets retrieval targets: Recall@5 > 0.80, Precision@5 > 0.60, MRR > 0.70, NDCG@5 > 0.75
[ ] LLM-as-Judge scores generated answers on Relevance, Accuracy, Completeness, and Citation Quality
[ ] Average generation quality score > 4.0 across all criteria
[ ] All experiment results saved as JSON with full configuration metadata
Visualizations
[ ] Retrieval metrics comparison heatmap generated using Matplotlib, Seaborn, or Plotly
[ ] Configuration dimension impact chart showing which dimension matters most
[ ] Before/after improvement comparison chart with delta values
[ ] Generation quality radar chart for best configuration
[ ] All visualizations saved as PNG files with clear titles, labeled axes, and legible fonts
Iteration Logs and Traceability
[ ] Every configuration change documented with reason, before/after metrics, and delta
[ ] Final configuration decisions traceable to specific experiment results
[ ] Improvement iterations logged in structured format (change, reason, metric before, metric after, delta)
[ ] Results show clear winner among configurations with data-driven explanation
Interfaces
[ ] CLI tools work for ingestion, querying, and evaluation
[ ] Streamlit web UI provides interactive QA experience with configuration sidebar
[ ] Web UI displays answers with citations and source chunk viewer
Documentation
[ ] README explains how to run the full pipeline end-to-end (ingestion, querying, experiments)
[ ] Experiment comparison report included with quantitative analysis
[ ] You can explain why certain configurations outperformed others based on the data -->