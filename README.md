# schema-smith
> 🚧 Work in Progress

**SchemaSmith** is a Python-based, domain-agnostic data cleaning and validation pipeline designed to process messy CSV datasets.

It combines traditional data-cleaning techniques with semantic column matching using **Sentence Transformers** to identify inconsistent or noisy column names.


## Currently working on:
- Improving the semantic column matcher
- Evaluating matching performance
- Refining the data-quality reporting
- Improving test coverage

  ## Evaluation Results

SchemaSmith's semantic column matcher was evaluated on a test set of 100 column names. The test set covers six types of input: exact matches, casing and spacing variations, abbreviations, noisy names, typos, and synonyms.

The matcher uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate embeddings and cosine similarity to rank canonical column names. Predictions with a confidence score below the configured threshold of `0.75` are flagged for review.

### Overall Results

| Metric             |       Result |
| ------------------ | -----------: |
| Test cases         |          100 |
| Top-1 accuracy     | 90% (90/100) |
| Top-3 accuracy     | 96% (96/100) |
| Silent errors      |       0 (0%) |
| Flagged for review |     10 (10%) |

**Top-1 accuracy** measures how often the first prediction is correct. **Top-3 accuracy** measures how often the correct canonical name appears among the three highest-ranked predictions.

### Results by Category

| Category           | Correct | Accuracy |
| ------------------ | ------: | -------: |
| Exact              |   10/10 |     100% |
| Casing and spacing |   15/15 |     100% |
| Abbreviation       |   19/20 |      95% |
| Noise              |   19/20 |      95% |
| Synonym            |   19/20 |      95% |
| Typo               |    8/15 |    53.3% |

### Improvement in Noise Handling

Noise-category accuracy improved from **75% (15/20) to 95% (19/20)** during development. This represents four additional correct predictions in the 20 noise test cases.

The remaining error in this category is `S@lary`, which is incorrectly matched to `department` instead of `salary`. This indicates that handling special characters and noisy column names still needs further refinement.

### Error Handling

In the latest evaluation, all 10 incorrect Top-1 predictions were flagged for review because their similarity scores were below the `0.75` threshold. No incorrect Top-1 predictions exceeded the threshold in this test run.

### Limitations

These results apply to the current 100-case test set and do not guarantee the same performance on unseen datasets. Typo handling remains the main weakness, with an accuracy of 53.3%. Further evaluation should focus on spelling variations, ambiguous abbreviations, and noisy inputs.

  

## Problem

Messy CSV datasets often contain inconsistent column names, missing values, incorrect data types, duplicate records, and anomalous values. These issues can make automated data processing unreliable and require significant manual cleanup.

## Solution

SchemaSmith automates the cleaning and validation process through a multi-stage pipeline. It analyzes the input dataset, cleans and normalizes its structure, detects data quality issues, and uses semantic similarity to match noisy column names to canonical fields.

The pipeline can also flag uncertain matches for human review instead of making low-confidence decisions automatically.



## Key Features

- **CSV Data Inspection** — Analyzes dataset structure, columns, missing values, and data types.
- **Column Name Cleaning** — Normalizes inconsistent and noisy column names.
- **Type Inference & Conversion** — Detects likely data types and converts values accordingly.
- **Missing Value Handling** — Handles missing values based on column type.
- **Duplicate Detection** — Identifies and removes duplicate records.
- **Anomaly Detection** — Uses Isolation Forest to flag potentially anomalous records.
- **Semantic Column Matching** — Uses Sentence Transformers and cosine similarity to match noisy column names with canonical fields.
- **Confidence-Based Review** — Flags uncertain semantic matches for human review.
- **Evaluation & Reporting** — Includes matcher evaluation and generates an HTML data-quality report.

  ## Pipeline

SchemaSmith processes a messy CSV dataset through a sequence of cleaning, validation, and semantic matching stages.


Input CSV
   │
   ▼
Load & Inspect
   │
   ▼
Clean Column Names
   │
   ▼
Normalize Missing Values
   │
   ▼
Infer & Convert Data Types
   │
   ▼
Detect & Remove Duplicates
   │
   ▼
Handle Missing Values
   │
   ▼
Detect Anomalies
   │
   ▼
Generate Embeddings
   │
   ▼
Semantic Column Matching
   │
   ▼
Confidence Check
   │
   ├── High Confidence ──► Safe Renaming
   │
   └── Low Confidence ───► Human Review
                              │
                              ▼
                       Cleaned Dataset
                              │
                              ▼
                         HTML Report


## Technologies

- **Python**
- **pandas & NumPy** — data processing and cleaning
- **Scikit-learn** — anomaly detection with Isolation Forest
- **Sentence Transformers** — semantic embeddings
- **Jinja2** — HTML report generation
- **pytest** — testing and evaluation
- **uv** — Python project and dependency management

## Project Structure


schema-smith/
├── src/                         # Python package
├── main.py                      # Main pipeline entry point
├── evaluator.py                 # Semantic matcher evaluation
├── canonical_column_names.py    # Canonical field definitions
├── dirty_data/                  # Sample messy CSV datasets
├── templates/                   # HTML report template
├── test_data_for_matcher.json   # Test cases for semantic matching
├── pyproject.toml               # Project configuration and dependencies
├── uv.lock                      # Locked dependencies
└── README.md

## How to Run

### 1. Clone the repository


git clone https://github.com/mohamed-amine-zirari/schema-smith.git
cd schema-smith
### 2. Install dependencies

Using uv:

uv sync
### 3. Run the pipeline

Place a CSV file inside dirty_data/, then run:

uv run main.py --file your_file.csv

### Optional parameters:

uv run main.py --file your_file.csv --threshold 0.75 --contamination 0.05
--threshold controls the confidence threshold for semantic column matching.
--contamination controls the expected proportion of anomalies detected by Isolation Forest.
Output

### The pipeline generates:

A cleaned CSV file
A review file for low-confidence column matches
An HTML data-quality report
