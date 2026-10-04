# schema-smith
> 🚧 Work in Progress

**SchemaSmith** is a Python-based, domain-agnostic data cleaning and validation pipeline designed to process messy CSV datasets.

It combines traditional data-cleaning techniques with semantic column matching using **Sentence Transformers** to identify inconsistent or noisy column names.


## Currently working on:
- Improving the semantic column matcher
- Evaluating matching performance
- Refining the data-quality reporting
- Improving test coverage
  

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
