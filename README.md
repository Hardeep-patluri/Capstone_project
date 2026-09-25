# Capstone Project

A complete three-module data and AI project covering **data engineering, analytics & machine learning, and a RAG-based support assistant**.

The project is organized into three independent modules, each with its own implementation, dependencies, documentation, and end-to-end execution workflow.

---

## Project Overview

| Module | Focus | Main Technologies |
|---|---|---|
| **Data Pipeline** | Web scraping, cleaning, database storage, SQL analysis | Python, Requests, BeautifulSoup, Pandas, SQLite |
| **Analytics** | EDA, data storytelling, classification, regression, model tuning | Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SMOTE, Joblib |
| **Support Assistant** | RAG, semantic retrieval, workflow orchestration, API service | Sentence Transformers, ChromaDB, LangGraph, Pydantic, FastAPI |

---

# Repository Structure

```text
Capstone_project/
│
├── README.md
│
├── data_pipeline/
│   ├── module_1_datapipeline.ipynb
│   ├── books.db
│   ├── books_three_categories.csv
│   ├── sql_query_results.txt
│   ├── requirements.txt
│   └── README.md
│
├── analytics/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── titanic.csv
│   ├── titanic_survival_pipeline.joblib
│   ├── charts/
│   ├── requirements.txt
│   └── README.md
│
└── support_assistant/
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    ├── ingest.py
    ├── prompts.py
    ├── graph_app.py
    ├── schemas.py
    ├── main.py
    ├── requirements.txt
    ├── Dockerfile
    └── README.md
Technology Stack
Data Engineering
Python
Requests
BeautifulSoup
Pandas
SQLite
SQL
Data Analytics & Machine Learning
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Imbalanced-learn
Joblib
GenAI / RAG
Sentence Transformers
all-MiniLM-L6-v2
ChromaDB
LangGraph
Pydantic
FastAPI
Uvicorn
Docker
Module 1 — Data Pipeline

The Data Pipeline module implements a complete pipeline from web data extraction to database analysis.

Workflow
books.toscrape.com
        ↓
Web Scraping
        ↓
Data Cleaning
        ↓
Type Conversion
        ↓
GBP → INR Conversion
        ↓
SQLite Database
        ↓
SQL Queries
        ↓
Pandas Validation
What it does
Scrapes books from three categories:
Travel
Mystery
Historical Fiction
Extracts book title, price, rating, availability, and category.
Cleans and converts scraped values into appropriate data types.

Converts GBP prices to INR using the project-defined fixed rate:

1 GBP = 105.50 INR

Stores the data in a normalized SQLite database.
Executes SQL queries covering filtering, sorting, limiting, distinct values,
set/range conditions, and joins.
Cross-checks SQL results using pandas.
Run
cd data_pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Open:

module_1_datapipeline.ipynb

Run all cells from top to bottom.

Outputs
books_three_categories.csv
books.db
sql_query_results.txt

Detailed documentation:

Data Pipeline README

Module 2 — Analytics

The Analytics module performs exploratory data analysis, data storytelling,
classification, regression, class-imbalance analysis, and model tuning using
the Titanic dataset.

Workflow
Titanic Dataset
      ↓
Data Profiling
      ↓
Missing-Value Analysis
      ↓
Data Cleaning
      ↓
EDA & Data Story
      ↓
Train/Test Split
      ↓
Preprocessing Pipeline
      ↓
Classification Models
      ↓
Imbalance Handling
      ↓
Random Forest Tuning
      ↓
Regression Analysis
      ↓
Saved ML Pipeline
Classification Models

The project trains and evaluates:

Logistic Regression
Decision Tree
Random Forest

Evaluation includes:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
ROC Curve

The project also examines class imbalance using baseline training,
class-weight balancing, and SMOTE.

Model Tuning

A Random Forest is tuned using GridSearchCV.

The final fitted preprocessing and model pipeline is saved as:

titanic_survival_pipeline.joblib
Regression

A regression task is also performed using fare as the target and evaluated
using:

MAE
RMSE
R²
Adjusted R²
Run
cd analytics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run the notebooks in order:

01_eda.ipynb
02_modeling.ipynb
Outputs
titanic.csv
titanic_survival_pipeline.joblib
charts/

Detailed documentation:

Analytics README

Module 3 — Support Assistant

The Support Assistant is a RAG-based policy question-answering service
built around a fixed eight-document Zepto policy corpus.

Architecture
8 Policy Documents
        ↓
Ingestion & Chunking
        ↓
Local Embeddings
        ↓
ChromaDB
        ↓
Query Embedding
        ↓
Top-3 Retrieval
        ↓
LangGraph Routing
        ↓
Answer Generation
        ↓
Pydantic Validation
        ↓
FastAPI
Key Components

Document ingestion

Eight policy documents are stored under:

support_assistant/docs/

Embeddings

The project uses the local:

all-MiniLM-L6-v2

embedding model.

Vector database

ChromaDB stores the document embeddings and supports cosine-similarity
retrieval.

LangGraph

The graph contains three main stages:

classify_intent
      ↓
   routing
   ↙     ↘
retrieve  direct
and_answer answer

Structured output

Responses are validated using Pydantic and contain:

{
  "answer": "...",
  "sources": [],
  "confidence": 1.0
}

Mock LLM mode

The required baseline uses deterministic mock behavior:

MOCK_LLM unset → Mock mode
MOCK_LLM=1     → Mock mode
MOCK_LLM=0     → Optional real LLM mode

No external LLM API key is required for the required baseline.

Run
cd support_assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Build the vector database:

python ingest.py

Start the API:

uvicorn main:app --host 0.0.0.0 --port 7860

The service runs at:

http://localhost:7860
Example Request
curl -X POST "http://localhost:7860/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the delivery time?"}'
Docker
docker build -t support-assistant .
docker run --rm -p 7860:7860 support-assistant

Detailed documentation:

Support Assistant README

Setup Strategy

This repository uses one requirements.txt per module.

Each module is installed independently:

data_pipeline/requirements.txt
analytics/requirements.txt
support_assistant/requirements.txt

This keeps the environments separated because each module uses a different
set of dependencies.

Running the Complete Project

The modules are independent and can be executed separately.

1. Data Pipeline
cd data_pipeline
pip install -r requirements.txt

Run:

module_1_datapipeline.ipynb
2. Analytics
cd analytics
pip install -r requirements.txt

Run:

01_eda.ipynb
02_modeling.ipynb
3. Support Assistant
cd support_assistant
pip install -r requirements.txt
python ingest.py
uvicorn main:app --host 0.0.0.0 --port 7860
Design Decisions

Data Pipeline:

A web-scraping workflow was selected to demonstrate extraction, cleaning,
transformation, persistent database storage, SQL querying, and pandas-based
validation in one pipeline.

Analytics:

The analytics workflow separates data exploration from machine learning.
Preprocessing is handled systematically, multiple classification models are
compared, class imbalance is addressed explicitly, and the final fitted
pipeline is saved for reuse.

Support Assistant:

A local RAG architecture was selected to demonstrate document ingestion,
semantic retrieval, graph-based orchestration, structured outputs, and API
serving. The deterministic mock mode keeps the required workflow reproducible
without depending on an external LLM provider.