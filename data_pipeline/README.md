# Module 1 — Data Pipeline (`/data_pipeline`)

Scrapes live book-catalog data from a public scraping-practice site, cleans and
types it, converts price to INR at a fixed project rate, loads it into a
normalized SQLite database, and queries it with both raw SQL and pandas.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. All other imports used (`sqlite3`, `re`, `os`,
`urllib.parse`) are from the standard library.

## How to run

1. Open `module_1_datapipeline.ipynb` in Jupyter / JupyterLab / VS Code.
2. Run all cells top to bottom (`Run All`). No manual copy-pasting is
   required — scraping, cleaning, DB creation, inserts, and all queries
   execute end to end.
3. If re-running from a previous execution, delete `books.db` first (or run
   the DB-setup cell, which now recreates it fresh) — `book_id` is inserted
   explicitly, so a stale database will raise a `UNIQUE constraint failed`
   error on a second run.

Outputs produced by a full run:
- `books_three_categories.csv` — raw scraped + cleaned dataset
- `books.db` — the populated SQLite database
- `sql_query_results.txt` — all 6 SQL queries with their output

## Data source & scraping approach

**Source:** [books.toscrape.com](https://books.toscrape.com) — a public
scraping-practice site, no login/API key/paid tier required.

**Scope:** all books across 3 full categories, following each category's
pagination (`next` link) until exhausted, rather than paginating the "all
products" listing:

| Category | Books scraped |
|---|---|
| Travel | 11 |
| Mystery | 32 |
| Historical Fiction | 26 |
| **Total** | **69** |

Three categories were chosen over the "all products" pages so the
`category` field carries real, meaningful groupings for the JOIN query later,
rather than a single undifferentiated value.

For each book: `title`, `price` (GBP, as listed), `star_rating` (text),
`availability` (text), `category`.

## Cleaning & type-conversion decisions

- **`price_gbp`** (float) — currency symbol stripped from the listed price
  string and cast to `float`.
- **`rating`** (int, 1–5) — text rating (`One`…`Five`) mapped to its integer
  value via a fixed lookup table.
- **`in_stock`** (bool) — derived from an exact match against the listing
  page's `"In stock"` text.
- **Malformed rows:** all 69 scraped rows parsed cleanly on every field (0
  nulls across every column after cleaning) — the source site's markup is
  consistent, so no row required dropping or imputation. The pipeline is
  still written defensively: any price that fails to parse is imputed with
  the column median, and any row with an unrecognized rating value is
  dropped, rather than letting either failure crash the run.

## Currency conversion

`price_inr` is computed from a fixed, project-defined baseline rate — **1
GBP = 105.50 INR** — stated here as required. This is not a live or
historical market rate and needs no external lookup:

```python
price_inr = price_gbp * 105.50
```

## Database schema

Normalized SQLite schema, two tables sharing a PK/FK relationship:

```sql
CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE NOT NULL
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    price_gbp   REAL,
    price_inr   REAL,
    rating      INTEGER,
    in_stock    INTEGER,
    category_id INTEGER REFERENCES categories(category_id)
);
```

## SQL queries

Six queries are defined and executed against the database (≥5 required),
collectively covering every required clause:

| Query | Demonstrates |
|---|---|
| `Q1_SELECT_WHERE` | `SELECT` / `WHERE` — books priced above £40 |
| `Q2_ORDER_BY` | `ORDER BY` — all books by rating, descending |
| `Q3_LIMIT` | `ORDER BY` + `LIMIT` — 10 priciest books |
| `Q4_DISTINCT` | `DISTINCT` — distinct category names |
| `Q5_BETWEEN` | `BETWEEN` — books rated 3–5 stars |
| `Q6_JOIN` | `JOIN` (+ `ORDER BY`, `LIMIT`) — top 10 rated books per category with `category_name` |

Each query's SQL text and result set are printed to stdout and written to
`sql_query_results.txt`.

## pandas cross-check

- `Q1_SELECT_WHERE` and `Q6_JOIN` are read back from SQLite via
  `pd.read_sql(...)`.
- The `Q6_JOIN` result is independently reproduced using `pd.merge(...)` on
  the in-memory DataFrames (no SQL involved), sorted the same way (`rating`
  desc, `price_gbp` desc), and compared to the SQL result — confirmed
  identical via `.equals()`.

## Files in this module

```
data_pipeline/
├── module_1_datapipeline.ipynb   # scrape → clean → convert → load → query
├── books.db                      # populated SQLite database
├── books_three_categories.csv    # cleaned scraped dataset
├── sql_query_results.txt         # all 6 queries with output
├── requirements.txt
└── README.md
```
