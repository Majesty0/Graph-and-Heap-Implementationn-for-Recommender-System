# Graph-Heap Recommender System

A hybrid recommendation engine that combines:
- **Graph-based collaborative filtering**
- **Heap-based top-N ranking**
- **Real-time interaction updates**

The project includes a polished **Streamlit dashboard** for interactive exploration and a **CLI mode** for quick terminal testing.

---

## Features

- Hybrid recommendation scoring:
  - **User-CF** component
  - **Item-CF** component
  - **Popularity** component
- Fast top-N retrieval using `heapq`
- User-item interaction graph powered by `networkx`
- Similar user and similar item discovery
- Real-time updates (apply new ratings and instantly recompute)
- Beautiful Streamlit UI with:
  - Recommendation tables
  - Score breakdown charts
  - Recommendation strength chart
  - Interactive user-item network graph

---

## Tech Stack

- Python 3.x
- Streamlit
- NetworkX
- Pandas
- Plotly

---

## Project Structure

```text
Graph-and-Heap-Implementationn-for-Recommender-System/
├── Recommender System.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---

## Setup

### 1) Clone and enter the project

```bash
git clone https://github.com/Majesty0/Graph-and-Heap-Implementationn-for-Recommender-System.git
cd Graph-and-Heap-Implementationn-for-Recommender-System
```

### 2) Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the App

### Streamlit GUI (recommended)

```bash
streamlit run "Recommender System.py"
```

If `streamlit` command is not recognized:

```bash
python -m streamlit run "Recommender System.py"
```

### CLI mode

```bash
python "Recommender System.py"
```

---

## How the Hybrid Score Works

For each candidate item:

\[
\text{Final Score} = 0.65 \times \text{User-CF} + 0.25 \times \text{Item-CF} + 0.10 \times \text{Popularity}
\]

- **User-CF**: weighted influence from similar users
- **Item-CF**: similarity to items the user already interacted with
- **Popularity**: normalized item interaction frequency

---

## Notes

- The app auto-detects Streamlit runtime and renders GUI in Streamlit context.
- When launched with plain Python, it runs a CLI demonstration.

---

## Author

Maintained in this repository: https://github.com/Majesty0/Graph-and-Heap-Implementationn-for-Recommender-System
