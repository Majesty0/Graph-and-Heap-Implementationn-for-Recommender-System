# Graph-Heap Recommender System

A hybrid recommendation engine that combines:
- **Graph-based collaborative filtering**
- **Heap-based top-N ranking**
- **Real-time interaction updates**

The project includes a polished **Streamlit dashboard** for interactive exploration and a **CLI mode** for quick terminal testing.

> **📚 For in-depth technical discussion**: Read [ARCHITECTURE.md](ARCHITECTURE.md) for algorithms, design decisions, complexity analysis, and extension points.

---

## What This Project Does

At its core: **Given a user, recommend items they will likely enjoy.**

This is solved using a **hybrid three-factor approach**:

1. **User-based Collaborative Filtering**: "What do users similar to me prefer?"
2. **Item-based Collaborative Filtering**: "How similar is this to items I already like?"
3. **Popularity**: "How well-known/trusted is this item?"

These three signals are blended (65% user-CF, 25% item-CF, 10% popularity) to balance exploration and exploitation.

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
- Modular engine design and real-time updates
- Dual-index architecture for O(1) lookups

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

## Screenshots

> Add your UI images in a `screenshots/` folder at the project root.

- `dashboard-overview`![alt text](image.png)
- `recommendation-analytics`![alt text](image-1.png)
- `network-graph`![alt text](image-2.png)
- `demo.gif` (optional)
- `analytics`![alt text](image-3.png)


#### Video Demo:

```html
<video width="100%" controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

> **Max size**: ~100MB on GitHub. GitHub supports MP4, WebM, Ogg formats.

### Optional Demo GIF

![Live Demo](screenshots/demo.gif)

---

## How the Hybrid Score Works

For each candidate item not yet rated by the target user:

```
Final Score = 0.65 × User-CF + 0.25 × Item-CF + 0.10 × Popularity
```

### Components

**User-CF (65%)**: Aggregated ratings from similar users
```
User-CF = Σ [similarity(target, neighbor) × rating(neighbor, item)]
```

**Item-CF (25%)**: How similar is this to items I already rated?
```
Item-CF = Σ [similarity(seen_item, candidate) × rating(target, seen_item)]
```

**Popularity (10%)**: Item interaction frequency
```
Popularity = |Users who rated item| / Total Users
```

→ **See [ARCHITECTURE.md](ARCHITECTURE.md) for mathematical details and complexity analysis.**

---

## Core Concepts

### Dual-Index Architecture

```python
user_items = {"U1": {"I1": 5.0, "I2": 4.0}, ...}  # Fast user lookup
item_users = {"I1": {"U1": 5.0, ...}, ...}         # Fast item lookup
graph = nx.Graph()                                 # For visualization
```

Redundancy for speed: both O(1) lookups.

### Similarity Metrics

Both users and items use a 60/40 blend:
- **Graph Overlap (60%)**: Jaccard-like ratio of common connections
- **Cosine Similarity (40%)**: Alignment of rating vectors on common items

### Top-N Selection

Using `heapq.nlargest(n, candidates)` instead of full sort: O(n log n) but faster in practice.

---

## Technical Notes

- **In-memory**: No database required
- **Real-time**: Update recommendations instantly via `add_interaction()`
- **Modular**: Engine testable independently of UI
- **Extensible**: Add custom similarity metrics or scoring components

→ **See [ARCHITECTURE.md](ARCHITECTURE.md) for performance, design decisions, and extension examples.**

---

## Author

Maintained in this repository: https://github.com/Majesty0/Graph-and-Heap-Implementationn-for-Recommender-System
