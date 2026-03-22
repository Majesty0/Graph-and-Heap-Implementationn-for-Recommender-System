# Architecture & Design Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Core Architecture](#core-architecture)
4. [Algorithm Details](#algorithm-details)
5. [Data Structures](#data-structures)
6. [Design Decisions](#design-decisions)
7. [Performance Considerations](#performance-considerations)
8. [UI & Visualization Layer](#ui--visualization-layer)

---

## Project Overview

The **Graph-Heap Recommender System** is a hybrid recommendation engine that demonstrates the practical application of two fundamental computer science concepts:

- **Graph Theory**: User-item interactions modeled as a bipartite graph
- **Heap/Priority Queue**: Efficient top-N selection using O(n log k) operations

### What This Project Does

At its core, the system answers the question: **"What items should we recommend to a given user?"**

It accomplishes this through a sophisticated 3-factor hybrid approach:

1. **User-based Collaborative Filtering (User-CF)**: Finds similar users and leverages their preferences
2. **Item-based Collaborative Filtering (Item-CF)**: Finds similar items to what the user already likes
3. **Popularity Bias**: Considers global demand across all users

The final recommendation score is a weighted combination of these three signals, delivering nuanced, context-aware recommendations.

---

## Problem Statement

### Real-World Context

In recommendation systems (e-commerce, streaming, social media), we face three key challenges:

1. **Cold Start**: New items/users with limited interaction history
2. **Exploration vs. Exploitation**: Balance between recommending known-good items vs. new discoveries
3. **Scalability**: Computing recommendations in real-time for millions of users/items

### This Project's Approach

- **Hybrid Scoring** handles the exploration-exploitation tradeoff through weighted combination
- **Graph-based modeling** naturally represents multi-faceted relationships
- **Heap-based selection** provides near-linear time top-N retrieval
- **Real-time updates** allow immediate reflection of user feedback

---

## Core Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│           Graph-Heap Recommender System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │      HybridRecommendationEngine (Core Logic)          │  │
│  │  ─────────────────────────────────────────────────   │  │
│  │  • graph: NetworkX bipartite graph                   │  │
│  │  • user_items: dict[user_id → dict[item_id → rating]]│  │
│  │  • item_users: dict[item_id → dict[user_id → rating]]│  │
│  │                                                       │  │
│  │  Key Methods:                                        │  │
│  │  • add_interaction() - Update graph & lookups       │  │
│  │  • _user_similarity() - Compute user similarity     │  │
│  │  • _item_similarity() - Compute item similarity     │  │
│  │  • recommend_top_n() - Hybrid scoring & ranking    │  │
│  │  • find_similar_users() - Top-K similar users      │  │
│  │  • find_similar_items() - Top-K similar items      │  │
│  └───────────────────────────────────────────────────────┘  │
│                         ▲                                    │
│        ┌────────────────┴────────────────┐                  │
│        │                                  │                  │
│  ┌─────▼──────────┐          ┌───────────▼──────┐           │
│  │  CLI Interface │          │  Streamlit UI    │           │
│  │  (run_cli_demo)│          │ (run_streamlit_  │           │
│  │                │          │      app)        │           │
│  │  • Console     │          │  • Dashboard     │           │
│  │    output      │          │  • Charts/Graphs │           │
│  │  • Static mode │          │  • Real-time     │           │
│  │                │          │    updates       │           │
│  └────────────────┘          │  • Network viz   │           │
│                               └──────────────────┘           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Model

```
Bipartite User-Item Graph:

Users                          Items
  U1 ─── 5.0 ──────────────── I1
  │  \  4.0 ──────────────── I2
  │   \ 2.5 ──────────────── I3
  │    \
  U2 ────── 4.5 ──────────── I4
  │  \  3.0 ──────────────── I5
  │   \ 4.0
  ...

Each edge has a weight (user's rating of the item).
Nodes store metadata (node_type: "user" or "item").
```

### Dual Index Structure

```python
# User → Items view (fast lookup: "What did this user rate?")
user_items = {
    "U1": {"I1": 5.0, "I2": 4.0, "I3": 2.5},
    "U2": {"I1": 4.5, "I3": 3.0, "I4": 4.0},
    ...
}

# Item → Users view (fast lookup: "Who rated this item?")
item_users = {
    "I1": {"U1": 5.0, "U2": 4.5, ...},
    "I2": {"U1": 4.0, ...},
    ...
}

# NetworkX Graph (structural queries and visualization)
graph = nx.Graph()
# Nodes: U:U1, U:U2, I:I1, I:I2, ...
# Edges: (U:U1, I:I1, weight=5.0), ...
```

---

## Algorithm Details

### 1. User Similarity Computation

Matching two users is a two-part metric:

#### Part A: Graph Overlap Similarity

Measures how much users' "taste neighborhoods" overlap:

```
Graph Overlap = |Common Items| / sqrt(|Items_U1| × |Items_U2|)
```

**Intuition**: Two users who rate the same 5 items out of 10 each are more similar than two users who rate the same 5 items out of 100 each.

**Example**:
- U1 rated: {I1, I2, I3}
- U2 rated: {I1, I3, I4}
- Common: {I1, I3}
- Graph Overlap = 2 / sqrt(3 × 3) = 2/3 ≈ 0.667

#### Part B: Cosine Similarity (Rating Vectors)

Captures the direction of preference:

```
Cosine(U1, U2) = (U1·U2) / (||U1|| × ||U2||)
```

where vectors consist only of common items' ratings.

**Intuition**: Users who rate items similarly (high for good, low for bad) are aligned in taste.

**Example**:
- U1 ratings on common items: [5.0, 2.5]  
- U2 ratings on common items: [4.5, 3.0]
- Dot product: 5.0×4.5 + 2.5×3.0 = 22.5 + 7.5 = 30.0
- ||U1|| = sqrt(25 + 6.25) = sqrt(31.25) ≈ 5.59
- ||U2|| = sqrt(20.25 + 9) = sqrt(29.25) ≈ 5.41
- Cosine ≈ 30.0 / (5.59 × 5.41) ≈ 0.997

#### Final User Similarity

```
similarity(U1, U2) = 0.6 × Graph_Overlap + 0.4 × Cosine
```

The 60/40 weighting balances:
- **Graph Overlap (60%)**: Structural similarity (do they interact with similar items?)
- **Cosine (40%)**: Rating patterns (do they rate items similarly?)

### 2. Item Similarity Computation

Mirror of user similarity, but comparing items by shared raters:

```python
# Graph Overlap: Common raters ratio
# Cosine: Rating vector alignment of those raters
# Final: 0.6 × Graph + 0.4 × Cosine
```

**Key difference**: Items are compared through users who rated them, not through direct item-item relationships.

### 3. Hybrid Recommendation Scoring

For each **candidate item** (not yet rated by the target user):

#### Component 1: User-CF Score

"What do users similar to me rate this item?"

```
User-CF(item) = Σ [similarity(target, neighbor) × rating(neighbor, item)]
                 for each neighbor in top_k_similar_users
```

**Example**:
- Target User: U1  
- Similar Users: U2 (sim=0.7, rated I4=4.5), U3 (sim=0.5, rated I4=3.5)
- User-CF(I4) = 0.7×4.5 + 0.5×3.5 = 3.15 + 1.75 = 4.9

#### Component 2: Item-CF Score

"How similar is this item to items I already like?"

```
Item-CF(item) = Σ [similarity(target_item, candidate) × rating(target, target_item)]
                 for each item target_user has rated
```

**Example**:
- Target User U1 rated: I1 (5.0), I2 (4.0)
- Similarity: sim(I1, I4)=0.6, sim(I2, I4)=0.4  
- Item-CF(I4) = 0.6×5.0 + 0.4×4.0 = 3.0 + 1.6 = 4.6

#### Component 3: Popularity Score

"How many users have rated this item?"

```
Popularity(item) = |Users who rated item| / Total_Users
```

Ranges from 0 to 1. Prevents recommending obscure items.

**Example**:
- 5 total users, 3 rated I4
- Popularity(I4) = 3/5 = 0.6

#### Final Score (Weighted Ensemble)

```
Final_Score(item) = 0.65 × User-CF + 0.25 × Item-CF + 0.10 × Popularity
```

**Weighting rationale**:
- **User-CF (65%)**: Most important; users like what their peers like
- **Item-CF (25%)**: Secondary signal; consistency with user's taste
- **Popularity (10%)**: Tie-breaker; avoid too-obscure recommendations

### 4. Top-N Selection Using Heap

```python
# Naive approach: O(n log n) sort
top_items = sorted(all_scored_items, key=lambda x: x[1], reverse=True)[:n]

# Optimized approach: O(n log n) heap
import heapq
top_items = heapq.nlargest(n, all_scored_items)
```

For large item catalogs, `heapq.nlargest()` is more efficient when k (top-N) << n (total items).

---

## Data Structures

### NetworkX Graph

```python
# Bipartite graph representation
graph = nx.Graph()

# Add users and items as nodes
graph.add_node("U:U1", node_type="user")
graph.add_node("I:I1", node_type="item")

# Add edges with ratings as weights
graph.add_edge("U:U1", "I:I1", weight=5.0)
```

**Why NetworkX?**
- Handles bipartite relationships naturally
- Supports node/edge metadata
- Enables network visualization (Plotly)
- Graph algorithms (though not extensively used here)

### defaultdict for Fast Lookups

```python
user_items = defaultdict(dict)  # O(1) user lookup, O(1) item within user
item_users = defaultdict(dict)  # O(1) item lookup, O(1) user within item
```

Both dictionaries are maintained in sync during `add_interaction()`.

### heapq for Efficient Ranking

```python
# Convert scores to (score, id) tuples
candidates = [(score, user_id) for score, user_id in candidates if score > 0]

# O(n log k) selection of top-k without full sort
top_k = heapq.nlargest(k, candidates)
```

---

## Design Decisions

### 1. Hybrid Scoring Approach

**Alternative**: Single-signal systems (pure user-CF, pure item-CF, pure popularity)

**Chosen**: Weighted ensemble

**Rationale**:
- User-CF alone misses items your peers haven't seen
- Item-CF alone struggles with uncorrelated users
- Popularity alone recommends mainstream items to everyone
- Combination provides balanced recommendations

### 2. Graph + Dual-Index Architecture

**Alternative**: Just use the graph; query it every time

**Chosen**: Graph + dual-index dicts

**Rationale**:
- Graph provides visualization and structural insights
- Dictionaries provide O(1) lookup for recommendation computation
- Memory trade-off (store redundant info) for O(n) speed-up

### 3. Similarity Metric (60/40 Graph-Cosine Split)

**Alternative**: Pure cosine, pure graph overlap, or other combinations

**Chosen**: Weighted blend

**Rationale**:
- Graph overlap captures breadth of interaction
- Cosine captures depth of alignment
- 60/40 empirically balances both signals

### 4. Real-Time Updates

**Alternative**: Batch recomputation (rebuild engine nightly)

**Chosen**: Incremental updates via `add_interaction()`

**Rationale**:
- Supports interactive exploration in UI
- Reflects user feedback immediately
- DatabaseAgnostic (works in-memory)

### 5. Modular UI/Engine Separation

**Alternative**: Monolithic script mixing logic and UI

**Chosen**: Clean `HybridRecommendationEngine` class + UI layer

**Rationale**:
- Engine testable in isolation
- Supports multiple UI modes (CLI, Streamlit, future web service)
- Easier to extend (add new similarity metrics, scoring logic)

---

## Performance Considerations

### Time Complexity

| Operation                | Complexity      | Notes                                                 |
|--------------------------|-----------------|-------------------------------------------------------|
| `add_interaction()`       | O(1)            | Dict insertion + graph node/edge add                 |
| `_user_similarity()`      | O(I)            | I = avg items per user (usually small)               |
| `_item_similarity()`      | O(U)            | U = avg users per item (usually small)               |
| `find_similar_users()`    | O(U² × I)       | Compare all user pairs, check common items           |
| `find_similar_items()`    | O(I² × U)       | Compare all item pairs, check common raters          |
| `recommend_top_n()`       | O(k×I + k×I + C log n) | k=neighbor depth, C=candidates, n=top_n      |
| `heapq.nlargest(n, C)`    | O(C log n)      | Efficient top-N selection                           |

**Scalability Notes**:
- Works well for up to ~10k users × 10k items with good locality
- `find_similar_users()` and `find_similar_items()` scale quadratically but are cached client-side in UI
- For large-scale systems, would need approximate algorithms (LSH, sketches)

### Space Complexity

| Structure           | Space         |
|-------------------|-------------|
| `user_items`      | O(U × I_avg)   |
| `item_users`      | O(I × U_avg)   |
| `graph`           | O(U + I + E)   |
| Total             | O(U×I) worst case |

---

## UI & Visualization Layer

### Streamlit Dashboard

The Streamlit UI (`run_streamlit_app()`) wraps the engine with:

1. **Sidebar Controls**:
   - User selection
   - Top-N and neighbor-depth sliders
   - Item probe selector
   - Real-time interaction updater

2. **Main Metrics**:
   - User count, item count, recommendation count, latency

3. **Recommendation Table**:
   - Rank, item ID, final score, component scores

4. **Charts**:
   - Recommendation Strength (bar chart of final scores)
   - Score Components (stacked bar: User-CF, Item-CF, Popularity)

5. **Similarity Insights**:
   - Top-K similar users to target user
   - Top-K similar items to probe item

6. **Interactive Network Graph** (Plotly):
   - Bipartite graph visualization
   - Purple nodes = users, cyan nodes = items
   - Node size = degree (connection count)
   - Target user highlighted

7. **Help Section**:
   - Expandable legend explaining all charts and colors

### CLI Mode

The CLI mode (`run_cli_demo()`) prints:
- Initial recommendations for U1
- Similar users and items
- Effect of a real-time rating update

---

## Extension Points

### Adding New Similarity Metrics

```python
def _user_similarity_pearson(self, u1, u2):
    """Pearson correlation instead of cosine."""
    # Compute on common items...
    return pearson_correlation

# Update recommend_top_n() to use new metric
```

### Adding New Scoring Components

```python
def _genre_similarity(self, item1, item2):
    """Genre overlap as addition to item similarity."""
    # If items had genre metadata...
    
# Update recommend_top_n() to include genre component
```

### Saving/Loading Engine State

```python
import pickle
with open('engine_state.pkl', 'wb') as f:
    pickle.dump(self.user_items, f)  # Save lookups
```

### Batch Import from CSV

```python
import csv
with open('ratings.csv') as f:
    for user, item, rating in csv.reader(f):
        engine.add_interaction(user, item, float(rating))
```

---

## Summary

The **Graph-Heap Recommender System** demonstrates how classical data structures and algorithms—graphs, heaps, similarity metrics—combine to create a practical recommendation engine. The hybrid scoring approach balances multiple signals, the graph structure enables insightful visualization, and the heap selection ensures efficient top-N ranking.

This design is production-aware (real-time updates, modular architecture) while remaining simple enough to understand and extend.
