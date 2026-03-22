import heapq
import os
import time
from collections import defaultdict

import networkx as nx

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


class HybridRecommendationEngine:
    def __init__(self, interactions):
        self.graph = nx.Graph()
        self.user_items = defaultdict(dict)
        self.item_users = defaultdict(dict)
        self._build_graph(interactions)

    @staticmethod
    def _u_node(user_id):
        return f"U:{user_id}"

    @staticmethod
    def _i_node(item_id):
        return f"I:{item_id}"

    def _build_graph(self, interactions):
        for user_id, item_id, rating in interactions:
            self.add_interaction(user_id, item_id, rating)

    def add_interaction(self, user_id, item_id, rating):
        self.user_items[user_id][item_id] = rating
        self.item_users[item_id][user_id] = rating
        self.graph.add_node(self._u_node(user_id), node_type="user")
        self.graph.add_node(self._i_node(item_id), node_type="item")
        self.graph.add_edge(self._u_node(user_id), self._i_node(item_id), weight=rating)

    def _user_similarity(self, u1, u2):
        items_u1 = self.user_items[u1]
        items_u2 = self.user_items[u2]

        if not items_u1 or not items_u2:
            return 0.0

        set_u1 = set(items_u1)
        set_u2 = set(items_u2)
        common = set_u1 & set_u2

        if not common:
            return 0.0

        graph_overlap = len(common) / ((len(set_u1) * len(set_u2)) ** 0.5)

        dot = sum(items_u1[item] * items_u2[item] for item in common)
        norm_u1 = sum(score * score for score in items_u1.values()) ** 0.5
        norm_u2 = sum(score * score for score in items_u2.values()) ** 0.5
        cosine = dot / (norm_u1 * norm_u2) if norm_u1 > 0 and norm_u2 > 0 else 0.0

        return 0.6 * graph_overlap + 0.4 * cosine

    def _item_similarity(self, i1, i2):
        users_i1 = self.item_users[i1]
        users_i2 = self.item_users[i2]

        if not users_i1 or not users_i2:
            return 0.0

        set_i1 = set(users_i1)
        set_i2 = set(users_i2)
        common_users = set_i1 & set_i2

        if not common_users:
            return 0.0

        graph_overlap = len(common_users) / ((len(set_i1) * len(set_i2)) ** 0.5)

        dot = sum(users_i1[user] * users_i2[user] for user in common_users)
        norm_i1 = sum(score * score for score in users_i1.values()) ** 0.5
        norm_i2 = sum(score * score for score in users_i2.values()) ** 0.5
        cosine = dot / (norm_i1 * norm_i2) if norm_i1 > 0 and norm_i2 > 0 else 0.0

        return 0.6 * graph_overlap + 0.4 * cosine

    def find_similar_users(self, user_id, top_k=3):
        if user_id not in self.user_items:
            return []

        candidates = []
        for other_user in self.user_items:
            if other_user == user_id:
                continue
            similarity = self._user_similarity(user_id, other_user)
            if similarity > 0:
                candidates.append((similarity, other_user))

        top = heapq.nlargest(top_k, candidates)
        return [(user, score) for score, user in top]

    def find_similar_items(self, item_id, top_k=3):
        if item_id not in self.item_users:
            return []

        candidates = []
        for other_item in self.item_users:
            if other_item == item_id:
                continue
            similarity = self._item_similarity(item_id, other_item)
            if similarity > 0:
                candidates.append((similarity, other_item))

        top = heapq.nlargest(top_k, candidates)
        return [(item, score) for score, item in top]

    def recommend_top_n(self, user_id, top_n=5, neighbor_k=3):
        if user_id not in self.user_items:
            return []

        seen_items = set(self.user_items[user_id])
        all_items = set(self.item_users)
        candidate_items = all_items - seen_items

        if not candidate_items:
            return []

        similar_users = self.find_similar_users(user_id, top_k=neighbor_k)

        user_cf_scores = defaultdict(float)
        for neighbor_id, sim_score in similar_users:
            for item_id, rating in self.user_items[neighbor_id].items():
                if item_id not in seen_items:
                    user_cf_scores[item_id] += sim_score * rating

        item_cf_scores = defaultdict(float)
        for seen_item, user_rating in self.user_items[user_id].items():
            for candidate in candidate_items:
                sim_item = self._item_similarity(seen_item, candidate)
                if sim_item > 0:
                    item_cf_scores[candidate] += sim_item * user_rating

        popularity_scores = {}
        total_users = max(1, len(self.user_items))
        for item in candidate_items:
            popularity_scores[item] = len(self.item_users[item]) / total_users

        final_scored = []
        for item in candidate_items:
            score = (
                0.65 * user_cf_scores[item]
                + 0.25 * item_cf_scores[item]
                + 0.10 * popularity_scores[item]
            )
            final_scored.append((score, item))

        top_items = heapq.nlargest(top_n, final_scored)
        return [
            {
                "item": item,
                "score": round(score, 4),
                "user_cf": round(user_cf_scores[item], 4),
                "item_cf": round(item_cf_scores[item], 4),
                "popularity": round(popularity_scores[item], 4),
            }
            for score, item in top_items
        ]

    def recommend_realtime(self, user_id, top_n=5, neighbor_k=3):
        started = time.perf_counter()
        results = self.recommend_top_n(user_id, top_n=top_n, neighbor_k=neighbor_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return results, round(elapsed_ms, 3)


def print_recommendations(user_id, recommendations, latency_ms):
    print(f"\nTop {len(recommendations)} recommendations for User {user_id} (latency: {latency_ms} ms)")
    print("-" * 90)
    print(f"{'Rank':<6}{'Item':<10}{'Score':<12}{'User-CF':<12}{'Item-CF':<12}{'Popularity':<12}")
    print("-" * 90)
    for index, rec in enumerate(recommendations, 1):
        print(
            f"{index:<6}{rec['item']:<10}{rec['score']:<12}{rec['user_cf']:<12}{rec['item_cf']:<12}{rec['popularity']:<12}"
        )


def print_similarities(engine, user_id, probe_item):
    print(f"\nMost similar users to {user_id}:")
    similar_users = engine.find_similar_users(user_id, top_k=3)
    if not similar_users:
        print("No similar users found.")
    else:
        for rank, (other_user, score) in enumerate(similar_users, 1):
            print(f"{rank}. User {other_user} (similarity={score:.4f})")

    print(f"\nMost similar items to {probe_item}:")
    similar_items = engine.find_similar_items(probe_item, top_k=3)
    if not similar_items:
        print("No similar items found.")
    else:
        for rank, (other_item, score) in enumerate(similar_items, 1):
            print(f"{rank}. Item {other_item} (similarity={score:.4f})")


def default_interactions():
    return [
        ("U1", "I1", 5.0), ("U1", "I2", 4.0), ("U1", "I3", 2.5),
        ("U2", "I1", 4.5), ("U2", "I3", 3.0), ("U2", "I4", 4.0),
        ("U3", "I2", 5.0), ("U3", "I4", 3.5), ("U3", "I5", 4.0),
        ("U4", "I1", 2.0), ("U4", "I5", 4.5), ("U4", "I6", 5.0),
        ("U5", "I2", 4.5), ("U5", "I3", 4.0), ("U5", "I6", 3.5),
    ]


def _inject_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #0f172a 0%, #111827 35%, #020617 100%);
            }
            .hero {
                padding: 1.2rem 1.4rem;
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(34, 211, 238, 0.25), rgba(99, 102, 241, 0.28));
                border: 1px solid rgba(226, 232, 240, 0.18);
                margin-bottom: 1rem;
            }
            .hero h1 {
                margin: 0;
                color: #f8fafc;
                font-size: 1.95rem;
            }
            .hero p {
                margin-top: .35rem;
                color: #cbd5e1;
                font-size: 0.98rem;
            }
            .card {
                padding: 0.95rem;
                border-radius: 14px;
                background: rgba(15, 23, 42, 0.65);
                border: 1px solid rgba(148, 163, 184, 0.25);
                margin-bottom: 0.9rem;
            }
            .card-title {
                font-weight: 650;
                color: #e2e8f0;
                margin-bottom: 0.45rem;
            }
            [data-testid="stMetric"] {
                background: rgba(15, 23, 42, 0.70);
                border: 1px solid rgba(148, 163, 184, 0.24);
                border-radius: 12px;
                padding: .55rem .7rem;
            }
            .stDataFrame {
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_engine():
    if "engine" not in st.session_state:
        st.session_state.engine = HybridRecommendationEngine(default_interactions())


def _build_network_figure(engine, focus_user=None):
    graph = engine.graph
    node_positions = nx.spring_layout(graph, seed=42)

    edge_x = []
    edge_y = []
    edge_text = []
    for source, target, data in graph.edges(data=True):
        x0, y0 = node_positions[source]
        x1, y1 = node_positions[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{source} ↔ {target} | Rating: {data.get('weight', 0)}")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.2, color="rgba(148,163,184,0.55)"),
        hoverinfo="none",
        mode="lines",
    )

    user_x, user_y, user_text, user_size = [], [], [], []
    item_x, item_y, item_text, item_size = [], [], [], []

    for node, attrs in graph.nodes(data=True):
        x, y = node_positions[node]
        degree = max(1, graph.degree[node])
        is_focus = node == f"U:{focus_user}" if focus_user else False
        if attrs.get("node_type") == "user":
            user_x.append(x)
            user_y.append(y)
            user_text.append(f"{node}<br>Connections: {degree}")
            user_size.append(22 if is_focus else 13 + degree * 1.5)
        else:
            item_x.append(x)
            item_y.append(y)
            item_text.append(f"{node}<br>Connections: {degree}")
            item_size.append(12 + degree * 1.2)

    user_trace = go.Scatter(
        x=user_x,
        y=user_y,
        mode="markers",
        hoverinfo="text",
        text=user_text,
        marker=dict(color="#6366f1", size=user_size, line=dict(width=1, color="#1e1b4b")),
        name="Users",
    )

    item_trace = go.Scatter(
        x=item_x,
        y=item_y,
        mode="markers",
        hoverinfo="text",
        text=item_text,
        marker=dict(color="#22d3ee", size=item_size, line=dict(width=1, color="#164e63")),
        name="Items",
    )

    figure = go.Figure(data=[edge_trace, user_trace, item_trace])
    figure.update_layout(
        showlegend=True,
        margin=dict(l=15, r=15, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="closest",
        height=460,
    )
    return figure


def run_streamlit_app():
    if st is None or pd is None or go is None:
        missing = []
        if st is None:
            missing.append("streamlit")
        if pd is None:
            missing.append("pandas")
        if go is None:
            missing.append("plotly")
        raise ModuleNotFoundError(
            "Missing GUI dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install -r requirements.txt"
        )

    st.set_page_config(
        page_title="Graph-Heap Recommender System",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()
    _ensure_engine()

    st.markdown(
        """
        <div class="hero">
            <h1>Graph-Heap Recommender System</h1>
            <p>Graph + Heap powered recommendations with real-time updates and beautifully structured analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    engine = st.session_state.engine
    users = sorted(engine.user_items.keys())
    items = sorted(engine.item_users.keys())

    with st.sidebar:
        st.header("Controls")
        target_user = st.selectbox("Select user", users, index=0)
        top_n = st.slider("Top recommendations", min_value=1, max_value=10, value=4)
        neighbor_k = st.slider("Neighbor depth", min_value=1, max_value=max(1, len(users) - 1), value=3)
        probe_item = st.selectbox("Probe item similarity", items, index=min(1, len(items) - 1))

        st.markdown("---")
        st.subheader("Real-time interaction")
        update_user = st.selectbox("User", users, key="update_user")
        update_item = st.selectbox("Item", items, key="update_item")
        update_rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.5)

        col_a, col_b = st.columns(2)
        with col_a:
            add_clicked = st.button("Apply", use_container_width=True)
        with col_b:
            reset_clicked = st.button("Reset", use_container_width=True)

    if add_clicked:
        engine.add_interaction(update_user, update_item, update_rating)
        st.success(f"Updated: {update_user} rated {update_item} = {update_rating}")

    if reset_clicked:
        st.session_state.engine = HybridRecommendationEngine(default_interactions())
        st.success("Engine reset to default interactions.")
        st.rerun()

    recs, latency = engine.recommend_realtime(target_user, top_n=top_n, neighbor_k=neighbor_k)
    similar_users = engine.find_similar_users(target_user, top_k=neighbor_k)
    similar_items = engine.find_similar_items(probe_item, top_k=neighbor_k)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Users", len(engine.user_items))
    metric_col2.metric("Items", len(engine.item_users))
    metric_col3.metric("Recommendations", len(recs))
    metric_col4.metric("Latency", f"{latency} ms")

    with st.expander("How to read this dashboard", expanded=False):
        st.markdown(
            """
            - **Recommendation Strength chart**: higher bars mean stronger final recommendation score.
            - **Score Components chart**: 
              - `User-CF` (purple) shows neighbor-user influence.
              - `Item-CF` (cyan) shows similarity-to-seen-items influence.
              - `Popularity` (amber) shows item demand across users.
            - **Network graph**:
              - Purple nodes are users; cyan nodes are items.
              - Larger nodes have more connections.
              - The selected user is emphasized with a larger marker.
            """
        )

    left, right = st.columns([2.1, 1.2], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">Top Recommendations</div></div>', unsafe_allow_html=True)
        if recs:
            rows = []
            for rank, rec in enumerate(recs, 1):
                rows.append(
                    {
                        "Rank": rank,
                        "Item": rec["item"],
                        "Score": rec["score"],
                        "User-CF": rec["user_cf"],
                        "Item-CF": rec["item_cf"],
                        "Popularity": rec["popularity"],
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

            ranking_df = pd.DataFrame(rows)[["Item", "Score"]].set_index("Item")
            st.markdown('<div class="card"><div class="card-title">Recommendation Strength</div></div>', unsafe_allow_html=True)
            st.bar_chart(ranking_df, y="Score", color="#22d3ee", use_container_width=True)

            components_df = pd.DataFrame(rows)[["Item", "User-CF", "Item-CF", "Popularity"]].set_index("Item")
            st.markdown('<div class="card"><div class="card-title">Score Components</div></div>', unsafe_allow_html=True)
            st.bar_chart(
                components_df,
                y=["User-CF", "Item-CF", "Popularity"],
                color=["#6366f1", "#22d3ee", "#f59e0b"],
                use_container_width=True,
            )
        else:
            st.info("No recommendations available for the selected user.")

    with right:
        st.markdown('<div class="card"><div class="card-title">Similarity Insights</div></div>', unsafe_allow_html=True)

        st.markdown(f"**Users similar to {target_user}**")
        if similar_users:
            st.dataframe(
                [{"User": u, "Similarity": round(score, 4)} for u, score in similar_users],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No similar users found.")

        st.markdown(f"**Items similar to {probe_item}**")
        if similar_items:
            st.dataframe(
                [{"Item": i, "Similarity": round(score, 4)} for i, score in similar_items],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No similar items found.")

    st.markdown('<div class="card"><div class="card-title">Interactive User-Item Network</div></div>', unsafe_allow_html=True)
    network_figure = _build_network_figure(engine, focus_user=target_user)
    st.plotly_chart(network_figure, use_container_width=True)


def run_cli_demo():
    interactions = default_interactions()

    engine = HybridRecommendationEngine(interactions)

    target_user = "U1"
    initial_recs, initial_latency = engine.recommend_realtime(target_user, top_n=4, neighbor_k=3)
    print("HYBRID RECOMMENDATION ENGINE PROTOTYPE")
    print("=" * 90)
    print("Mode: Development + Creation (live prototype)")
    print_recommendations(target_user, initial_recs, initial_latency)
    print_similarities(engine, user_id=target_user, probe_item="I2")

    print("\nApplying real-time interaction update: U1 rates I4 = 5.0")
    engine.add_interaction("U1", "I4", 5.0)

    updated_recs, updated_latency = engine.recommend_realtime(target_user, top_n=4, neighbor_k=3)
    print_recommendations(target_user, updated_recs, updated_latency)


def _is_running_in_streamlit():
    if st is None:
        return False

    try:
        from streamlit.runtime import exists

        if exists():
            return True
    except Exception:
        pass

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass

    if os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_RUNTIME"):
        return True

    return False


if __name__ == "__main__":
    if _is_running_in_streamlit():
        run_streamlit_app()
    else:
        print("CLI mode detected. To open the GUI run: streamlit run \"Recommender System.py\"")
        run_cli_demo()