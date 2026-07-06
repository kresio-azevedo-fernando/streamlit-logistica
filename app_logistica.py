"""
Streamlit App — Logistics Optimisation
Simplex + Dijkstra · Live Interactive Demo
Author: Kresio Azevedo Fernando
Portfolio: kresio-azevedo-fernando.github.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import heapq
from scipy.optimize import linprog

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Logistics Optimisation · Kresio Fernando",
    page_icon="🏭",
    layout="wide",
)

st.markdown("""
<style>
body, .stApp { background-color: #09090f; color: #e8e8f0; }
.metric-card {
    background: #0f0f1a; border: 1px solid rgba(187,148,118,0.25);
    border-radius: 8px; padding: 1rem; text-align: center;
}
.metric-val { font-size: 1.8rem; font-weight: 700; color: #bb9476; }
.metric-lbl { font-size: 0.75rem; color: #9494a8; margin-top: 4px; }
h1, h2, h3 { color: #e8e8f0; }
.stButton > button {
    background: #bb9476; color: #000; font-weight: 700;
    border: none; border-radius: 4px;
}
.stButton > button:hover { background: #9d7a5e; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.title("🏭 Logistics Optimisation with Operations Research")
st.markdown(
    "**Simplex (Linear Programming) + Dijkstra (Graph Theory)** · "
    "[Portfolio](https://kresio-azevedo-fernando.github.io) · "
    "[LinkedIn](https://linkedin.com/in/kresio-bi-business-data-analyst)"
)
st.markdown("---")

# ── REAL DATA FROM DASHBOARD ─────────────────────────────────
CATEGORIES = ["Pharma", "Automotive", "Electronics", "Groceries", "Apparel"]
PRODUCTS    = [660, 658, 651, 618, 617]
OP_COST     = [1880, 1830, 1793, 1718, 1676]
STORAGE_COST= [120881668, 113858151, 120022890, 108707589, 102818635]
HANDLING    = [654493064, 615234722, 613455286, 556220335, 538852536]
DAILY_DEMAND= [16807, 17107, 16087, 15772, 15722]
FULFILLMENT = [561, 561, 554, 524, 523]

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📊 Dashboard KPIs", "⚙️ Simplex Optimisation", "🗺️ Dijkstra Routes"]
)

# ════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD KPIs
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Real Data — Warehouse KPIs (Power BI Dashboard)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-val">3,204</div><div class="metric-lbl">Total Products</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-val">85%</div><div class="metric-lbl">Service Level (target 94%)</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-val">14,746</div><div class="metric-lbl">Stockouts / Month</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-val">€12,246</div><div class="metric-lbl">Op Cost per Item</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    df_cat = pd.DataFrame({
        "Category":       CATEGORIES,
        "Products":       PRODUCTS,
        "Daily Demand":   DAILY_DEMAND,
        "Fulfillment":    FULFILLMENT,
        "Handling (€M)":  [round(h/1e6,1) for h in HANDLING],
        "Storage (€M)":   [round(s/1e6,1) for s in STORAGE_COST],
        "Op Cost/Item":   OP_COST,
    })

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cost by Category (€M)**")
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#09090f")
        ax.set_facecolor("#0f0f1a")
        ax.barh(df_cat["Category"], [h/1e6 for h in HANDLING],
                color="#bb9476", label="Handling")
        ax.barh(df_cat["Category"], [s/1e6 for s in STORAGE_COST],
                left=[h/1e6 for h in HANDLING], color="#6eb5ff", label="Storage")
        ax.legend(facecolor="#141420", labelcolor="#e8e8f0", fontsize=8)
        ax.tick_params(colors="#9494a8")
        ax.set_xlabel("Cost (€M)", color="#9494a8")
        for spine in ax.spines.values(): spine.set_color("#1a1a28")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Daily Demand vs Fulfillment**")
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#09090f")
        ax.set_facecolor("#0f0f1a")
        x = range(len(CATEGORIES))
        ax.bar([i-0.2 for i in x], DAILY_DEMAND, 0.35,
               color="#bb9476", label="Daily Demand")
        ax.bar([i+0.2 for i in x], FULFILLMENT, 0.35,
               color="#1D9E75", label="Fulfillment")
        ax.set_xticks(list(x))
        ax.set_xticklabels(CATEGORIES, rotation=15, fontsize=8, color="#9494a8")
        ax.tick_params(colors="#9494a8")
        ax.legend(facecolor="#141420", labelcolor="#e8e8f0", fontsize=8)
        for spine in ax.spines.values(): spine.set_color("#1a1a28")
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════
# TAB 2 — SIMPLEX
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader("⚙️ Simplex — Stock Allocation Optimisation")
    st.markdown("Adjust the parameters and click **Run Simplex** to calculate the optimal stock allocation.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Budget & Constraints**")
        total_budget = st.slider("Total stock budget (units)", 500000, 1200000, 844027, step=10000)
        target_sl    = st.slider("Target service level (%)", 85, 99, 94)
        st.markdown("**Min stock per category (units)**")
        min_stocks = []
        defaults = [100000, 100000, 95000, 90000, 90000]
        for cat, def_v in zip(CATEGORIES, defaults):
            v = st.number_input(f"{cat}", min_value=50000,
                                max_value=300000, value=def_v, step=5000)
            min_stocks.append(v)

    with col2:
        if st.button("🚀 Run Simplex Optimisation", use_container_width=True):
            # Stockout rate per category
            stockout_rate = np.array([
                1 - (f/d) for f, d in zip(FULFILLMENT, DAILY_DEMAND)
            ])
            cost_per_unit = np.array(OP_COST, dtype=float)

            # Objective: minimise cost - savings from reduced stockouts
            savings_per_unit = stockout_rate * 850
            net_cost = cost_per_unit - savings_per_unit

            # Constraints
            A_ub = np.ones((1, 5))
            b_ub = np.array([total_budget])
            bounds = [(min_stocks[i], total_budget) for i in range(5)]

            result = linprog(net_cost, A_ub=A_ub, b_ub=b_ub,
                             bounds=bounds, method="highs")

            if result.status == 0:
                optimal = result.x
                uniform = np.array([total_budget/5]*5)

                df_res = pd.DataFrame({
                    "Category":       CATEGORIES,
                    "Uniform (units)": uniform.astype(int),
                    "Optimal (units)": optimal.astype(int),
                    "Change":         (optimal - uniform).astype(int),
                    "Stockout Rate":  [f"{r*100:.1f}%" for r in stockout_rate],
                })

                st.markdown("**Simplex Result — Optimal vs Uniform Allocation**")
                st.dataframe(df_res, use_container_width=True)

                surplus_redist = abs((optimal - uniform)[optimal < uniform].sum())
                pct_redist = surplus_redist / total_budget * 100
                extra_impact = surplus_redist * 0.23 * 850 / 1e6

                r1, r2, r3 = st.columns(3)
                r1.metric("Stock redistributed", f"{surplus_redist:,.0f} units")
                r2.metric("% redistributed", f"{pct_redist:.1f}%")
                r3.metric("Additional impact", f"+€{extra_impact:.1f}M")

                fig, ax = plt.subplots(figsize=(8, 4), facecolor="#09090f")
                ax.set_facecolor("#0f0f1a")
                x = range(5)
                ax.bar([i-0.2 for i in x], uniform/1000, 0.35,
                       color="#2d2d3a", label="Uniform", edgecolor="#1a1a28")
                ax.bar([i+0.2 for i in x], optimal/1000, 0.35,
                       color="#bb9476", label="Optimal", edgecolor="#1a1a28")
                ax.set_xticks(list(x))
                ax.set_xticklabels(CATEGORIES, color="#9494a8")
                ax.tick_params(colors="#9494a8")
                ax.set_ylabel("Stock (K units)", color="#9494a8")
                ax.legend(facecolor="#141420", labelcolor="#e8e8f0")
                for spine in ax.spines.values(): spine.set_color("#1a1a28")
                ax.set_title("Stock Allocation — Uniform vs Simplex Optimal",
                             color="#e8e8f0")
                st.pyplot(fig)
                plt.close()

                st.success(f"✅ Simplex converged in {result.nit} iterations · Method: HiGHS")
            else:
                st.error(f"Solver error: {result.message}")

# ════════════════════════════════════════════════════════════
# TAB 3 — DIJKSTRA
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🗺️ Dijkstra — Warehouse Route Optimisation")
    st.markdown("Select origin and destination to calculate the minimum-distance picking path.")

    NODES = [
        "Receiving Dock", "Zone A — Pharma", "Zone B — Automotive",
        "Zone C — Electronics", "Zone D — Groceries", "Zone E — Apparel",
        "Staging Area", "Dispatch Dock 1", "Dispatch Dock 2", "Quality Control",
    ]
    EDGES = [
        ("Receiving Dock", "Zone A — Pharma", 45, 1.3),
        ("Receiving Dock", "Zone B — Automotive", 52, 1.2),
        ("Receiving Dock", "Zone C — Electronics", 38, 1.1),
        ("Receiving Dock", "Quality Control", 20, 1.0),
        ("Zone A — Pharma", "Zone B — Automotive", 30, 1.4),
        ("Zone A — Pharma", "Staging Area", 60, 1.2),
        ("Zone A — Pharma", "Zone D — Groceries", 55, 1.1),
        ("Zone B — Automotive", "Zone C — Electronics", 25, 1.3),
        ("Zone B — Automotive", "Staging Area", 48, 1.1),
        ("Zone C — Electronics", "Zone D — Groceries", 35, 1.0),
        ("Zone C — Electronics", "Staging Area", 42, 1.2),
        ("Zone D — Groceries", "Zone E — Apparel", 28, 1.0),
        ("Zone D — Groceries", "Staging Area", 50, 1.1),
        ("Zone E — Apparel", "Staging Area", 38, 1.0),
        ("Staging Area", "Dispatch Dock 1", 15, 1.5),
        ("Staging Area", "Dispatch Dock 2", 18, 1.4),
        ("Quality Control", "Zone A — Pharma", 35, 1.0),
        ("Quality Control", "Staging Area", 55, 1.0),
    ]

    def build_graph():
        G = nx.Graph()
        G.add_nodes_from(NODES)
        for src, dst, dist, cong in EDGES:
            G.add_edge(src, dst, weight=round(dist*cong, 1),
                       distance=dist, congestion=cong)
        return G

    def dijkstra_manual(G, source):
        dist = {n: float("inf") for n in G.nodes}
        prev = {n: None for n in G.nodes}
        dist[source] = 0
        heap = [(0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]: continue
            for v, data in G[u].items():
                alt = dist[u] + data["weight"]
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))
        return dist, prev

    def reconstruct(prev, target):
        path, node = [], target
        while node:
            path.append(node)
            node = prev[node]
        return list(reversed(path))

    G = build_graph()

    col1, col2 = st.columns([1, 2])
    with col1:
        origin = st.selectbox("Origin", NODES, index=0)
        destination = st.selectbox("Destination", NODES, index=6)
        congestion_factor = st.slider("Congestion multiplier", 0.5, 2.0, 1.0, 0.1)

        if st.button("🔍 Calculate Optimal Route", use_container_width=True):
            if origin == destination:
                st.warning("Origin and destination must be different.")
            else:
                # Apply congestion factor
                G2 = build_graph()
                for u, v, data in G2.edges(data=True):
                    G2[u][v]["weight"] = round(data["distance"] * data["congestion"] * congestion_factor, 1)

                dist_map, prev_map = dijkstra_manual(G2, origin)
                optimal_path = reconstruct(prev_map, destination)
                optimal_dist = dist_map[destination]

                # Sequential (random) comparison
                try:
                    all_nodes = [n for n in NODES if n not in [origin, destination]]
                    random_path = [origin] + all_nodes[:3] + [destination]
                    random_dist = sum(
                        G2[random_path[i]][random_path[i+1]]["weight"]
                        if G2.has_edge(random_path[i], random_path[i+1])
                        else nx.shortest_path_length(G2, random_path[i], random_path[i+1], weight="weight")
                        for i in range(len(random_path)-1)
                    )
                except Exception:
                    random_dist = optimal_dist * 1.4

                saving_pct = (random_dist - optimal_dist) / random_dist * 100 if random_dist > 0 else 0
                fuel_saving = max(random_dist - optimal_dist, 0) * 0.18

                st.session_state["path"]        = optimal_path
                st.session_state["dist"]        = optimal_dist
                st.session_state["saving_pct"]  = saving_pct
                st.session_state["fuel_saving"] = fuel_saving
                st.session_state["G"]           = G2

    with col2:
        if "path" in st.session_state:
            path = st.session_state["path"]
            dist = st.session_state["dist"]
            G2   = st.session_state["G"]

            m1, m2, m3 = st.columns(3)
            m1.metric("Optimal distance", f"{dist:.1f}m")
            m2.metric("Distance saving", f"{st.session_state['saving_pct']:.1f}%")
            m3.metric("Fuel saving/route", f"€{st.session_state['fuel_saving']:.2f}")

            st.markdown(f"**Optimal path:** {' → '.join(path)}")

            # Draw graph
            path_edges = set()
            for i in range(len(path)-1):
                path_edges.add((path[i], path[i+1]))
                path_edges.add((path[i+1], path[i]))

            pos = nx.spring_layout(G2, seed=42, k=2.5)
            node_colors = []
            for node in G2.nodes:
                if node == origin:            node_colors.append("#1D9E75")
                elif node == destination:     node_colors.append("#bb9476")
                elif node in path:            node_colors.append("#c8a96e")
                elif node == "Quality Control": node_colors.append("#6eb5ff")
                else:                         node_colors.append("#2d2d3a")

            edge_colors = ["#ff4444" if (u,v) in path_edges or (v,u) in path_edges
                           else "#444455" for u,v in G2.edges]
            edge_widths = [3.5 if (u,v) in path_edges or (v,u) in path_edges
                           else 1.2 for u,v in G2.edges]

            fig, ax = plt.subplots(figsize=(8, 5), facecolor="#09090f")
            ax.set_facecolor("#09090f")
            nx.draw_networkx_nodes(G2, pos, node_color=node_colors,
                                   node_size=700, ax=ax)
            nx.draw_networkx_labels(G2, pos, font_size=6,
                                    font_color="white", ax=ax)
            nx.draw_networkx_edges(G2, pos, edge_color=edge_colors,
                                   width=edge_widths, ax=ax)
            edge_labels = {(u,v): f"{d['weight']}m"
                           for u, v, d in G2.edges(data=True)}
            nx.draw_networkx_edge_labels(G2, pos, edge_labels=edge_labels,
                                         font_size=5, font_color="#9494a8", ax=ax)
            legend = [
                mpatches.Patch(color="#1D9E75", label="Origin"),
                mpatches.Patch(color="#bb9476", label="Destination"),
                mpatches.Patch(color="#c8a96e", label="Path node"),
                mpatches.Patch(color="#ff4444", label="Optimal path"),
            ]
            ax.legend(handles=legend, loc="upper left",
                      facecolor="#141420", labelcolor="white", fontsize=7)
            ax.set_title("Warehouse Graph — Dijkstra Optimal Path",
                         color="#e8e8f0", fontsize=11)
            st.pyplot(fig)
            plt.close()

st.markdown("---")
st.markdown(
    "**Kresio Azevedo Fernando** · BI & Decision Optimisation Specialist · "
    "[kresio-azevedo-fernando.github.io](https://kresio-azevedo-fernando.github.io)"
)
