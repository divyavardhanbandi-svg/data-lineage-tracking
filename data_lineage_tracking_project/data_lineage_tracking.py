"""
Data Lineage Tracking
---------------------
A small, dependency-light lineage tracker for pandas pipelines.

What it does
  1. Registers source datasets and records a fingerprint (rows, columns, hash)
  2. Records every transformation step: inputs -> job -> output
  3. Tracks column-level changes (added / dropped / kept columns)
  4. Answers lineage questions: upstream sources, downstream impact
  5. Exports a JSON lineage file, a text report and a graph image

Run:      python data_lineage_tracking.py
Install:  pip install pandas numpy networkx matplotlib
"""

import functools
import hashlib
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Core tracker
# --------------------------------------------------------------------------
class LineageTracker:
    """Keeps a directed graph: dataset -> job -> dataset."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.data = {}  # dataset name -> DataFrame (latest version)

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _fingerprint(df: pd.DataFrame) -> str:
        """Short content hash so changes in the data are detectable."""
        hashed = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.sha256(hashed.tobytes()).hexdigest()[:12]

    def _add_dataset(self, name: str, df: pd.DataFrame, kind: str, origin: str):
        self.data[name] = df
        self.graph.add_node(
            name,
            node_type="dataset",
            kind=kind,  # source | intermediate | output
            origin=origin,
            rows=int(len(df)),
            columns=list(map(str, df.columns)),
            fingerprint=self._fingerprint(df),
            created_at=self._now(),
        )

    # ---- registration ----------------------------------------------------
    def register_source(self, name: str, df: pd.DataFrame, origin: str = "in-memory"):
        """Register a raw input dataset (a file, table, API response...)."""
        self._add_dataset(name, df, kind="source", origin=origin)
        return df

    def run_step(self, job: str, inputs: list, output: str, func, is_output=False, **kwargs):
        """Run `func(*input_frames, **kwargs)` and record the lineage."""
        missing = [i for i in inputs if i not in self.data]
        if missing:
            raise KeyError(f"Unknown input dataset(s): {missing}")

        frames = [self.data[i] for i in inputs]
        result = func(*frames, **kwargs)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(f"Step '{job}' must return a DataFrame")

        # Column-level lineage: compare output columns to the union of inputs
        in_cols = set().union(*[set(map(str, f.columns)) for f in frames])
        out_cols = set(map(str, result.columns))

        self._add_dataset(
            output, result, kind="output" if is_output else "intermediate", origin=job
        )

        job_id = f"job:{job}"
        self.graph.add_node(
            job_id,
            node_type="job",
            label=job,
            function=getattr(func, "__name__", "lambda"),
            parameters={k: str(v) for k, v in kwargs.items()},
            columns_added=sorted(out_cols - in_cols),
            columns_dropped=sorted(in_cols - out_cols),
            columns_kept=sorted(in_cols & out_cols),
            rows_in=[int(len(f)) for f in frames],
            rows_out=int(len(result)),
            executed_at=self._now(),
        )
        for name in inputs:
            self.graph.add_edge(name, job_id)
        self.graph.add_edge(job_id, output)
        return result

    def track(self, job: str, inputs: list, output: str, is_output=False):
        """Decorator form of run_step. The function receives the input frames."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(**kwargs):
                return self.run_step(job, inputs, output, func, is_output, **kwargs)

            return wrapper

        return decorator

    # ---- queries ---------------------------------------------------------
    def _datasets_only(self, names):
        return sorted(n for n in names if self.graph.nodes[n]["node_type"] == "dataset")

    def upstream(self, dataset: str) -> list:
        """Every dataset this one was derived from (its ancestry)."""
        return self._datasets_only(nx.ancestors(self.graph, dataset))

    def downstream(self, dataset: str) -> list:
        """Every dataset that depends on this one (impact analysis)."""
        return self._datasets_only(nx.descendants(self.graph, dataset))

    def sources_of(self, dataset: str) -> list:
        """Only the raw source datasets behind a dataset."""
        return [d for d in self.upstream(dataset) if self.graph.nodes[d]["kind"] == "source"]

    def path(self, start: str, end: str) -> list:
        """Readable path of datasets/jobs between two datasets."""
        try:
            p = nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return []
        return [
            f"[job] {self.graph.nodes[n]['label']}"
            if self.graph.nodes[n]["node_type"] == "job" else n
            for n in p
        ]

    def column_origin(self, dataset: str, column: str) -> list:
        """Find the job that introduced a column, walking back from `dataset`."""
        trail = []
        for ds in [dataset] + self.upstream(dataset)[::-1]:
            job_id = f"job:{self.graph.nodes[ds]['origin']}"
            if job_id in self.graph and column in self.graph.nodes[job_id]["columns_added"]:
                trail.append((ds, self.graph.nodes[job_id]["label"]))
        return trail

    def detect_changes(self, name: str, df: pd.DataFrame) -> bool:
        """True if `df` differs from the version recorded in the lineage."""
        return self._fingerprint(df) != self.graph.nodes[name]["fingerprint"]

    # ---- exports ---------------------------------------------------------
    def to_json(self, path="lineage.json"):
        payload = {
            "generated_at": self._now(),
            "nodes": [{"id": n, **attrs} for n, attrs in self.graph.nodes(data=True)],
            "edges": [{"from": u, "to": v} for u, v in self.graph.edges()],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def report(self, path="lineage_report.txt") -> str:
        lines = ["DATA LINEAGE REPORT", "=" * 60]
        for n, a in self.graph.nodes(data=True):
            if a["node_type"] != "dataset":
                continue
            lines.append(f"\n[{a['kind'].upper()}] {n}  ({a['rows']} rows, {len(a['columns'])} cols)")
            lines.append(f"  origin     : {a['origin']}")
            lines.append(f"  fingerprint: {a['fingerprint']}")
            ups = self.upstream(n)
            lines.append(f"  upstream   : {', '.join(ups) if ups else '-'}")
            downs = self.downstream(n)
            lines.append(f"  downstream : {', '.join(downs) if downs else '-'}")
        lines.append("\n" + "=" * 60 + "\nJOBS")
        for n, a in self.graph.nodes(data=True):
            if a["node_type"] != "job":
                continue
            lines.append(f"\n{a['label']}  ({a['function']})")
            lines.append(f"  rows in -> out : {a['rows_in']} -> {a['rows_out']}")
            lines.append(f"  columns added  : {a['columns_added'] or '-'}")
            lines.append(f"  columns dropped: {a['columns_dropped'] or '-'}")
        text = "\n".join(lines)
        with open(path, "w") as f:
            f.write(text)
        return text

    def draw(self, path="lineage_graph.png"):
        """Left-to-right layered drawing of the lineage graph."""
        layers = list(nx.topological_generations(self.graph))
        pos = {}
        for x, layer in enumerate(layers):
            for y, node in enumerate(sorted(layer)):
                pos[node] = (x, -(y - (len(layer) - 1) / 2))

        colors, sizes, labels = [], [], {}
        palette = {"source": "#4C78A8", "intermediate": "#9ECAE9", "output": "#54A24B"}
        for n, a in self.graph.nodes(data=True):
            if a["node_type"] == "job":
                colors.append("#F58518")
                sizes.append(1800)
                labels[n] = a["label"]
            else:
                colors.append(palette[a["kind"]])
                sizes.append(2600)
                labels[n] = n

        fig, ax = plt.subplots(figsize=(14, 6))
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax, arrows=True, arrowsize=18,
            edge_color="#888888", node_size=2600, width=1.4,
        )
        nx.draw_networkx_nodes(self.graph, pos, ax=ax, node_color=colors, node_size=sizes)
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=7, ax=ax)

        from matplotlib.lines import Line2D

        legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=11, label=l)
            for l, c in [("Source", palette["source"]), ("Intermediate", palette["intermediate"]),
                         ("Output", palette["output"]), ("Job", "#F58518")]
        ]
        ax.legend(handles=legend, loc="upper left", frameon=False)
        ax.set_title("Data Lineage Graph", fontsize=13)
        ax.margins(0.12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()
        return path


# --------------------------------------------------------------------------
# Demo pipeline: orders + customers + products -> revenue reports
# --------------------------------------------------------------------------
def make_demo_data():
    rng = np.random.default_rng(42)
    customers = pd.DataFrame(
        {
            "customer_id": range(1, 51),
            "name": [f"Customer {i}" for i in range(1, 51)],
            "region": rng.choice(["North", "South", "East", "West"], 50),
        }
    )
    products = pd.DataFrame(
        {
            "product_id": range(1, 11),
            "category": rng.choice(["Electronics", "Home", "Toys"], 10),
            "unit_price": rng.uniform(5, 200, 10).round(2),
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": range(1, 301),
            "customer_id": rng.integers(1, 56, 300),  # some invalid ids on purpose
            "product_id": rng.integers(1, 11, 300),
            "quantity": rng.integers(-1, 6, 300),  # some invalid quantities
        }
    )
    orders.loc[rng.choice(300, 10, replace=False), "quantity"] = np.nan
    return orders, customers, products


# Transformation functions (plain pandas; the tracker records them)
def clean_orders(orders):
    df = orders.dropna(subset=["quantity"])
    df = df[df["quantity"] > 0].copy()
    df["quantity"] = df["quantity"].astype(int)
    return df


def enrich_orders(orders, customers, products):
    return (
        orders.merge(customers, on="customer_id", how="inner")
        .merge(products, on="product_id", how="inner")
        .assign(revenue=lambda d: (d["quantity"] * d["unit_price"]).round(2))
    )


def revenue_by_region(enriched):
    return (
        enriched.groupby("region", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )


def revenue_by_category(enriched):
    return (
        enriched.groupby("category", as_index=False)["revenue"].sum()
        .sort_values("revenue", ascending=False)
    )


def top_customers(enriched, n=5):
    return (
        enriched.groupby(["customer_id", "name"], as_index=False)["revenue"].sum()
        .nlargest(n, "revenue")
    )


def main():
    orders, customers, products = make_demo_data()
    t = LineageTracker()

    # 1. Register raw sources
    t.register_source("raw_orders", orders, origin="orders.csv")
    t.register_source("raw_customers", customers, origin="crm.customers")
    t.register_source("raw_products", products, origin="erp.products")

    # 2. Run the pipeline (each step is recorded automatically)
    t.run_step("clean_orders", ["raw_orders"], "clean_orders", clean_orders)
    t.run_step(
        "enrich_orders",
        ["clean_orders", "raw_customers", "raw_products"],
        "enriched_orders",
        enrich_orders,
    )
    t.run_step("agg_region", ["enriched_orders"], "report_revenue_region",
               revenue_by_region, is_output=True)
    t.run_step("agg_category", ["enriched_orders"], "report_revenue_category",
               revenue_by_category, is_output=True)
    t.run_step("rank_customers", ["enriched_orders"], "report_top_customers",
               top_customers, is_output=True, n=5)

    # Decorator style also works:
    @t.track("flag_big_orders", inputs=["enriched_orders"], output="big_orders", is_output=True)
    def flag_big_orders(df, threshold=500):
        return df[df["revenue"] >= threshold]

    flag_big_orders(threshold=500)

    # 3. Ask lineage questions
    print("\n=== LINEAGE QUERIES ===")
    print("Upstream of report_top_customers :", t.upstream("report_top_customers"))
    print("Raw sources behind that report   :", t.sources_of("report_top_customers"))
    print("If raw_customers changes, impacted:", t.downstream("raw_customers"))
    print("Path raw_orders -> big_orders    :", " -> ".join(t.path("raw_orders", "big_orders")))
    print("Where was 'revenue' created?     :", t.column_origin("report_revenue_region", "revenue"))

    # 4. Change detection
    changed = orders.copy()
    changed.loc[0, "quantity"] = 99
    print("raw_orders changed since lineage?:", t.detect_changes("raw_orders", changed))
    print("raw_orders unchanged             :", t.detect_changes("raw_orders", orders))

    # 5. Export
    t.to_json("lineage.json")
    t.report("lineage_report.txt")
    t.draw("lineage_graph.png")
    print("\nSaved: lineage.json, lineage_report.txt, lineage_graph.png")


if __name__ == "__main__":
    main()
