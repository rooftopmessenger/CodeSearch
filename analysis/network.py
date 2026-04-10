"""
analysis/network.py — Verse-Year connection graph for CodeSearch-Ultra.

Builds a weighted bipartite NetworkX graph from ELS search results where:
  - Left nodes  = verse references (e.g. "Genesis 1:1")
  - Right nodes = Hebrew year codes found in those verses (e.g. "תשפו")
  - Edge weight = consensus semantic score

The graph enables:
  - Clustering of thematically related verses across multiple search runs.
  - Identification of verses that appear repeatedly across different year anchors
    (high-betweenness nodes — the strongest "convergence points").
  - Export of the adjacency matrix as a Polars DataFrame for CSV / Parquet
    archiving.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import networkx as nx
import polars as pl

if TYPE_CHECKING:
    from engine.search import ConsensusScore

# ── backend path resolution ───────────────────────────────────────────────────
# Use importlib to avoid the 'engine' package shadowing backend/engine.py.
_BACKEND = Path(__file__).resolve().parent.parent / "backend"


def _load_backend(name: str):
    spec = _ilu.spec_from_file_location(f"_backend_{name}", _BACKEND / f"{name}.py")
    mod  = _ilu.module_from_spec(spec)   # type: ignore[arg-type]
    spec.loader.exec_module(mod)         # type: ignore[union-attr]
    return mod


Match = _load_backend("engine").Match

# ── Chronos Anchors (mirrors engine/search.py) ────────────────────────────────
_CHRONOS_ANCHORS: tuple[str, ...] = ("תשפו", "תשצ", "תשצו")


class VerseYearNetwork:
    """
    Bipartite graph mapping verse references to year-code nodes.

    Usage
    -----
    >>> net = VerseYearNetwork()
    >>> net.ingest(results)          # list[(Match, ConsensusScore)]
    >>> net.ingest(more_results)     # incremental — safe to call multiple times
    >>> top = net.top_convergences(n=10)
    >>> net.export_adjacency("run_69_network.csv")
    """

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        results: Sequence[tuple[Match, ConsensusScore]],
    ) -> None:
        """
        Add a batch of (Match, ConsensusScore) pairs to the graph.

        Nodes
        -----
        - Verse node id:   ``"verse:<verse_ref>"``    type="verse"
        - Year-code id:    ``"year:<word>"``           type="year_code"
        - Word node id:    ``"word:<word>"``           type="word"

        Edges carry ``weight=consensus_score`` and ``is_decadal_anchor``.
        """
        for m, cs in results:
            v_id = f"verse:{m.verse}"
            w_id = f"word:{m.word}"

            # Add / update verse node
            if not self._graph.has_node(v_id):
                self._graph.add_node(v_id, type="verse", book=m.book)

            # Add / update word node
            if not self._graph.has_node(w_id):
                self._graph.add_node(w_id, type="word")

            # Add / update verse-word edge
            if self._graph.has_edge(v_id, w_id):
                prev = self._graph[v_id][w_id]["weight"]
                self._graph[v_id][w_id]["weight"] = max(prev, cs.consensus)
                if cs.is_decadal_anchor:
                    self._graph[v_id][w_id]["is_decadal_anchor"] = True
            else:
                self._graph.add_edge(
                    v_id, w_id,
                    weight=cs.consensus,
                    hebert_score=cs.hebert_score,
                    alephbert_score=cs.alephbert_score,
                    is_decadal_anchor=cs.is_decadal_anchor,
                    skip=m.skip,
                    start=m.start,
                )

            # Link year codes found in the sequence to the year-code node layer
            for anchor in _CHRONOS_ANCHORS:
                if anchor in m.sequence or anchor in m.word:
                    y_id = f"year:{anchor}"
                    if not self._graph.has_node(y_id):
                        self._graph.add_node(y_id, type="year_code", anchor=anchor)
                    if not self._graph.has_edge(v_id, y_id):
                        self._graph.add_edge(
                            v_id, y_id,
                            weight=cs.consensus,
                            is_decadal_anchor=True,
                        )

    # ── Analysis ──────────────────────────────────────────────────────────────

    def top_convergences(self, n: int = 10) -> pl.DataFrame:
        """
        Return the *n* verse nodes with the highest weighted degree (i.e. the
        verses that are semantically linked to the most search terms with the
        heaviest consensus scores).

        Returns a Polars DataFrame with columns:
            verse, book, weighted_degree, degree, is_decadal_anchor
        """
        rows: list[dict] = []
        for node, data in self._graph.nodes(data=True):
            if data.get("type") != "verse":
                continue
            nbrs = self._graph[node]
            w_deg = sum(d.get("weight", 0.0) for d in nbrs.values())
            deg   = len(nbrs)
            decadal = any(d.get("is_decadal_anchor", False) for d in nbrs.values())
            rows.append({
                "verse":             node.removeprefix("verse:"),
                "book":              data.get("book", ""),
                "weighted_degree":   w_deg,
                "degree":            deg,
                "is_decadal_anchor": decadal,
            })

        df = pl.DataFrame(rows)
        if df.is_empty():
            return df
        return df.sort("weighted_degree", descending=True).head(n)

    def betweenness_centrality(self) -> pl.DataFrame:
        """
        Compute betweenness centrality for all nodes.

        High-centrality verse nodes are structural bridges between multiple
        search-term clusters — the primary "convergence epicentres."

        Returns a Polars DataFrame with columns: node, type, centrality
        """
        scores = nx.betweenness_centrality(self._graph, weight="weight")
        rows = [
            {
                "node":       n,
                "type":       self._graph.nodes[n].get("type", "unknown"),
                "centrality": v,
            }
            for n, v in scores.items()
        ]
        df = pl.DataFrame(rows)
        if df.is_empty():
            return df
        return df.sort("centrality", descending=True)

    def subgraph_for_book(self, book: str) -> nx.Graph:
        """Return the induced subgraph restricted to verse-nodes from *book*."""
        keep = {
            n for n, d in self._graph.nodes(data=True)
            if d.get("book") == book or d.get("type") != "verse"
        }
        return self._graph.subgraph(keep).copy()

    # ── Persistence ───────────────────────────────────────────────────────────

    def export_adjacency(self, path: str | Path) -> Path:
        """
        Write the edge list as a CSV with columns:
            source, target, weight, hebert_score, alephbert_score,
            is_decadal_anchor, skip, start

        Returns the resolved Path that was written.
        """
        path = Path(path)
        rows = [
            {
                "source":            u,
                "target":            v,
                "weight":            d.get("weight", 0.0),
                "hebert_score":      d.get("hebert_score", 0.0),
                "alephbert_score":   d.get("alephbert_score", 0.0),
                "is_decadal_anchor": d.get("is_decadal_anchor", False),
                "skip":              d.get("skip", 0),
                "start":             d.get("start", 0),
            }
            for u, v, d in self._graph.edges(data=True)
        ]
        df = pl.DataFrame(rows)
        df.write_csv(path)
        return path

    def export_graphml(self, path: str | Path) -> Path:
        """Write the full graph as GraphML for Gephi / Cytoscape visualisation."""
        path = Path(path)
        nx.write_graphml(self._graph, str(path))
        return path

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH ANALYZER — PyVis Force-Directed Hub
# ══════════════════════════════════════════════════════════════════════════════

class GraphAnalyzer:
    """
    Force-directed network visualizer built on ``VerseYearNetwork`` + PyVis.

    Converts a ``VerseYearNetwork`` graph into an interactive HTML component
    for Streamlit embedding.  Nodes are sized by weighted degree; the תשצ
    (2030 Wall) gravity-well is highlighted in red when detected.

    Usage
    -----
    >>> net = VerseYearNetwork()
    >>> net.ingest(results)
    >>> ga  = GraphAnalyzer(net)
    >>> html_path = ga.render(output_path="docs/network_hub.html")
    >>> st.components.v1.html(open(html_path).read(), height=700, scrolling=True)
    """

    # 2030 Wall anchor — identified as primary gravity-well node
    _GRAVITY_WELL = "year:תשצ"

    # Color palette
    _COLOR_VERSE   = "#4FC3F7"   # light blue  — verse nodes
    _COLOR_WORD    = "#FF6B35"   # orange       — search-term nodes
    _COLOR_YEAR    = "#A5D6A7"   # green        — year-code nodes
    _COLOR_GRAVITY = "#EF5350"   # red          — 2030 gravity well

    def __init__(self, network: VerseYearNetwork) -> None:
        self._net = network

    def render(
        self,
        output_path: str | Path = "docs/network_hub.html",
        *,
        height: str = "700px",
        bgcolor: str = "#1A1A2E",
        font_color: str = "#E0E0E0",
        gravity: float = -12_000,
        spring_length: int = 180,
    ) -> Path:
        """
        Render the force-directed graph to a self-contained HTML file.

        Parameters
        ----------
        output_path : Path
            Destination file.  Parent directories are created if absent.
        height : str
            PyVis canvas height (CSS value, default "700px").
        bgcolor : str
            Canvas background colour.
        font_color : str
            Default node label colour.
        gravity : float
            Barnes-Hut gravitational constant (more negative → tighter clusters).
        spring_length : int
            Default spring length for edges.

        Returns
        -------
        Path
            The resolved path to the written HTML file.
        """
        try:
            from pyvis.network import Network  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyvis is required for GraphAnalyzer.render().  "
                "Run: uv add pyvis"
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        g: nx.Graph = self._net._graph

        # Compute weighted degree for node sizing
        w_degrees: dict[str, float] = {}
        for n in g.nodes():
            w_degrees[n] = sum(d.get("weight", 0.0) for d in g[n].values())

        max_w = max(w_degrees.values(), default=1.0) or 1.0

        pv = Network(
            height=height,
            width="100%",
            bgcolor=bgcolor,
            font_color=font_color,
            directed=False,
        )
        pv.barnes_hut(
            gravity=int(gravity),
            central_gravity=0.3,
            spring_length=spring_length,
            spring_strength=0.04,
            damping=0.09,
        )

        for node_id, data in g.nodes(data=True):
            ntype    = data.get("type", "unknown")
            is_gwell = node_id == self._GRAVITY_WELL

            size  = 10 + 30 * (w_degrees.get(node_id, 0.0) / max_w)
            color = (
                self._COLOR_GRAVITY if is_gwell else
                self._COLOR_YEAR    if ntype == "year_code" else
                self._COLOR_WORD    if ntype == "word" else
                self._COLOR_VERSE
            )
            label = node_id.split(":", 1)[-1]
            title = (
                f"<b>{label}</b><br>"
                f"Type: {ntype}<br>"
                f"Weighted degree: {w_degrees.get(node_id, 0.0):.3f}"
                + (" <b>[2030 GRAVITY WELL]</b>" if is_gwell else "")
            )
            pv.add_node(
                node_id,
                label=label,
                color=color,
                size=float(size),
                title=title,
                font={"size": 11, "color": font_color},
            )

        for u, v, data in g.edges(data=True):
            w = data.get("weight", 0.1)
            is_anchor = data.get("is_decadal_anchor", False)
            pv.add_edge(
                u, v,
                value=float(w),
                color="#EF5350" if is_anchor else "#546E7A",
                title=f"consensus={w:.3f}" + (" ★ ANCHOR" if is_anchor else ""),
            )

        # Write inline HTML (no external CDN calls, fully self-contained)
        pv.save_graph(str(output_path))
        return output_path

    def gravity_well_report(self) -> dict:
        """
        Analyse the 2030 Wall node (תשצ) as a gravity well.

        Returns a dict with:
          - ``degree``: raw number of connected verses/words
          - ``weighted_degree``: sum of edge consensus weights
          - ``neighbours``: list of connected node ids sorted by weight desc
          - ``is_dominant``: True when weighted_degree is top-1 among all year nodes
        """
        g = self._net._graph
        if not g.has_node(self._GRAVITY_WELL):
            return {
                "degree": 0,
                "weighted_degree": 0.0,
                "neighbours": [],
                "is_dominant": False,
            }

        nbrs = g[self._GRAVITY_WELL]
        wd = sum(d.get("weight", 0.0) for d in nbrs.values())
        sorted_nbrs = sorted(
            nbrs.items(),
            key=lambda kv: kv[1].get("weight", 0.0),
            reverse=True,
        )

        # Compare against other year-code nodes
        all_year_wd = {
            n: sum(d.get("weight", 0.0) for d in g[n].values())
            for n, data in g.nodes(data=True)
            if data.get("type") == "year_code"
        }
        is_dominant = (
            bool(all_year_wd) and
            max(all_year_wd.values()) == wd and
            wd > 0
        )

        return {
            "degree": len(nbrs),
            "weighted_degree": wd,
            "neighbours": [nid for nid, _ in sorted_nbrs],
            "is_dominant": is_dominant,
        }
