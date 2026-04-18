"""
Hazard-Aware Routing — Static Matplotlib Visualization (v3).

Writes two PNG artifacts under `data/viz/`:
  * edge_stats.png  — 2x3 panel of edge-level hazard statistics
  * hazard_map.png  — static map of the road network colored by combined
                      hazard, with NOAH flood / landslide polygons overlaid

Failure semantics (see approved plan):
  * Missing graphml              → hard error, exit 1
  * Missing preprocessing manifest → warn-skip, compute blocking counts from graph
  * Missing NOAH shapefile(s)    → warn-skip polygon overlay, map still renders
  * Missing required edge attrs  → hard error with edge id

Usage:
    python -m src.data.visualize_hazards
    python -m src.data.visualize_hazards --out-dir data/viz
    python -m src.data.visualize_hazards --skip-map
    python -m src.data.visualize_hazards --skip-stats
    python -m src.data.visualize_hazards --debug
"""

from __future__ import annotations

# Configure matplotlib backend before any pyplot import — headless-safe on Windows.
import matplotlib

matplotlib.use("Agg")

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import box


# =============================================================================
# CONFIG — stay in lockstep with src/data/prepare_data.py
# =============================================================================

DEFAULT_GRAPH_PATH = "data/la_trinidad_hazard_graph.graphml"
DEFAULT_MANIFEST_PATH = "data/preprocessing_manifest.json"
DEFAULT_OUT_DIR = "data/viz"
DEFAULT_FLOOD_SHP = "data/raw/noah/flood_hazard/Benguet_Flood_25year.shp"
DEFAULT_LS_SHP = "data/raw/noah/landslide_susceptibility/Benguet_LandslideHazards.shp"

FLOOD_CLASS_COLORS = {0: "#cccccc", 1: "#6baed6", 2: "#3182bd", 3: "#08306b"}
LANDSLIDE_CLASS_COLORS = {0: "#cccccc", 1: "#fdd835", 2: "#fb8c00", 3: "#c62828"}

DET_RI_THRESHOLDS: Mapping[int, Mapping[str, float]] = {
    1: {"flood_block_threshold": 1.1, "landslide_block_threshold": 1.1},
    2: {"flood_block_threshold": 1.0, "landslide_block_threshold": 1.1},
    3: {"flood_block_threshold": 0.6, "landslide_block_threshold": 1.1},
    4: {"flood_block_threshold": 0.6, "landslide_block_threshold": 1.0},
    5: {"flood_block_threshold": 0.6, "landslide_block_threshold": 0.6},
}

# Canonical-3 mapping (must match prepare_data.py).
CANONICAL_FLOOD = {1: 0.2, 2: 0.6, 3: 1.0}
CANONICAL_LS = {1: 0.2, 2: 0.6, 3: 1.0}

REQUIRED_EDGE_ATTRS = (
    "flood_score", "landslide_score",
    "flood_class", "landslide_class",
    "length", "base_time", "sample_count",
)

HAZARD_CMAP = LinearSegmentedColormap.from_list(
    "hazard_g2r",
    [
        (0.00, (76 / 255, 175 / 255, 80 / 255)),   # green
        (0.33, (255 / 255, 235 / 255, 59 / 255)),  # yellow
        (0.66, (251 / 255, 140 / 255, 0 / 255)),   # orange
        (1.00, (198 / 255, 40 / 255, 40 / 255)),   # red
    ],
    N=256,
)

logger = logging.getLogger("visualize_hazards")


# =============================================================================
# EDGE ARRAYS — vectorized storage for all plots
# =============================================================================


@dataclass(frozen=True)
class EdgeArrays:
    flood_score: np.ndarray      # (N,)
    landslide_score: np.ndarray
    flood_class: np.ndarray
    landslide_class: np.ndarray
    length: np.ndarray
    base_time: np.ndarray
    sample_count: np.ndarray
    combined: np.ndarray         # (fs + ls) / 2, cached
    segments: np.ndarray         # (N, 2, 2): [[x1, y1], [x2, y2]] per edge

    @property
    def n(self) -> int:
        return int(self.flood_score.shape[0])


def road_bbox(
    arrays: EdgeArrays, pad_frac: float = 0.05
) -> tuple[float, float, float, float]:
    """Return (min_x, min_y, max_x, max_y) of the road network, padded by pad_frac."""
    xs = arrays.segments[:, :, 0]
    ys = arrays.segments[:, :, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    px = (x_max - x_min) * pad_frac
    py = (y_max - y_min) * pad_frac
    return (x_min - px, y_min - py, x_max + px, y_max + py)


# =============================================================================
# LOAD
# =============================================================================


def load_graph(path: Path) -> tuple[EdgeArrays, dict[str, tuple[float, float]]]:
    """Read graphml, validate required attrs, build vectorized EdgeArrays."""
    if not path.exists():
        raise FileNotFoundError(
            f"graphml not found at {path}. "
            f"Run `python -m src.data.prepare_data` first."
        )
    logger.info(f"Loading graphml: {path}")
    G = nx.read_graphml(str(path))
    logger.info(
        f"  pipeline_version={G.graph.get('pipeline_version', '?')}  "
        f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}"
    )

    node_xy: dict[str, tuple[float, float]] = {}
    for node, data in G.nodes(data=True):
        try:
            x = float(data["x"])
            y = float(data["y"])
        except (KeyError, ValueError, TypeError):
            pos = str(data.get("pos", ""))
            try:
                x_s, y_s = pos.split(",")
                x, y = float(x_s), float(y_s)
            except Exception as exc:
                raise ValueError(
                    f"node {node} missing valid x/y and pos attrs"
                ) from exc
        node_xy[str(node)] = (x, y)

    n = G.number_of_edges()
    fs = np.zeros(n, dtype=np.float64)
    ls = np.zeros(n, dtype=np.float64)
    fc = np.zeros(n, dtype=np.int64)
    lc = np.zeros(n, dtype=np.int64)
    lg = np.zeros(n, dtype=np.float64)
    bt = np.zeros(n, dtype=np.float64)
    sc = np.zeros(n, dtype=np.int64)
    segments = np.zeros((n, 2, 2), dtype=np.float64)

    for i, (u, v, data) in enumerate(G.edges(data=True)):
        u_s, v_s = str(u), str(v)
        if u_s not in node_xy or v_s not in node_xy:
            raise ValueError(f"edge {u_s}->{v_s} references unknown node")
        missing = [a for a in REQUIRED_EDGE_ATTRS if a not in data]
        if missing:
            raise ValueError(f"edge {u_s}->{v_s} missing required attrs: {missing}")
        fs[i] = float(data["flood_score"])
        ls[i] = float(data["landslide_score"])
        fc[i] = int(float(data["flood_class"]))
        lc[i] = int(float(data["landslide_class"]))
        lg[i] = float(data["length"])
        bt[i] = float(data["base_time"])
        sc[i] = int(float(data["sample_count"]))
        x1, y1 = node_xy[u_s]
        x2, y2 = node_xy[v_s]
        segments[i, 0] = (x1, y1)
        segments[i, 1] = (x2, y2)

    arrays = EdgeArrays(
        flood_score=fs, landslide_score=ls,
        flood_class=fc, landslide_class=lc,
        length=lg, base_time=bt, sample_count=sc,
        combined=(fs + ls) / 2.0,
        segments=segments,
    )
    logger.info(f"  built EdgeArrays: {arrays.n:,} edges, combined mean={arrays.combined.mean():.3f}")
    return arrays, node_xy


def load_manifest(path: Path) -> Optional[dict]:
    if not path.exists():
        logger.warning(f"  manifest not found at {path}; blocking chart uses graph-computed counts")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        logger.info(f"  loaded manifest: {path} (pipeline_version={m.get('pipeline_version', '?')})")
        return m
    except Exception as exc:
        logger.warning(f"  failed to read manifest ({exc}); blocking chart uses graph-computed counts")
        return None


def load_shapefiles(
    flood_path: Path,
    ls_path: Path,
    flood_column: str,
    ls_column: str,
    bbox: tuple[float, float, float, float],
) -> tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
    """
    Load NOAH flood / landslide shapefiles, clip to the road bbox, then simplify.

    Clipping is the biggest perf win: NOAH shapefiles are province-wide
    MultiPolygons with millions of vertices but we only plot the La Trinidad
    road bbox. pyogrio engine with a column subset keeps read time small.
    """
    clip_poly = box(*bbox)

    def _read(path: Path, label: str, column: str) -> Optional[gpd.GeoDataFrame]:
        if not path.exists():
            logger.warning(
                f"  {label} shapefile not found at {path}; skipping that polygon overlay"
            )
            return None
        t0 = time.perf_counter()
        try:
            gdf = gpd.read_file(
                str(path), engine="pyogrio", columns=[column]
            )
        except Exception:
            # Fall back to default engine if pyogrio rejects the columns arg.
            gdf = gpd.read_file(str(path))
        t_read = time.perf_counter() - t0

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        elif str(gdf.crs) != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        t1 = time.perf_counter()
        gdf = gdf.clip(clip_poly)
        t_clip = time.perf_counter() - t1

        t2 = time.perf_counter()
        # Topology-naive simplify is much faster on huge multipolygons and the
        # visual effect at ~5e-4 degrees (~55 m) is imperceptible at this zoom.
        gdf["geometry"] = gdf["geometry"].simplify(
            0.0005, preserve_topology=False
        )
        # Drop rows whose geometry collapsed to empty after simplification.
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(
            drop=True
        )
        t_simp = time.perf_counter() - t2

        logger.info(
            f"  loaded {label} shapefile: {len(gdf)} polygons "
            f"(read={t_read:.2f}s clip={t_clip:.2f}s simplify={t_simp:.2f}s)"
        )
        return gdf

    return (
        _read(flood_path, "flood", flood_column),
        _read(ls_path, "landslide", ls_column),
    )


# =============================================================================
# PLOT: EDGE STATS (2x3 panels)
# =============================================================================


def _bar_class_counts(
    ax: plt.Axes,
    class_arr: np.ndarray,
    colors: Mapping[int, str],
    title: str,
    xlabel: str,
    x_labels: list[str],
) -> None:
    classes = [0, 1, 2, 3]
    counts = [int((class_arr == c).sum()) for c in classes]
    bars = ax.bar(
        classes, counts,
        color=[colors[c] for c in classes],
        edgecolor="black", linewidth=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("# edges")
    ax.set_xticks(classes)
    ax.set_xticklabels(x_labels)
    y_headroom = max(counts) * 0.02 if counts else 0.0
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            c + y_headroom,
            f"{c:,}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.text(
        0.98, 0.98, f"n={int(class_arr.size):,}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, alpha=0.6,
    )


def plot_edge_stats(arrays: EdgeArrays, manifest: Optional[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    _bar_class_counts(
        axes[0, 0], arrays.flood_class, FLOOD_CLASS_COLORS,
        "Flood class distribution", "class (Var)",
        ["0\n(none)", "1\nLow", "2\nMod", "3\nHigh"],
    )
    _bar_class_counts(
        axes[0, 1], arrays.landslide_class, LANDSLIDE_CLASS_COLORS,
        "Landslide class distribution", "class (LH)",
        ["0\n(none)", "1\nLow", "2\nMod", "3\nHigh"],
    )

    # Combined hazard score
    ax = axes[0, 2]
    ax.hist(arrays.combined, bins=30, color="#c62828", edgecolor="black", linewidth=0.3)
    ax.set_title("Combined hazard score  (fs + ls) / 2")
    ax.set_xlabel("score")
    ax.set_ylabel("# edges")
    mean = float(arrays.combined.mean())
    ax.axvline(mean, color="black", ls="--", lw=1)
    ax.text(
        mean + 0.01, ax.get_ylim()[1] * 0.92,
        f"mean = {mean:.3f}", fontsize=8,
    )

    # Edge length (log-x)
    ax = axes[1, 0]
    lmax = float(arrays.length.max())
    bins = np.logspace(0, np.log10(lmax + 1.0), 40)
    ax.hist(arrays.length, bins=bins, color="#3182bd", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_title("Edge length distribution")
    ax.set_xlabel("length (m, log)")
    ax.set_ylabel("# edges")
    ax.text(
        0.98, 0.98,
        f"median={np.median(arrays.length):.0f} m\n"
        f"mean={arrays.length.mean():.0f} m\n"
        f"max={lmax:.0f} m",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    # Sample count per edge
    ax = axes[1, 1]
    ax.hist(arrays.sample_count, bins=30, color="#fb8c00", edgecolor="black", linewidth=0.3)
    ax.set_title("Sample count per edge  (10m spacing, ≥3)")
    ax.set_xlabel("# samples")
    ax.set_ylabel("# edges")
    ax.text(
        0.98, 0.98,
        f"min={int(arrays.sample_count.min())}\n"
        f"median={int(np.median(arrays.sample_count))}\n"
        f"max={int(arrays.sample_count.max())}",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
    )

    # Blocking eligibility by RI (prefer manifest, fall back to on-the-fly)
    ax = axes[1, 2]
    ris = [1, 2, 3, 4, 5]
    bp = manifest.get("blocking_preview") if manifest else None
    if bp:
        flood_eligible = [int(bp[f"RI{r}"]["flood_eligible"]) for r in ris]
        ls_eligible = [int(bp[f"RI{r}"]["landslide_eligible"]) for r in ris]
        union_eligible = [int(bp[f"RI{r}"]["union_eligible"]) for r in ris]
    else:
        flood_eligible, ls_eligible, union_eligible = [], [], []
        for r in ris:
            tf = DET_RI_THRESHOLDS[r]["flood_block_threshold"]
            tl = DET_RI_THRESHOLDS[r]["landslide_block_threshold"]
            f_mask = arrays.flood_score >= tf
            l_mask = arrays.landslide_score >= tl
            flood_eligible.append(int(f_mask.sum()))
            ls_eligible.append(int(l_mask.sum()))
            union_eligible.append(int((f_mask | l_mask).sum()))

    x = np.arange(len(ris))
    w = 0.28
    ax.bar(x - w, flood_eligible, w, color="#3182bd", label="flood", edgecolor="black", linewidth=0.3)
    ax.bar(x,     ls_eligible,    w, color="#c62828", label="landslide", edgecolor="black", linewidth=0.3)
    ax.bar(x + w, union_eligible, w, color="#424242", label="union", edgecolor="black", linewidth=0.3)
    ax.set_title("Blocking eligibility by RI")
    ax.set_xlabel("rainfall intensity")
    ax.set_ylabel("# eligible edges")
    ax.set_xticks(x)
    ax.set_xticklabels([f"RI{r}" for r in ris])
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "La Trinidad road network — edge hazard statistics  (canonical-3, v3 pipeline)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024.0
    logger.info(f"  wrote {out_path} ({size_kb:.1f} KB)")


# =============================================================================
# PLOT: HAZARD MAP
# =============================================================================


def _plot_polygon_layer(
    ax: plt.Axes,
    gdf: Optional[gpd.GeoDataFrame],
    column: str,
    colors: Mapping[int, str],
    alpha: float = 0.25,
) -> None:
    """Plot polygon overlay per class.

    Class 0 is the "no hazard" background — the absence of a colored polygon
    already conveys that, and rendering it as a province-wide gray fill is
    the single largest matplotlib cost on the hazard map. Skip it.

    `rasterized=True` forces matplotlib to flatten the polygons to pixels at
    figure DPI once, rather than walking every vertex in the PNG encoder.
    """
    if gdf is None or gdf.empty:
        return
    for cls in sorted(gdf[column].unique()):
        try:
            cls_int = int(cls)
        except (ValueError, TypeError):
            continue
        if cls_int == 0:
            continue
        color = colors.get(cls_int, "#999999")
        gdf[gdf[column] == cls].plot(
            ax=ax,
            facecolor=color, edgecolor=color,
            alpha=alpha, linewidth=0.5, zorder=0,
            rasterized=True,
        )


def plot_hazard_map(
    arrays: EdgeArrays,
    flood_gdf: Optional[gpd.GeoDataFrame],
    ls_gdf: Optional[gpd.GeoDataFrame],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))

    # Polygons underneath (landslide first so flood sits on top in the overlap)
    _plot_polygon_layer(ax, ls_gdf, "LH", LANDSLIDE_CLASS_COLORS)
    _plot_polygon_layer(ax, flood_gdf, "Var", FLOOD_CLASS_COLORS)

    # Edges as a single vectorized LineCollection
    widths = 0.6 + 2.5 * arrays.combined
    lc = LineCollection(
        arrays.segments,
        array=arrays.combined,
        cmap=HAZARD_CMAP,
        norm=plt.Normalize(vmin=0.0, vmax=1.0),
        linewidths=widths,
        zorder=1,
        alpha=0.9,
    )
    ax.add_collection(lc)

    # Auto-fit bounds from segments
    x_min = float(arrays.segments[:, :, 0].min())
    x_max = float(arrays.segments[:, :, 0].max())
    y_min = float(arrays.segments[:, :, 1].min())
    y_max = float(arrays.segments[:, :, 1].max())
    x_pad = (x_max - x_min) * 0.03
    y_pad = (y_max - y_min) * 0.03
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)

    cbar = fig.colorbar(lc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Combined hazard score  (flood + landslide) / 2", fontsize=10)

    ax.set_title(
        "La Trinidad (Benguet) — hazard-colored road network\n"
        f"{arrays.n:,} edges · canonical-3 mapping · pipeline v3",
        fontsize=13, fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" triggers an extra render pass; omit it. 120 dpi keeps
    # the file well under the previous 1.5 MB while staying crisp.
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    size_kb = out_path.stat().st_size / 1024.0
    logger.info(f"  wrote {out_path} ({size_kb:.1f} KB)")


# =============================================================================
# MAIN
# =============================================================================


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render static matplotlib PNGs of the v3 hazard graph."
    )
    parser.add_argument("--graph", default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--flood-shp", default=DEFAULT_FLOOD_SHP)
    parser.add_argument("--landslide-shp", default=DEFAULT_LS_SHP)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--skip-map", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )

    try:
        arrays, _node_xy = load_graph(Path(args.graph))
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except ValueError as exc:
        logger.error(f"graphml invalid: {exc}")
        return 1

    manifest = load_manifest(Path(args.manifest))
    out_dir = Path(args.out_dir)

    if not args.skip_stats:
        logger.info("Rendering edge-stats figure...")
        plot_edge_stats(arrays, manifest, out_dir / "edge_stats.png")

    if not args.skip_map:
        logger.info("Rendering hazard-map figure...")
        bbox = road_bbox(arrays)
        logger.info(
            f"  road bbox (padded 5%%): x=[{bbox[0]:.4f}, {bbox[2]:.4f}] "
            f"y=[{bbox[1]:.4f}, {bbox[3]:.4f}]"
        )
        flood_gdf, ls_gdf = load_shapefiles(
            Path(args.flood_shp),
            Path(args.landslide_shp),
            flood_column="Var",
            ls_column="LH",
            bbox=bbox,
        )
        plot_hazard_map(arrays, flood_gdf, ls_gdf, out_dir / "hazard_map.png")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
