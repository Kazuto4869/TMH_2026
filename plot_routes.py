"""Plot saved weekly routes without rerunning solvers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


DAY_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for saved-route plotting."""
    parser = argparse.ArgumentParser(description="Plot routes from saved weekly VRP result.json files.")
    parser.add_argument("--results-dir", default="results", help="Directory containing schedules/{solver}/result.json.")
    parser.add_argument("--locations", default="data/locations.csv", help="Location CSV containing x_km and y_km.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to results/comparison/route_plots.")
    parser.add_argument("--solver", action="append", help="Solver to plot. Repeat for multiple solvers; default is all saved solvers.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    parser.add_argument("--show-labels", action="store_true", help="Label customer ids on route points.")
    return parser


def load_locations(path: str | Path) -> dict[str, tuple[float, float]]:
    """Read location coordinates keyed by location id."""
    with Path(path).open(newline="", encoding="utf-8-sig") as file_obj:
        rows = csv.DictReader(file_obj)
        required = {"location_id", "x_km", "y_km"}
        if rows.fieldnames is None or not required <= set(rows.fieldnames):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        return {
            row["location_id"]: (float(row["x_km"]), float(row["y_km"]))
            for row in rows
        }


def result_files(results_dir: str | Path, solvers: list[str] | None) -> list[Path]:
    """Return selected saved result files in deterministic order."""
    schedules_dir = Path(results_dir) / "schedules"
    if solvers:
        paths = [schedules_dir / solver / "result.json" for solver in solvers]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing saved result: " + ", ".join(str(path) for path in missing))
        return paths
    paths = sorted(schedules_dir.glob("*/result.json"))
    if not paths:
        raise FileNotFoundError(f"No saved result files found under {schedules_dir}")
    return paths


def plot_result(
    result_path: str | Path,
    coordinates: dict[str, tuple[float, float]],
    output_dir: str | Path,
    *,
    dpi: int = 180,
    show_labels: bool = False,
) -> Path:
    """Render one seven-panel weekly route plot and return its path."""
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    solver = str(payload.get("solver") or Path(result_path).parent.name)
    routes = payload.get("schedule", {}).get("routes", [])
    routes_by_day = {int(route["day"]): route for route in routes}
    depot_id = "DEPOT" if "DEPOT" in coordinates else next(iter(coordinates))
    depot = coordinates[depot_id]
    delivered = {
        stop["customer_id"]
        for route in routes
        for stop in route.get("stops", [])
        if stop.get("hard_feasible", True)
    }
    customers = [customer_id for customer_id in coordinates if customer_id != depot_id]
    incomplete = [customer_id for customer_id in customers if customer_id not in delivered]

    _configure_matplotlib()
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True, sharey=True)
    flat_axes = list(axes.flat)
    metrics = payload.get("metrics", {})
    figure.suptitle(
        f"{solver} | objective={float(metrics.get('objective_value', 0.0)):,.2f} | "
        f"delivered={metrics.get('delivered_count', len(delivered))} | incomplete={metrics.get('incomplete_count', len(incomplete))}",
        fontsize=14,
    )

    for day in range(1, 8):
        axis = flat_axes[day - 1]
        color = DAY_COLORS[day - 1]
        route = routes_by_day.get(day, {})
        stop_ids = [
            stop["customer_id"]
            for stop in route.get("stops", [])
            if stop.get("hard_feasible", True)
        ]
        unknown = [customer_id for customer_id in stop_ids if customer_id not in coordinates]
        if unknown:
            raise KeyError(f"Missing coordinates for {unknown} in {result_path}")

        if incomplete:
            axis.scatter(
                [coordinates[c][0] for c in incomplete],
                [coordinates[c][1] for c in incomplete],
                s=8,
                color="#B8B8B8",
                alpha=0.35,
                label="Incomplete" if day == 1 else None,
                zorder=1,
            )
        path_ids = [depot_id, *stop_ids, depot_id]
        path_x = [coordinates[location_id][0] for location_id in path_ids]
        path_y = [coordinates[location_id][1] for location_id in path_ids]
        if stop_ids:
            axis.plot(path_x, path_y, color=color, linewidth=1.1, alpha=0.85, zorder=2)
            axis.scatter(path_x[1:-1], path_y[1:-1], s=16, color=color, edgecolors="white", linewidths=0.35, zorder=3)
            for start, end in zip(path_ids[:-1], path_ids[1:]):
                start_x, start_y = coordinates[start]
                end_x, end_y = coordinates[end]
                axis.annotate(
                    "",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 0.7, "alpha": 0.65},
                    zorder=2,
                )
        axis.scatter([depot[0]], [depot[1]], marker="*", s=115, color="#C00000", edgecolors="white", linewidths=0.6, zorder=4)
        if show_labels:
            for customer_id in stop_ids:
                x_coord, y_coord = coordinates[customer_id]
                axis.annotate(customer_id, (x_coord, y_coord), xytext=(3, 3), textcoords="offset points", fontsize=5)
        axis.set_title(
            f"Day {day}: {len(stop_ids)} stops | {float(route.get('route_distance_km', 0.0)):.1f} km",
            fontsize=10,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#DDDDDD", linewidth=0.5, alpha=0.7)
        axis.set_xlabel("x (km)")
        axis.set_ylabel("y (km)")

    legend_axis = flat_axes[-1]
    legend_axis.axis("off")
    legend_axis.text(
        0.05,
        0.85,
        "Legend\n★ Depot\n● Delivered\n● Incomplete\n→ Direction",
        fontsize=16,
        va="top",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(output_dir) / f"{solver}_routes.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _configure_matplotlib() -> None:
    """Use writable cache paths and a non-interactive rendering backend."""
    temp_dir = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(temp_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(temp_dir))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")


def main(argv: list[str] | None = None) -> int:
    """Plot all selected saved solver results."""
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir) / "comparison" / "route_plots"
    coordinates = load_locations(args.locations)
    paths = result_files(args.results_dir, args.solver)
    for path in paths:
        output_path = plot_result(
            path,
            coordinates,
            output_dir,
            dpi=args.dpi,
            show_labels=args.show_labels,
        )
        print(f"saved_route_plot={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
