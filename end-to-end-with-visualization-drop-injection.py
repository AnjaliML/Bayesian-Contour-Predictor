#!/usr/bin/env python3
"""Run and visualize the drop-injection end-to-end active-learning workflow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

from classify_drops import oh_c


REPO_ROOT = Path(__file__).resolve().parent
PROPOSE_SCRIPT = REPO_ROOT / "propose_next_sweep.py"
CLASSIFIER_SCRIPT = REPO_ROOT / "classify_drops.py"


@dataclass(frozen=True)
class Domain:
    rr_min: float
    rr_max: float
    oh_min: float
    oh_max: float


@dataclass(frozen=True)
class CompletedRun:
    case_id: str
    rr: float
    oh: float
    label: int
    sweep: int


@dataclass(frozen=True)
class ProposedRun:
    case_id: str
    rr: float
    oh: float
    proposal_type: str
    score: float
    p_positive_pred: float
    y_c_pred: float
    y_c_q05: float
    y_c_q95: float
    y_c_std: float
    reason: str


class DirectoryHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Animate a drop-injection active-learning campaign by calling "
            "propose_next_sweep.py and classify_drops.py under the hood."
        )
    )
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--initial-points", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rr-min", type=float, default=2.0)
    parser.add_argument("--rr-max", type=float, default=12.0)
    parser.add_argument("--oh-min", type=float, default=0.01)
    parser.add_argument("--oh-max", type=float, default=0.20)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--posterior-samples", type=int, default=40)
    parser.add_argument("--preview-points", type=int, default=48)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--no-hold", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1")
    if args.initial_points < 2:
        raise ValueError("--initial-points must be at least 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.rr_min >= args.rr_max:
        raise ValueError("--rr-min must be less than --rr-max")
    if args.oh_min <= 0 or args.oh_min >= args.oh_max:
        raise ValueError("--oh-min must be positive and less than --oh-max")
    if args.grid_size < 5:
        raise ValueError("--grid-size must be at least 5")
    if args.posterior_samples < 0:
        raise ValueError("--posterior-samples cannot be negative")
    if args.preview_points < 2:
        raise ValueError("--preview-points must be at least 2")
    if args.delay < 0:
        raise ValueError("--delay cannot be negative")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / "visualization_runs" / f"drop-injection-{stamp}"


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def format_float(value: float) -> str:
    return f"{value:.10g}"


def classify_drop(rr: float, oh: float) -> int:
    result = subprocess.run(
        [sys.executable, str(CLASSIFIER_SCRIPT), format_float(oh), format_float(rr)],
        check=True,
        text=True,
        capture_output=True,
    )
    output = result.stdout.strip()
    if output not in {"0", "1"}:
        raise RuntimeError(f"classify_drops.py returned unexpected output {output!r}")
    return int(output)


def make_initial_design(count: int, domain: Domain, seed: int) -> list[CompletedRun]:
    rng = random.Random(seed)
    rr_slots = list(range(count))
    oh_slots = list(range(count))
    rng.shuffle(rr_slots)
    rng.shuffle(oh_slots)

    log_oh_min = math.log10(domain.oh_min)
    log_oh_max = math.log10(domain.oh_max)
    runs: list[CompletedRun] = []
    for index in range(count):
        rr_fraction = (rr_slots[index] + 0.5) / count
        oh_fraction = (oh_slots[index] + 0.5) / count
        rr = domain.rr_min + rr_fraction * (domain.rr_max - domain.rr_min)
        oh = 10 ** (log_oh_min + oh_fraction * (log_oh_max - log_oh_min))
        label = classify_drop(rr, oh)
        runs.append(
            CompletedRun(
                case_id=str(index + 1),
                rr=rr,
                oh=oh,
                label=label,
                sweep=0,
            )
        )
    return runs


def completed_runs_to_rows(runs: Sequence[CompletedRun]) -> list[dict[str, str]]:
    return [
        {
            "caseId": run.case_id,
            "Rr": format_float(run.rr),
            "Oh": format_float(run.oh),
            "id": str(run.label),
        }
        for run in runs
    ]


def write_completed_sweep(path: Path, runs: Sequence[CompletedRun]) -> None:
    write_csv(path, ["caseId", "Rr", "Oh", "id"], completed_runs_to_rows(runs))


def run_proposal(
    input_files: Sequence[Path],
    outfile: Path,
    domain: Domain,
    *,
    n_simulations: int,
    seed: int,
    grid_size: int,
    posterior_samples: int,
    n_new: int | None = None,
    n_repeats: int | None = None,
) -> list[dict[str, str]]:
    command = [
        sys.executable,
        str(PROPOSE_SCRIPT),
        *[str(path) for path in input_files],
        "--outfile",
        str(outfile),
        "--n-simulations",
        str(n_simulations),
        "--seed",
        str(seed),
        "--x-col",
        "Rr",
        "--y-col",
        "Oh",
        "--mode",
        "monotone-y",
        "--monotone-direction",
        "decreasing",
        "--y-scale",
        "log10",
        "--x-min",
        format_float(domain.rr_min),
        "--x-max",
        format_float(domain.rr_max),
        "--y-min",
        format_float(domain.oh_min),
        "--y-max",
        format_float(domain.oh_max),
        "--grid-size",
        str(grid_size),
        "--posterior-samples",
        str(posterior_samples),
    ]
    if n_new is not None:
        command.extend(["--n-new", str(n_new)])
    if n_repeats is not None:
        command.extend(["--n-repeats", str(n_repeats)])
    subprocess.run(command, check=True)
    return read_csv(outfile)


def proposal_from_row(row: dict[str, str]) -> ProposedRun:
    return ProposedRun(
        case_id=row["caseId"],
        rr=float(row["x"]),
        oh=float(row["y"]),
        proposal_type=row.get("proposal_type", "new"),
        score=float(row.get("score", "0") or 0.0),
        p_positive_pred=float(row.get("p_positive_pred", "0.5") or 0.5),
        y_c_pred=float(row.get("y_c_pred", row["y"]) or row["y"]),
        y_c_q05=float(row.get("y_c_q05", row["y"]) or row["y"]),
        y_c_q95=float(row.get("y_c_q95", row["y"]) or row["y"]),
        y_c_std=float(row.get("y_c_std", "0") or 0.0),
        reason=row.get("reason", ""),
    )


def proposal_rows_to_completed(
    rows: Sequence[dict[str, str]], sweep: int
) -> list[CompletedRun]:
    completed: list[CompletedRun] = []
    for row in rows:
        rr = float(row["x"])
        oh = float(row["y"])
        completed.append(
            CompletedRun(
                case_id=row["caseId"],
                rr=rr,
                oh=oh,
                label=classify_drop(rr, oh),
                sweep=sweep,
            )
        )
    return completed


def find_true_contour(domain: Domain, samples: int = 80) -> list[dict[str, float]]:
    contour: list[dict[str, float]] = []
    for index in range(samples):
        fraction = index / (samples - 1) if samples > 1 else 0.5
        rr = domain.rr_min + fraction * (domain.rr_max - domain.rr_min)
        oh = oh_c(rr)
        if domain.oh_min <= oh <= domain.oh_max:
            contour.append({"Rr": rr, "Oh": oh})
    return contour


def build_preview_contour(rows: Sequence[dict[str, str]]) -> list[dict[str, float]]:
    seen: dict[float, dict[str, float]] = {}
    for row in rows:
        rr = float(row["x"])
        seen[rr] = {
            "Rr": rr,
            "y_c_pred": float(row.get("y_c_pred", row["y"]) or row["y"]),
            "y_c_q05": float(row.get("y_c_q05", row["y"]) or row["y"]),
            "y_c_q95": float(row.get("y_c_q95", row["y"]) or row["y"]),
        }
    return [seen[rr] for rr in sorted(seen)]


def write_state(output_dir: Path, state: dict[str, Any]) -> None:
    with (output_dir / "state.json").open("w") as handle:
        json.dump(state, handle, indent=2)


def completed_payload(runs: Sequence[CompletedRun]) -> list[dict[str, Any]]:
    return [
        {
            "caseId": run.case_id,
            "Rr": run.rr,
            "Oh": run.oh,
            "id": run.label,
            "sweep": run.sweep,
        }
        for run in runs
    ]


def proposals_payload(proposals: Sequence[ProposedRun]) -> list[dict[str, Any]]:
    return [
        {
            "caseId": proposal.case_id,
            "Rr": proposal.rr,
            "Oh": proposal.oh,
            "proposal_type": proposal.proposal_type,
            "score": proposal.score,
            "p_positive_pred": proposal.p_positive_pred,
            "y_c_pred": proposal.y_c_pred,
            "y_c_q05": proposal.y_c_q05,
            "y_c_q95": proposal.y_c_q95,
            "y_c_std": proposal.y_c_std,
            "reason": proposal.reason,
        }
        for proposal in proposals
    ]


def update_visual_state(
    output_dir: Path,
    *,
    status: str,
    iteration: int,
    domain: Domain,
    completed: Sequence[CompletedRun],
    proposals: Sequence[ProposedRun],
    contour: Sequence[dict[str, float]],
    true_contour: Sequence[dict[str, float]],
    messages: Sequence[str],
) -> None:
    state = {
        "status": status,
        "iteration": iteration,
        "domain": {
            "rr_min": domain.rr_min,
            "rr_max": domain.rr_max,
            "oh_min": domain.oh_min,
            "oh_max": domain.oh_max,
        },
        "completed": completed_payload(completed),
        "proposals": proposals_payload(proposals),
        "contour": list(contour),
        "true_contour": list(true_contour),
        "messages": list(messages[-12:]),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_state(output_dir, state)


def write_html(output_dir: Path) -> None:
    (output_dir / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drop Injection Active Learning Visualizer</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --ink: #1c2430;
      --muted: #607085;
      --line: #d7dee8;
      --drop: #0b7f62;
      --no-drop: #b83a4b;
      --proposal: #d99b15;
      --curve: #265ecf;
      --truth: #202936;
      --band: rgba(38, 94, 207, 0.14);
    }
    body {
      margin: 0;
      background: #f6f7f9;
      color: var(--ink);
    }
    main {
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 340px;
      gap: 18px;
      min-height: 100vh;
      padding: 18px;
      box-sizing: border-box;
    }
    .plot-shell, aside {
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(28, 36, 48, 0.06);
    }
    .plot-shell {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    header {
      padding: 14px 16px 10px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0 0 4px;
      font-weight: 700;
      letter-spacing: 0;
    }
    #subtitle {
      color: var(--muted);
      font-size: 13px;
    }
    canvas {
      width: 100%;
      height: min(72vh, 720px);
      display: block;
    }
    aside {
      padding: 14px;
      overflow: auto;
    }
    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 14px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
    }
    .stat b {
      display: block;
      font-size: 18px;
      line-height: 1.1;
    }
    .stat span {
      color: var(--muted);
      font-size: 12px;
    }
    h2 {
      font-size: 13px;
      margin: 16px 0 8px;
      text-transform: uppercase;
      letter-spacing: 0;
      color: var(--muted);
    }
    .legend {
      display: grid;
      gap: 8px;
      font-size: 13px;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      display: inline-block;
      border: 2px solid transparent;
      box-sizing: border-box;
    }
    .line-swatch {
      width: 28px;
      height: 0;
      border-top: 3px solid var(--curve);
      display: inline-block;
    }
    .truth-swatch {
      border-top-color: var(--truth);
      border-top-style: dashed;
    }
    #messages {
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 8px;
      font-size: 12px;
      color: var(--ink);
    }
    #messages li {
      border-left: 3px solid var(--line);
      padding-left: 8px;
      color: #344154;
    }
    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
      }
      canvas {
        height: 62vh;
      }
    }
  </style>
</head>
<body>
<main>
  <section class="plot-shell">
    <header>
      <h1>Drop Injection Active Learning</h1>
      <div id="subtitle">Waiting for campaign state...</div>
    </header>
    <canvas id="plot" width="1100" height="720"></canvas>
  </section>
  <aside>
    <div class="stats">
      <div class="stat"><b id="iteration">0</b><span>Iteration</span></div>
      <div class="stat"><b id="completed">0</b><span>Completed runs</span></div>
      <div class="stat"><b id="drops">0</b><span>Drops</span></div>
      <div class="stat"><b id="pending">0</b><span>Current proposals</span></div>
    </div>
    <h2>Legend</h2>
    <div class="legend">
      <div class="legend-row"><span class="swatch" style="background: var(--drop)"></span>Drops, id = 1</div>
      <div class="legend-row"><span class="swatch" style="background: var(--no-drop)"></span>No drops, id = 0</div>
      <div class="legend-row"><span class="swatch" style="border-color: var(--proposal)"></span>Proposed next run</div>
      <div class="legend-row"><span class="line-swatch"></span>Learned contour</div>
      <div class="legend-row"><span class="line-swatch truth-swatch"></span>Classifier contour</div>
    </div>
    <h2>Process</h2>
    <ul id="messages"></ul>
  </aside>
</main>
<script>
const canvas = document.getElementById("plot");
const ctx = canvas.getContext("2d");
let latest = null;
const pad = { left: 78, right: 28, top: 34, bottom: 64 };

function log10(v) { return Math.log(v) / Math.LN10; }

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(720, Math.floor(rect.width * ratio));
  canvas.height = Math.max(480, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function plotArea() {
  const rect = canvas.getBoundingClientRect();
  return {
    x0: pad.left,
    y0: pad.top,
    x1: rect.width - pad.right,
    y1: rect.height - pad.bottom,
  };
}

function scales(state) {
  const area = plotArea();
  const d = state.domain;
  const lo = log10(d.oh_min);
  const hi = log10(d.oh_max);
  return {
    x(rr) {
      return area.x0 + (rr - d.rr_min) / (d.rr_max - d.rr_min) * (area.x1 - area.x0);
    },
    y(oh) {
      return area.y1 - (log10(oh) - lo) / (hi - lo) * (area.y1 - area.y0);
    },
    area,
  };
}

function drawGrid(state, s) {
  const { area } = s;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#d7dee8";
  ctx.lineWidth = 1;
  ctx.strokeRect(area.x0, area.y0, area.x1 - area.x0, area.y1 - area.y0);

  ctx.fillStyle = "#607085";
  ctx.font = "12px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const d = state.domain;
  for (let i = 0; i <= 5; i++) {
    const rr = d.rr_min + (d.rr_max - d.rr_min) * i / 5;
    const x = s.x(rr);
    ctx.beginPath();
    ctx.moveTo(x, area.y0);
    ctx.lineTo(x, area.y1);
    ctx.strokeStyle = i === 0 || i === 5 ? "#d7dee8" : "#eef2f6";
    ctx.stroke();
    ctx.fillText(rr.toFixed(1), x, area.y1 + 12);
  }

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yTicks = [0.01, 0.02, 0.04, 0.08, 0.16].filter(v => v >= d.oh_min && v <= d.oh_max);
  for (const oh of yTicks) {
    const y = s.y(oh);
    ctx.beginPath();
    ctx.moveTo(area.x0, y);
    ctx.lineTo(area.x1, y);
    ctx.strokeStyle = "#eef2f6";
    ctx.stroke();
    ctx.fillText(oh.toFixed(3).replace(/0+$/, "").replace(/\\.$/, ""), area.x0 - 10, y);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillStyle = "#1c2430";
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillText("Rr", (area.x0 + area.x1) / 2, area.y1 + 48);
  ctx.save();
  ctx.translate(20, (area.y0 + area.y1) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Oh (log scale)", 0, 0);
  ctx.restore();
}

function drawLine(points, s, xKey, yKey, color, width, dashed = false) {
  const valid = points.filter(p => Number.isFinite(p[xKey]) && Number.isFinite(p[yKey]) && p[yKey] > 0);
  if (valid.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dashed ? [7, 6] : []);
  ctx.beginPath();
  valid.forEach((p, i) => {
    const x = s.x(p[xKey]);
    const y = s.y(p[yKey]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawBand(points, s) {
  const valid = points.filter(p => p.y_c_q05 > 0 && p.y_c_q95 > 0);
  if (valid.length < 2) return;
  ctx.save();
  ctx.fillStyle = "rgba(38, 94, 207, 0.14)";
  ctx.beginPath();
  valid.forEach((p, i) => {
    const x = s.x(p.Rr);
    const y = s.y(p.y_c_q95);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  [...valid].reverse().forEach(p => {
    ctx.lineTo(s.x(p.Rr), s.y(p.y_c_q05));
  });
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawPoints(state, s) {
  for (const p of state.completed) {
    const x = s.x(p.Rr);
    const y = s.y(p.Oh);
    ctx.beginPath();
    ctx.arc(x, y, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = p.id === 1 ? "#0b7f62" : "#b83a4b";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  for (const p of state.proposals) {
    const x = s.x(p.Rr);
    const y = s.y(p.Oh);
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.PI / 4);
    ctx.strokeStyle = "#d99b15";
    ctx.lineWidth = 2.4;
    ctx.strokeRect(-5.5, -5.5, 11, 11);
    ctx.restore();
  }
}

function draw() {
  if (!latest) return;
  const state = latest;
  const s = scales(state);
  drawGrid(state, s);
  drawBand(state.contour || [], s);
  drawLine(state.true_contour || [], s, "Rr", "Oh", "#202936", 2, true);
  drawLine(state.contour || [], s, "Rr", "y_c_pred", "#265ecf", 3, false);
  drawPoints(state, s);
}

function updateSidebar(state) {
  const drops = state.completed.filter(p => p.id === 1).length;
  document.getElementById("subtitle").textContent = `${state.status} · updated ${state.updated_at}`;
  document.getElementById("iteration").textContent = state.iteration;
  document.getElementById("completed").textContent = state.completed.length;
  document.getElementById("drops").textContent = drops;
  document.getElementById("pending").textContent = state.proposals.length;
  const messages = document.getElementById("messages");
  messages.innerHTML = "";
  for (const message of state.messages || []) {
    const li = document.createElement("li");
    li.textContent = message;
    messages.appendChild(li);
  }
}

async function poll() {
  try {
    const response = await fetch(`state.json?ts=${Date.now()}`, { cache: "no-store" });
    if (response.ok) {
      latest = await response.json();
      updateSidebar(latest);
      draw();
    }
  } catch (error) {
    document.getElementById("subtitle").textContent = "Waiting for local visualizer server...";
  } finally {
    setTimeout(poll, 350);
  }
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();
poll();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def start_server(output_dir: Path, port: int) -> tuple[ThreadingHTTPServer, str]:
    handler = lambda *args, **kwargs: DirectoryHandler(  # noqa: E731
        *args, directory=str(output_dir), **kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, actual_port = server.server_address
    return server, f"http://{host}:{actual_port}/index.html"


def sleep_if_requested(delay: float) -> None:
    if delay > 0:
        time.sleep(delay)


def run_campaign(args: argparse.Namespace) -> tuple[Path, str | None]:
    domain = Domain(
        rr_min=args.rr_min,
        rr_max=args.rr_max,
        oh_min=args.oh_min,
        oh_max=args.oh_max,
    )
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_html(output_dir)

    messages: list[str] = []
    true_contour = find_true_contour(domain)
    messages.append("Built the reference classifier contour using classify_drops.py.")
    update_visual_state(
        output_dir,
        status="starting",
        iteration=0,
        domain=domain,
        completed=[],
        proposals=[],
        contour=[],
        true_contour=true_contour,
        messages=messages,
    )

    server: ThreadingHTTPServer | None = None
    url: str | None = None
    if not args.no_server:
        server, url = start_server(output_dir, args.port)
        messages.append(f"Serving live visualization at {url}.")
        print(f"Live visualization: {url}")
        print(f"Artifacts: {output_dir}")
        if not args.no_browser:
            webbrowser.open(url)

    all_completed = make_initial_design(args.initial_points, domain, args.seed)
    completed_files: list[Path] = [output_dir / "Sweep-0_completed.csv"]
    write_completed_sweep(completed_files[0], all_completed)
    messages.append(f"Generated and classified {len(all_completed)} initial space-filling runs.")
    update_visual_state(
        output_dir,
        status="initial design complete",
        iteration=0,
        domain=domain,
        completed=all_completed,
        proposals=[],
        contour=[],
        true_contour=true_contour,
        messages=messages,
    )
    sleep_if_requested(args.delay)

    for iteration in range(1, args.iterations + 1):
        proposed_path = output_dir / f"Sweep-{iteration}_proposed.csv"
        proposal_rows = run_proposal(
            completed_files,
            proposed_path,
            domain,
            n_simulations=args.batch_size,
            seed=args.seed + iteration,
            grid_size=args.grid_size,
            posterior_samples=args.posterior_samples,
        )
        proposals = [proposal_from_row(row) for row in proposal_rows]

        preview_path = output_dir / f"Sweep-{iteration}_contour-preview.csv"
        preview_rows = run_proposal(
            completed_files,
            preview_path,
            domain,
            n_simulations=args.preview_points,
            seed=args.seed + 10_000 + iteration,
            grid_size=args.grid_size,
            posterior_samples=max(args.posterior_samples // 2, 1),
            n_new=args.preview_points,
            n_repeats=0,
        )
        preview_contour = build_preview_contour(preview_rows)

        messages.append(
            f"Sweep {iteration}: proposed {len(proposals)} runs with propose_next_sweep.py."
        )
        update_visual_state(
            output_dir,
            status=f"sweep {iteration} proposed",
            iteration=iteration,
            domain=domain,
            completed=all_completed,
            proposals=proposals,
            contour=preview_contour,
            true_contour=true_contour,
            messages=messages,
        )
        sleep_if_requested(args.delay)

        newly_completed: list[CompletedRun] = []
        for row in proposal_rows:
            run = proposal_rows_to_completed([row], iteration)[0]
            newly_completed.append(run)
            all_completed.append(run)
            update_visual_state(
                output_dir,
                status=f"sweep {iteration} running experiments",
                iteration=iteration,
                domain=domain,
                completed=all_completed,
                proposals=[
                    proposal
                    for proposal in proposals
                    if proposal.case_id not in {done.case_id for done in newly_completed}
                ],
                contour=preview_contour,
                true_contour=true_contour,
                messages=[
                    *messages,
                    f"Ran case {run.case_id}: Rr={run.rr:.3g}, Oh={run.oh:.3g}, id={run.label}.",
                ],
            )
            sleep_if_requested(args.delay / 2)

        completed_path = output_dir / f"Sweep-{iteration}_completed.csv"
        completed_files.append(completed_path)
        write_completed_sweep(completed_path, newly_completed)
        messages.append(
            f"Sweep {iteration}: completed labels with classify_drops.py and appended results."
        )
        update_visual_state(
            output_dir,
            status=f"sweep {iteration} complete",
            iteration=iteration,
            domain=domain,
            completed=all_completed,
            proposals=[],
            contour=preview_contour,
            true_contour=true_contour,
            messages=messages,
        )
        sleep_if_requested(args.delay)

    messages.append("Campaign complete.")
    update_visual_state(
        output_dir,
        status="complete",
        iteration=args.iterations,
        domain=domain,
        completed=all_completed,
        proposals=[],
        contour=preview_contour if args.iterations else [],
        true_contour=true_contour,
        messages=messages,
    )

    if server and not args.no_hold and sys.stdin.isatty():
        input("Press Enter to stop the local visualization server...")
        server.shutdown()
    return output_dir, url


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        output_dir, url = run_campaign(args)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote visualization artifacts to {output_dir}")
    if url:
        print(f"Live visualization URL: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
