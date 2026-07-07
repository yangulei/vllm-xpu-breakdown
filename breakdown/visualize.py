# SPDX-License-Identifier: Apache-2.0
"""HTML report generator — self-contained interactive visualization."""

from __future__ import annotations

import json
import os
from collections import defaultdict

from .classifier import Backend, ClassificationResult

# Colors for each backend
_BACKEND_COLORS = {
    Backend.VLLM_XPU_KERNELS: "#4CAF50",    # green
    Backend.VLLM_CUDA_KERNELS: "#76FF03",   # lime green
    Backend.TRITON: "#2196F3",               # blue
    Backend.TORCH_XPU_OPS: "#FF9800",        # orange
    Backend.TORCH_CUDA_OPS: "#E91E63",       # pink
    Backend.CPU: "#9E9E9E",                  # grey
    Backend.FRAMEWORK: "#BDBDBD",            # light grey
    Backend.CCL: "#7B1FA2",                  # purple
}

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>vLLM Ops Breakdown</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f5; color: #333; padding: 24px; }
  h1 { margin-bottom: 8px; }
  .subtitle { color: #666; margin-bottom: 24px; }
  .card { background: #fff; border-radius: 8px; padding: 24px; margin-bottom: 24px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .stats-row { display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 24px; }
  .stat-box { background: #fff; border-radius: 8px; padding: 16px 24px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 180px; }
  .stat-box .value { font-size: 24px; font-weight: bold; }
  .stat-box .label { color: #666; font-size: 14px; }
  .charts { display: flex; gap: 24px; flex-wrap: wrap; }
  .chart-container { flex: 1; min-width: 300px; }
  canvas { max-width: 100%; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { background: #f9f9f9; text-align: left; padding: 10px 12px; cursor: pointer;
       border-bottom: 2px solid #ddd; user-select: none; }
  th:hover { background: #eee; }
  td { padding: 8px 12px; border-bottom: 1px solid #eee; }
  tr:hover { background: #f5f7fa; }
  .backend-badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
                   color: #fff; font-size: 12px; font-weight: 500; }
  .bar { height: 18px; border-radius: 3px; display: inline-block; min-width: 2px; }
  .filter-row { margin-bottom: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
  .filter-btn { padding: 6px 14px; border-radius: 16px; border: 1px solid #ddd;
                background: #fff; cursor: pointer; font-size: 13px; }
  .filter-btn.active { background: #333; color: #fff; border-color: #333; }
  #search { padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px;
            width: 300px; font-size: 14px; }
</style>
</head>
<body>

<h1>vLLM Ops/Kernels Breakdown</h1>
<p class="subtitle">Dispatch analysis for GPU inference</p>

<div class="stats-row" id="stats-row"></div>

<div class="card">
  <h2 style="margin-bottom:16px">Backend Distribution</h2>
  <div class="charts">
    <div class="chart-container">
      <canvas id="pieChart" width="400" height="300"></canvas>
    </div>
    <div class="chart-container">
      <canvas id="barChart" width="600" height="300"></canvas>
    </div>
  </div>
</div>

<div class="card">
  <h2 style="margin-bottom:16px">All Operations</h2>
  <div class="filter-row">
    <input type="text" id="search" placeholder="Search op name...">
    <button class="filter-btn active" data-backend="all">All</button>
  </div>
  <div style="overflow-x:auto">
    <table id="ops-table">
      <thead>
        <tr>
          <th data-sort="name">Op Name</th>
          <th data-sort="backend">Backend</th>
          <th data-sort="category">Category</th>
          <th data-sort="device_time_us" data-type="num">Device Time</th>
          <th data-sort="pct">% of Total</th>
          <th data-sort="count" data-type="num">Calls</th>
          <th data-sort="cpu_time_us" data-type="num">CPU Time</th>
        </tr>
      </thead>
      <tbody id="ops-tbody"></tbody>
    </table>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

const COLORS = __COLORS_JSON__;

function fmtTime(us) {
  if (us >= 1e6) return (us/1e6).toFixed(2) + 's';
  if (us >= 1e3) return (us/1e3).toFixed(2) + 'ms';
  return us.toFixed(0) + 'µs';
}

// Stats row
(function() {
  const row = document.getElementById('stats-row');
  const items = [
    { label: 'Total Device Time', value: fmtTime(DATA.summary.total_device_time_us) },
    { label: 'Total CPU Time', value: fmtTime(DATA.summary.total_cpu_time_us) },
    { label: 'Unique Ops', value: DATA.summary.total_unique_ops },
  ];
  for (const [k,v] of Object.entries(DATA.summary.backends)) {
    if (v.num_ops > 0) items.push({ label: k, value: v.num_ops + ' ops' });
  }
  items.forEach(i => {
    const d = document.createElement('div');
    d.className = 'stat-box';
    d.innerHTML = `<div class="value">${i.value}</div><div class="label">${i.label}</div>`;
    row.appendChild(d);
  });
})();

// Simple canvas pie chart
function drawPie(canvasId, data) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const cx = 150, cy = 150, r = 120;
  const total = data.reduce((s,d) => s + d.value, 0);
  if (total === 0) return;
  let angle = -Math.PI/2;
  data.forEach((d, i) => {
    const slice = (d.value / total) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, angle, angle + slice);
    ctx.fillStyle = d.color;
    ctx.fill();
    angle += slice;
  });
  // Legend
  let ly = 20;
  data.forEach(d => {
    if (d.value === 0) return;
    ctx.fillStyle = d.color;
    ctx.fillRect(310, ly, 14, 14);
    ctx.fillStyle = '#333';
    ctx.font = '13px sans-serif';
    const pct = (d.value / total * 100).toFixed(1);
    ctx.fillText(`${d.label} (${pct}%)`, 330, ly + 12);
    ly += 22;
  });
}

// Simple bar chart (top ops)
function drawBars(canvasId, ops, maxBars) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const sorted = [...ops].sort((a,b) => b.device_time_us - a.device_time_us).slice(0, maxBars);
  if (sorted.length === 0) return;
  const maxVal = sorted[0].device_time_us;
  const barH = 22, gap = 4, leftMargin = 200, topMargin = 10;
  canvas.height = topMargin + sorted.length * (barH + gap) + 10;
  const barArea = canvas.width - leftMargin - 80;
  sorted.forEach((op, i) => {
    const y = topMargin + i * (barH + gap);
    const w = maxVal > 0 ? (op.device_time_us / maxVal) * barArea : 0;
    ctx.fillStyle = COLORS[op.backend] || '#999';
    ctx.fillRect(leftMargin, y, Math.max(w, 2), barH);
    ctx.fillStyle = '#333';
    ctx.font = '12px monospace';
    const label = op.name.length > 30 ? op.name.slice(0,30)+'…' : op.name;
    ctx.textAlign = 'right';
    ctx.fillText(label, leftMargin - 8, y + 15);
    ctx.textAlign = 'left';
    ctx.fillText(fmtTime(op.device_time_us), leftMargin + w + 6, y + 15);
  });
}

// Pie data
const pieData = Object.entries(DATA.summary.backends)
  .filter(([k,v]) => v.device_time_us > 0)
  .map(([k,v]) => ({ label: k, value: v.device_time_us, color: COLORS[k] || '#999' }));
drawPie('pieChart', pieData);
drawBars('barChart', DATA.ops, 15);

// Table
const tbody = document.getElementById('ops-tbody');
const totalDev = DATA.summary.total_device_time_us;

function renderTable(ops) {
  tbody.innerHTML = '';
  ops.forEach(op => {
    const pct = totalDev > 0 ? (op.device_time_us / totalDev * 100).toFixed(2) : '0.00';
    const barW = totalDev > 0 ? Math.max(op.device_time_us / totalDev * 100, 0.5) : 0;
    const color = COLORS[op.backend] || '#999';
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><code>${op.name}</code></td>
      <td><span class="backend-badge" style="background:${color}">${op.backend}</span></td>
      <td>${op.category}</td>
      <td>${fmtTime(op.device_time_us)}</td>
      <td><span class="bar" style="width:${barW}%;background:${color}"></span> ${pct}%</td>
      <td>${op.count}</td>
      <td>${fmtTime(op.cpu_time_us)}</td>`;
    tbody.appendChild(tr);
  });
}

let currentOps = DATA.ops;
let currentSort = { key: 'device_time_us', asc: false };
let currentFilter = 'all';
let searchText = '';

function applyFilters() {
  let ops = DATA.ops;
  if (currentFilter !== 'all') ops = ops.filter(o => o.backend === currentFilter);
  if (searchText) ops = ops.filter(o => o.name.toLowerCase().includes(searchText));
  ops = [...ops].sort((a,b) => {
    let va = a[currentSort.key], vb = b[currentSort.key];
    if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    if (va < vb) return currentSort.asc ? -1 : 1;
    if (va > vb) return currentSort.asc ? 1 : -1;
    return 0;
  });
  renderTable(ops);
}

// Filter buttons
const filterRow = document.querySelector('.filter-row');
Object.keys(DATA.summary.backends).forEach(b => {
  if (DATA.summary.backends[b].num_ops === 0) return;
  const btn = document.createElement('button');
  btn.className = 'filter-btn';
  btn.dataset.backend = b;
  btn.textContent = b;
  filterRow.appendChild(btn);
});

filterRow.addEventListener('click', e => {
  if (!e.target.classList.contains('filter-btn')) return;
  filterRow.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  e.target.classList.add('active');
  currentFilter = e.target.dataset.backend;
  applyFilters();
});

document.getElementById('search').addEventListener('input', e => {
  searchText = e.target.value.toLowerCase();
  applyFilters();
});

document.querySelectorAll('th[data-sort]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.sort;
    if (key === currentSort.key) currentSort.asc = !currentSort.asc;
    else { currentSort.key = key; currentSort.asc = false; }
    applyFilters();
  });
});

applyFilters();
</script>
</body>
</html>
"""


def generate_html(result: ClassificationResult, output_dir: str) -> str:
    """Generate a self-contained HTML report. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "breakdown.html")

    total_dev = result.total_device_time_us

    # Build backend summary
    backend_summary: dict[str, dict] = {}
    for backend in Backend:
        ops = [o for o in result.ops if o.backend == backend]
        dev_time = sum(o.device_time_us for o in ops)
        backend_summary[backend.value] = {
            "device_time_us": dev_time,
            "pct_device_time": (dev_time / total_dev * 100) if total_dev > 0 else 0,
            "num_ops": len(ops),
            "num_calls": sum(o.count for o in ops),
        }

    data = {
        "summary": {
            "total_device_time_us": total_dev,
            "total_cpu_time_us": result.total_cpu_time_us,
            "total_unique_ops": len(result.ops),
            "backends": backend_summary,
        },
        "ops": sorted(
            [
                {
                    "name": op.name,
                    "backend": op.backend.value,
                    "category": op.category,
                    "device_time_us": op.device_time_us,
                    "cpu_time_us": op.cpu_time_us,
                    "count": op.count,
                    "pct": (op.device_time_us / total_dev * 100) if total_dev > 0 else 0,
                }
                for op in result.ops
            ],
            key=lambda o: o["device_time_us"],
            reverse=True,
        ),
    }

    colors = {b.value: c for b, c in _BACKEND_COLORS.items()}

    html = _HTML_TEMPLATE.replace(
        "__DATA_JSON__", json.dumps(data)
    ).replace(
        "__COLORS_JSON__", json.dumps(colors)
    )

    with open(path, "w") as f:
        f.write(html)
    return path
