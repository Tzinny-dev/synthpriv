"""Generacion de informes HTML autocontenidos."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from jinja2 import Template

from synthpriv.metrics.base import MetricResult

_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>synthpriv - Informe de evaluacion</title>
<style>
  :root { --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; --muted:#59636e; --line:#d0d7de; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin:0; color:#1f2328; background:#f6f8fa; }
  header { background:#24292f; color:#fff; padding:20px 28px; }
  header h1 { margin:0 0 6px; font-size:22px; }
  header p { margin:2px 0; opacity:.85; font-size:14px; }
  main { max-width:1000px; margin:24px auto; padding:0 16px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:8px; padding:18px 22px; margin-bottom:18px; }
  .card h2 { margin:0 0 12px; font-size:17px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
  .stat { background:#f6f8fa; border:1px solid var(--line); border-radius:6px; padding:10px 12px; }
  .stat .label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }
  .stat .value { font-size:20px; font-weight:600; margin-top:4px; }
  table { border-collapse:collapse; width:100%; font-size:14px; margin-top:8px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:13px; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600; }
  .passed { background:#dafbe1; color:var(--ok); }
  .failed { background:#ffebe9; color:var(--bad); }
  .reported { background:#fff8c5; color:var(--warn); }
  .error { background:#f6f8fa; color:var(--muted); border:1px dashed var(--line); }
  .bar { background:#d0d7de; border-radius:4px; height:8px; overflow:hidden; margin-top:6px; }
  .bar > span { display:block; height:100%; background:var(--ok); }
  .bar > span.warn { background:var(--warn); }
  .bar > span.bad { background:var(--bad); }
  code { background:#eff1f3; border-radius:4px; padding:1px 5px; font-size:13px; }
  footer { color:var(--muted); font-size:12px; text-align:center; padding:18px; }
  .muted { color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>synthpriv &mdash; Informe de datos sinteticos</h1>
  <p>Generador: <strong>{{ data.generator.key }}</strong> &middot;
     {{ data.generator.description }}</p>
  <p>Filas reales: {{ data.rows.real }} &middot; Filas sinteticas: {{ data.rows.synthetic }}</p>
</header>
<main>

  <div class="card">
    <h2>Resumen</h2>
    <div class="grid">
      <div class="stat"><div class="label">Utilidad&nbsp;OK</div><div class="value">{{ data.summary.passed }}</div></div>
      <div class="stat"><div class="label">Utilidad&nbsp;fallo</div><div class="value">{{ data.summary.failed }}</div></div>
      <div class="stat"><div class="label">Privacidad&nbsp;(n/a)</div><div class="value">{{ data.summary.reported }}</div></div>
      <div class="stat"><div class="label">Tiempo fit</div><div class="value">{{ "%.1fs"|format(data.timings.get('fit_seconds', 0.0)) }}</div></div>
    </div>
  </div>

  {% set mech = data.privacy_mechanism.configured %}
    {% set acc = data.privacy_mechanism.accountant %}
  <div class="card">
    <h2>Mecanismo de privacidad</h2>
    {% if mech.dp %}
      {% if acc.effective_epsilon is not none %}
        <p><span class="pill passed">DP activo</span> Epsilon acumulado real: <strong>{{ acc.effective_epsilon }}</strong> (presupuesto {{ mech.epsilon }}; ruido = {{ acc.noise_multiplier }})</p>
        {% if acc.ecdf_epsilon is not none %}
          <p><span class="pill passed">DP-ECDF activo</span> Epsilon total (entrenamiento + marginales): <strong>{{ "%.4f"|format(acc.total_epsilon) }}</strong>
          = {{ "%.4f"|format(acc.effective_epsilon) }} (DP-SGD) + {{ acc.ecdf_epsilon }} (DP-ECDF)</p>
        {% endif %}
      {% else %}
        <p><span class="pill reported">DP declarado, no medido</span> Presupuesto epsilon: {{ mech.epsilon }} (delta {{ mech.delta }}). El epsilon acumulado real solo se obtiene entrenando con synthpriv.</p>
      {% endif %}
    {% else %}
      <p><span class="pill reported">Sin garantia formal de DP</span> {{ mech.notes }}</p>
    {% endif %}
    {% set assurance = data.privacy_mechanism.assurance %}
    {% if assurance %}
      <p><span class="pill {{ 'passed' if assurance.status == 'ok' else 'failed' }}">DP {{ assurance.status }}</span> {{ assurance.message }}</p>
    {% endif %}
  </div>

  <div class="card">
    <h2>Utilidad</h2>
    <table>
      <tr><th>Metrica</th><th>Valor</th><th>Umbral</th><th>Estado</th></tr>
      {% for name, m in data.utility.items() %}
      <tr>
        <td><strong>{{ name }}</strong><br><span class="muted">{{ m.description }}</span></td>
        <td>{{ m.value }}</td>
        <td>{{ m.threshold }}</td>
        <td><span class="pill {{ m.status }}">{{ m.status }}</span></td>
      </tr>
      {% if m.details %}
      <tr><td colspan="4" class="muted">{{ details_cells(m.details) }}</td></tr>
      {% endif %}
      {% endfor %}
    </table>
  </div>

  <div class="card">
    <h2>Privacidad (riesgo de re-identificacion)</h2>
    <table>
      <tr><th>Metrica</th><th>Valor</th><th>Umbral</th><th>Estado</th></tr>
      {% for name, m in data.privacy_metrics.items() %}
      <tr>
        <td><strong>{{ name }}</strong><br><span class="muted">{{ m.description }}</span></td>
        <td>{{ m.value }}</td>
        <td>{{ m.threshold }}</td>
        <td><span class="pill {{ m.status }}">{{ m.status }}</span></td>
      </tr>
      {% if m.details %}
      <tr><td colspan="4" class="muted">{{ details_cells(m.details) }}</td></tr>
      {% endif %}
      {% endfor %}
    </table>
  </div>

  <footer>Generado con synthpriv. Recuerda: los datos sinteticos sin DP no son anonimizacion garantizada.</footer>
</main>
</body>
</html>
"""


def details_cells(details: dict[str, Any]) -> str:
    """Serializa detalles de metricas en una linea legible."""
    items = []
    for k, v in details.items():
        if isinstance(v, (list, dict)) and v:
            continue  # tablas detalladas se omiten en el resumen
        items.append(f"{k}={v}")
    return html.escape(" | ".join(items))


_TMPL = Template(_TEMPLATE)
_TMPL.globals["details_cells"] = details_cells


def render_html(
    data: dict[str, Any],
    path: str | Path,
) -> Path:
    """Renderiza ``data`` (salida de ``EvaluationReport.data``) a HTML."""
    doc = _TMPL.render(data=data)
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Informe del barrido epsilon-utilidad
# ---------------------------------------------------------------------------

_SWEEP_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>synthpriv - Barrido epsilon/utilidad</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin:0; color:#1f2328; background:#f6f8fa; }
  header { background:#24292f; color:#fff; padding:20px 28px; }
  header h1 { margin:0 0 6px; font-size:22px; }
  main { max-width:1000px; margin:24px auto; padding:0 16px; }
  .card { background:#fff; border:1px solid #d0d7de; border-radius:8px; padding:18px 22px; margin-bottom:18px; }
  .card h2 { margin:0 0 12px; font-size:17px; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid #d0d7de; }
  th { color:#59636e; font-size:12px; text-transform:uppercase; }
  svg { width:100%; height:auto; }
  .chart-title { font-size:14px; font-weight:600; margin:14px 0 4px; }
  .muted { color:#59636e; font-size:13px; }
  footer { color:#59636e; font-size:12px; text-align:center; padding:18px; }
</style>
</head>
<body>
<header>
  <h1>Barrido epsilon vs utilidad</h1>
  <p>Generador: dp-gan &middot; Puntos: {{ rows|length }} &middot; Delta: {{ delta }}</p>
</header>
<main>
  <div class="card">
    <h2>Tabla</h2>
    <table>
      <tr>
        {% for col in columns %}<th>{{ col }}</th>{% endfor %}
      </tr>
      {% for row in rows %}
      <tr>
        {% for col in columns %}<td>{{ row.get(col) }}</td>{% endfor %}
      </tr>
      {% endfor %}
    </table>
  </div>

  {% for chart in charts %}
  <div class="card">
    <div class="chart-title">{{ chart.title }}</div>
    <p class="muted">{{ chart.description }}</p>
    {{ chart.svg_html|safe }}
  </div>
  {% endfor %}
</main>
<footer>Generado con synthpriv. Eje X: epsilon acumulado real (accountant RDP).</footer>
</body>
</html>
"""

_SWEEP_TMPL = Template(_SWEEP_TEMPLATE)


def _svg_line_chart(xs: list[float], ys: list[float], label_x: str, label_y: str,
                    refs: list[tuple[str, float]] | None = None,
                    width: int = 640, height: int = 240) -> str:
    """Curva SVG 0-100% con polyline, puntos y lineas de referencia (refs)."""
    if not xs:
        return ""
    refs = refs or []
    xmin, xmax = min(xs), max(xs)
    y_vals = list(ys) + [y for _, y in refs]
    ymin, ymax = min(y_vals), max(y_vals)
    span_x = (xmax - xmin) or 1.0
    span_y = (ymax - ymin) or 1.0
    pad_x, pad_y = 46, 28

    def px(x: float) -> float:
        return pad_x + (x - xmin) / span_x * (width - 2 * pad_x)

    def py(y: float) -> float:
        return (height - pad_y) - (y - ymin) / span_y * (height - 2 * pad_y)

    lines = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    dots = "".join(
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="#1a7f37"/>'
        for x, y in zip(xs, ys)
    )
    ref_lines = "".join(
        f'<line x1="{pad_x}" y1="{py(y):.1f}" x2="{width - pad_x}" y2="{py(y):.1f}" '
        f'stroke="#d4a72c" stroke-dasharray="4 3"/>'
        f'<text x="{width - pad_x - 2}" y="{max(py(y) - 4, 10):.1f}" font-size="10" '
        f'fill="#9a6700" text-anchor="end">{label} {y:g}</text>'
        for label, y in refs
    )
    ticks = "".join(
        f'<text x="{pad_x + (x - xmin) / span_x * (width - 2 * pad_x):.1f}" y="{height - pad_y + 16}" '
        f'font-size="11" fill="#59636e" text-anchor="middle">{x:g}</text>'
        for x in [xmin, (xmin + xmax) / 2, xmax]
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="curva {label_y} vs {label_x}">'
        f'<line x1="{pad_x}" y1="{height - pad_y}" x2="{width - pad_x}" y2="{height - pad_y}" '
        f'stroke="#d0d7de"/>'
        f'<text x="{width / 2}" y="{height - 6}" font-size="12" fill="#59636e" text-anchor="middle">'
        f'{label_x}</text>'
        f'<text x="12" y="16" font-size="12" fill="#59636e">{label_y}</text>'
        f'{ref_lines}'
        f'<polyline points="{lines}" fill="none" stroke="#1f883d" stroke-width="2"/>'
        f'{ticks}{dots}</svg>'
    )


def render_sweep_html(result, path: str | Path) -> Path:
    """Renderiza un ``SweepResult`` a HTML autocontenido con las curvas."""
    from synthpriv.sweep import SweepResult

    assert isinstance(result, SweepResult), "se esperaba un SweepResult"
    df = result.dataframe()
    rows = df.to_dict("records")
    columns = list(df.columns)

    charts = []
    for metric in df.select_dtypes(include=["number"]).columns:
        if metric in ("target_epsilon", "measured_epsilon", "delta", "fit_seconds"):
            continue
        sub = df.dropna(subset=["measured_epsilon", metric])
        if len(sub) < 2:
            continue
        charts.append({
            "title": metric.replace("util_", "Utilidad: ").replace("priv_", "Privacidad: "),
            "description": "Valor de la metrica frente al epsilon acumulado real (menor epsilon = mas privado).",
            "svg_html": _svg_line_chart(
                list(sub["measured_epsilon"]), list(sub[metric]),
                "epsilon acumulado real", metric,
            ),
        })

    html_doc = _SWEEP_TMPL.render(rows=rows, columns=columns, charts=charts,
                                  delta=result.rows[0].get("delta", "-") if result.rows else "-")
    path = Path(path)
    path.write_text(html_doc, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Informe del benchmark dp-gan vs baselines (sin DP)
# ---------------------------------------------------------------------------

_BENCHMARK_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>synthpriv - Benchmark dp-gan vs baselines</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin:0; color:#1f2328; background:#f6f8fa; }
  header { background:#24292f; color:#fff; padding:20px 28px; }
  header h1 { margin:0 0 6px; font-size:22px; }
  header p { margin:2px 0; opacity:.85; font-size:14px; }
  main { max-width:1000px; margin:24px auto; padding:0 16px; }
  .card { background:#fff; border:1px solid #d0d7de; border-radius:8px; padding:18px 22px; margin-bottom:18px; }
  .card h2 { margin:0 0 12px; font-size:17px; }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th, td { text-align:left; padding:7px 10px; border-bottom:1px solid #d0d7de; }
  th { color:#59636e; font-size:12px; text-transform:uppercase; }
  svg { width:100%; height:auto; }
  .chart-title { font-size:14px; font-weight:600; margin:14px 0 4px; }
  .muted { color:#59636e; font-size:13px; }
  footer { color:#59636e; font-size:12px; text-align:center; padding:18px; }
</style>
</head>
<body>
<header>
  <h1>Benchmark dp-gan (DP) vs baselines SDV</h1>
  <p>Baselines sin DP: {{ baselines|join(", ") }} &middot; Delta: {{ delta }}</p>
</header>
<main>
  <div class="card">
    <h2>Tabla (por punto de epsilon / baseline)</h2>
    <p class="muted">Las lineas discontinuas del mismo color en las curvas son los valores de cada baseline.</p>
    <table>
      <tr>{% for col in columns %}<th>{{ col }}</th>{% endfor %}</tr>
      {% for row in rows %}
      <tr>{% for col in columns %}<td>{{ row.get(col) }}</td>{% endfor %}</tr>
      {% endfor %}
    </table>
  </div>

  {% for chart in charts %}
  <div class="card">
    <div class="chart-title">{{ chart.title }}</div>
    <p class="muted">{{ chart.description }}</p>
    {{ chart.svg_html|safe }}
  </div>
  {% endfor %}
</main>
<footer>Generado con synthpriv. Eje X: epsilon acumulado real (accountant RDP). Menor epsilon = mas privado.</footer>
</body>
</html>
"""

_BENCHMARK_TMPL = Template(_BENCHMARK_TEMPLATE)


def render_benchmark_html(result, path: str | Path) -> Path:
    """Renderiza un ``BenchmarkResult`` a HTML con curvas dp-gan y refs de baselines."""
    from synthpriv.benchmark import BenchmarkResult

    assert isinstance(result, BenchmarkResult), "se esperaba un BenchmarkResult"
    df = result.dataframe()
    rows = df.to_dict("records")
    columns = list(df.columns)

    charts = []
    metric_columns = [
        c for c in df.columns
        if (c.startswith("util_") or c.startswith("priv_"))
    ]
    for metric in metric_columns:
        curve = result.curve("dp-gan", metric)
        if len(curve) < 2:
            continue
        refs = [
            (b, result.baseline_value(b, metric))
            for b in result.baselines
            if result.baseline_value(b, metric) is not None
        ]
        title = metric.replace("util_", "Utilidad: ").replace("priv_", "Privacidad: ")
        charts.append({
            "title": title,
            "description": "dp-gan (verde) frente al valor de cada baseline sin DP (discontinuo). "
                           "Si a epsilon alto el dp-gan no alcanza el baseline, la arquitectura limita; "
                           "la distancia a epsilon bajo es el coste de la privacidad.",
            "svg_html": _svg_line_chart(
                [p["x"] for p in curve], [p["y"] for p in curve],
                "epsilon acumulado real", metric, refs=refs,
            ),
        })

    html_doc = _BENCHMARK_TMPL.render(
        rows=rows, columns=columns, charts=charts,
        baselines=result.baselines,
        delta=result.rows[0].get("delta", "-") if result.rows else "-",
    )
    path = Path(path)
    path.write_text(html_doc, encoding="utf-8")
    return path