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
      {% else %}
        <p><span class="pill reported">DP declarado, no medido</span> Presupuesto epsilon: {{ mech.epsilon }} (delta {{ mech.delta }}). El epsilon acumulado real solo se obtiene entrenando con synthpriv.</p>
      {% endif %}
    {% else %}
      <p><span class="pill reported">Sin garantia formal de DP</span> {{ mech.notes }}</p>
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


def render_html(data: dict[str, Any], path: str | Path) -> Path:
    """Renderiza ``data`` (salida de ``EvaluationReport.data``) a HTML."""
    doc = _TMPL.render(data=data)
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path