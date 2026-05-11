#!/usr/bin/env python3
"""
Load a sink graph from JSON (NetworkX node-link format, as written by
``example/build_sink_graph.py``) and write an interactive HTML figure (Plotly).

Event nodes show an **Event summary**, then a **Registry JSON** block: full
:class:`~cula.models.EventInfo` when ``registry`` is present in the file, or a
**graph-fields-only** JSON (and a re-export hint) for slim exports such as
``gnn/data/sinks/*.json``. On load, ``registry`` is re-copied from the JSON
``nodes`` list onto the NetworkX graph so it cannot be dropped on round-trip.

Default layout is **flow**: sink on the **far right**, other nodes in columns to the
left by **undirected graph distance** from the sink (same idea as
``example/visualize_sink_graph.py``), with light barycentric ordering within layers.

Edge traces use ``hoverinfo='none'`` (not ``skip``) so line segments are not
chosen for ``hovermode='closest'`` with an empty tooltip. **Sink** and **event**
nodes also get a final transparent marker layer (full opacity, alpha-0 fill) so
dense edges cannot steal the hover hit target while Plotly.js still emits hovers.

**Click** a node to show only that node, its incident edges, and its neighbors
(double-click the plot to restore the full graph).

Requires: pip install -e ".[viz]"  (plotly + networkx)
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from collections import defaultdict, deque
from pathlib import Path
from statistics import median
from typing import Any

import networkx as nx

try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # type: ignore[assignment]


_KIND_COLOR: dict[str, str] = {
    "sink": "#c62828",
    "org": "#1565c0",
    "site": "#2e7d32",
    "material": "#6a1b9a",
    "container": "#ef6c00",
    "event": "#00838f",
    "lca_db": "#5d4037",
    "lca_activity": "#78909c",
}


def load_graph(path: Path) -> nx.DiGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.node_link_graph(data, directed=data.get("directed", True))
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)
    # Re-apply every attribute from the JSON ``nodes`` entries onto ``G`` (except
    # ``id``) so ``registry``, ``step_location``, and other nested dicts survive.
    for node in data.get("nodes") or []:
        nid = node.get("id")
        if nid is None:
            continue
        ns = str(nid)
        if ns not in G:
            continue
        for k, v in node.items():
            if k == "id":
                continue
            G.nodes[ns][k] = v
    return G


def _escape_for_pre(text: str) -> str:
    """
    Escape only characters that break HTML / XSS. Do **not** escape double quotes,
    so JSON in hovers stays readable (``html.escape`` would turn ``"`` into ``&quot;``).
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _json_hover_html(data: Any, max_chars: int) -> str:
    """
    Pretty-print *data* as JSON for Plotly HTML hovers.

    Newlines become ``<br>`` (Plotly collapses bare ``\\n``). Only ``&``, ``<``,
    ``>`` are escaped so JSON quotes stay readable.
    """
    s = json.dumps(data, indent=2, ensure_ascii=False)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n… [truncated]"
    esc = _escape_for_pre(s)
    return esc.replace("\n", "<br>")


def _registry_hover_fragment(registry: Any, max_chars: int) -> str:
    if registry is None:
        return "<i>(no registry payload)</i>"
    return _json_hover_html(registry, max_chars)


def _event_graph_payload_dict(attrs: dict[str, Any]) -> dict[str, Any]:
    """Subset of node attrs as JSON when API ``registry`` is missing from the file."""
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if k == "registry" and v is None:
            continue
        if callable(v):
            continue
        try:
            json.dumps(v)
        except TypeError:
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _registry_json_hover_section(
    kind: str, attrs: dict[str, Any], max_chars: int
) -> str:
    """
    Labelled JSON block for hovers. Events always get JSON (full or graph-only).
    """
    if kind == "event":
        reg = attrs.get("registry")
        if isinstance(reg, dict) and len(reg) > 0:
            return (
                "<b>Full registry (EventInfo)</b><br>"
                + _json_hover_html(reg, max_chars)
            )
        payload = _event_graph_payload_dict(attrs)
        if not payload:
            return (
                "<b>Registry JSON</b><br><i>No data on this event node. "
                "Re-export with</i> "
                "<code>python example/build_sink_graph.py -o sink.json</code>"
            )
        hint = (
            "<i>This file has no API ``registry`` on events — only graph fields below. "
            "For proofs, emissions, and links run "
            "<code>example/build_sink_graph.py</code>.</i><br><br>"
        )
        return (
            "<b>Registry JSON (graph fields only)</b><br>"
            + hint
            + _json_hover_html(payload, max_chars)
        )
    reg = attrs.get("registry")
    if reg is not None:
        return "<b>Registry JSON</b><br>" + _json_hover_html(reg, max_chars)
    return "<i>(no registry payload)</i>"


def _event_node_hover_fragment(node_attrs: dict[str, Any]) -> str:
    """
    HTML fragment summarizing top-level event fields from :mod:`cula.sink_graph`
    (and placeholder nodes). Safe for Plotly ``hovertext``.
    """
    lines: list[str] = []

    if node_attrs.get("placeholder"):
        ref = node_attrs.get("missing_event_ref", "")
        lines.append("<b>Event placeholder</b> (no API payload for this id)")
        lines.append(f"<b>missing_event_ref:</b> {html.escape(str(ref))}")
        return "<br>".join(lines)

    def line(title: str, value: Any, *, fmt: str | None = None) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            text = "yes" if value else "no"
        elif fmt and isinstance(value, (int, float)):
            text = fmt % value
        else:
            text = str(value)
        lines.append(f"<b>{html.escape(title)}:</b> {html.escape(text)}")

    lines.append("<b>Event summary</b>")
    line("Lifecycle", node_attrs.get("lifecycle_type"))
    line("Interaction", node_attrs.get("interaction_type"))
    line("Graph event type", node_attrs.get("graph_event_type"))
    line("Step execution", node_attrs.get("step_execution_type"))
    line("Event ref", node_attrs.get("event_ref"))
    line("Created (ms)", node_attrs.get("created_ms"))
    line("Contribution factor", node_attrs.get("contribution_factor"), fmt="%.6f")
    line("Graph completed", node_attrs.get("event_graph_completed"))
    line("Proofs", node_attrs.get("proof_count"))
    line("Waypoints", node_attrs.get("waypoint_count"))
    line("Predecessor links", node_attrs.get("predecessor_link_count"))
    line("Emissions (kg CO2e)", node_attrs.get("emissions_total_kg_co2e"), fmt="%.6f")
    line("Emissions root id", node_attrs.get("emissions_root_id"))
    line("Payload containers", node_attrs.get("payload_container_count"))
    line("Input containers", node_attrs.get("input_container_count"))
    line("Output containers", node_attrs.get("output_container_count"))
    line("LCA activities (on event)", node_attrs.get("lca_activity_count"))
    line("Sink matrix", node_attrs.get("has_sink_matrix"))
    line("Transport (km)", node_attrs.get("transportation_distance_km"), fmt="%.4f")
    line("Sender site", node_attrs.get("sender_site_id"))
    line("Receiver site", node_attrs.get("receiver_site_id"))
    line("Step site", node_attrs.get("step_site_id"))

    loc = node_attrs.get("step_location")
    if isinstance(loc, dict):
        lat, lng = loc.get("lat"), loc.get("long")
        city = loc.get("city")
        street = loc.get("street")
        bits: list[str] = []
        if lat is not None and lng is not None:
            bits.append(f"{float(lat):.5f}, {float(lng):.5f}")
        if city:
            bits.append(str(city))
        if street:
            bits.append(str(street))
        if bits:
            lines.append("<b>Step location:</b> " + html.escape(" · ".join(bits)))

    if node_attrs.get("event_payload_missing"):
        lines.append("<i>Event payload missing on EventInfo</i>")

    if len(lines) <= 1:
        return ""
    return "<br>".join(lines)


def _node_hover_html(
    G: nx.DiGraph,
    n: Any,
    kind: str,
    *,
    show_registry_hover: bool,
    max_registry_chars: int,
    extra_hover: dict[str, str],
) -> str:
    attr = dict(G.nodes[n])
    nid = str(n)
    label = str(attr.get("label", ""))
    head = f"<b>{html.escape(label)}</b><br>kind: {html.escape(kind)}<br>id: {html.escape(nid)}"
    if nid in extra_hover:
        head += f"<br>{extra_hover[nid]}"
    if kind == "event":
        ev_part = _event_node_hover_fragment(attr)
        if ev_part:
            head += f"<br><br>{ev_part}"
    if show_registry_hover and kind not in ("event", "sink"):
        body = _registry_json_hover_section(kind, attr, max_registry_chars)
        head += f"<br><br>{body}"
    if len(head) > 1000:
        head = head[:1000] + "<br>… [truncated]"
    return head


def _find_sink_node(G: nx.DiGraph) -> Any:
    sinks = [n for n in G.nodes if G.nodes[n].get("kind") == "sink"]
    if len(sinks) != 1:
        raise ValueError(f"flow layout needs exactly one sink node; found {len(sinks)}")
    return sinks[0]


def flow_positions_sink_right(G: nx.DiGraph) -> dict[Any, tuple[float, float]]:
    """
    Sink on the far right; every other node's layer is the length of its longest
    directed path to the sink (guarantees predecessors are always left of their
    successors). Vertical order: barycentric sweeps across adjacent layers.
    """
    sink_node = _find_sink_node(G)

    # Longest directed path from each node to sink — process in reverse
    # topological order so that every successor is resolved before its
    # predecessors.
    dist: dict[Any, int] = {}
    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        topo = list(G.nodes)
    for n in reversed(topo):
        if n == sink_node:
            dist[n] = 0
        else:
            child_dists = [dist[s] for s in G.successors(n) if s in dist and dist[s] >= 0]
            dist[n] = max(child_dists) + 1 if child_dists else -1

    # Nodes that cannot reach the sink via directed edges (e.g. leaf
    # containers): place them one step to the right of their leftmost
    # predecessor so the predecessor→leaf direction is respected.
    # Process in topological order so predecessors are resolved first.
    for n in topo:
        if dist.get(n, -1) >= 0:
            continue
        pred_dists = [dist[p] for p in G.predecessors(n) if dist.get(p, -1) >= 0]
        if pred_dists:
            dist[n] = min(pred_dists) - 1
        else:
            dist[n] = -1

    # Anything still unreachable: fall back to undirected BFS distance.
    unreachable = [n for n in G.nodes if dist.get(n, -1) < 0]
    if unreachable:
        undirected = G.to_undirected()
        bfs_dist: dict[Any, int] = {}
        q: deque[Any] = deque([sink_node])
        bfs_dist[sink_node] = 0
        while q:
            cur = q.popleft()
            for nb in undirected.neighbors(cur):
                if nb not in bfs_dist:
                    bfs_dist[nb] = bfs_dist[cur] + 1
                    q.append(nb)
        max_reachable = max((d for d in dist.values() if d >= 0), default=0)
        for n in unreachable:
            dist[n] = bfs_dist.get(n, max_reachable + 1)

    # Shift all layers so the minimum is 0.
    min_d = min(dist.values()) if dist else 0
    if min_d < 0:
        for n in dist:
            dist[n] -= min_d

    max_d = max(dist.values()) if dist else 0

    layers: dict[int, list[Any]] = defaultdict(list)
    for n, d in dist.items():
        layers[d].append(n)

    dx = 3.0
    dy = 0.95

    def sort_key(n: Any) -> tuple[str, str]:
        return (str(G.nodes[n].get("kind", "")), str(G.nodes[n].get("label", "")))

    for d in layers:
        layers[d].sort(key=sort_key)

    undirected = G.to_undirected()
    layer_ids = sorted(layers.keys())
    index_in_layer: dict[Any, tuple[int, int]] = {}
    for d, nodes in layers.items():
        for i, n in enumerate(nodes):
            index_in_layer[n] = (d, i)

    for _ in range(6):
        for d in layer_ids:
            nodes = layers[d]
            scores: list[tuple[float, int, Any]] = []
            for i, n in enumerate(nodes):
                neighbor_indices: list[float] = []
                for nb in undirected.neighbors(n):
                    if nb in index_in_layer:
                        nd, ni = index_in_layer[nb]
                        if abs(nd - d) == 1:
                            neighbor_indices.append(float(ni))
                b = median(neighbor_indices) if neighbor_indices else float(i)
                scores.append((b, i, n))
            scores.sort(key=lambda t: (t[0], t[1]))
            new_order = [t[2] for t in scores]
            layers[d] = new_order
            for i, n in enumerate(new_order):
                index_in_layer[n] = (d, i)

    pos: dict[Any, tuple[float, float]] = {}
    for d in layer_ids:
        nodes = layers[d]
        width = max(0.0, (len(nodes) - 1) * dy)
        for i, n in enumerate(nodes):
            x = (max_d - d) * dx
            y = width / 2 - i * dy
            pos[n] = (float(x), float(y))
    return pos


def layout_positions(
    G: nx.DiGraph,
    algorithm: str,
    *,
    seed: int,
    iterations: int,
) -> dict[Any, tuple[float, float]]:
    n = G.number_of_nodes()
    if n == 0:
        return {}
    if algorithm == "flow":
        try:
            return flow_positions_sink_right(G)
        except ValueError as e:
            print(f"Warning: {e}; using spring layout instead.", file=sys.stderr)
            return nx.spring_layout(G, seed=seed, iterations=iterations)
    if algorithm == "spring":
        return nx.spring_layout(G, seed=seed, iterations=iterations)
    if algorithm == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    if algorithm == "circular":
        return nx.circular_layout(G)
    if algorithm == "spectral":
        return nx.spectral_layout(G)
    raise ValueError(f"Unknown layout: {algorithm!r}")


def _focus_post_script(ctx: dict[str, Any]) -> str:
    """JavaScript appended after Plotly init; ``{plot_id}`` is replaced by Plotly."""
    payload = json.dumps(ctx, separators=(",", ":")).replace("<", "\\u003c")
    return (
        "(function(){"
        "var D="
        + payload
        + ";"
        "var gd=document.getElementById('{plot_id}');"
        "if(!gd||!D)return;"
        "function visN(nid){var s={};s[nid]=1;var a=D.adj[nid];"
        "if(a){for(var i=0;i<a.length;i++)s[a[i]]=1;}return s;}"
        "function filtSegs(segs,nid){var X=[],Y=[],i,s;"
        "for(i=0;i<segs.length;i++){s=segs[i];"
        "if(s.u===nid||s.v===nid){X.push(s.xs[0],s.xs[1],null);"
        "Y.push(s.ys[0],s.ys[1],null);}}return[X,Y];}"
        "function applyFocus(nid){if(!nid)return;nid=String(nid);var v=visN(nid),xy,i,t,nt,op,sz,j;"
        "if(D.edgeGray){xy=filtSegs(D.edgeGray.segments,nid);"
        "Plotly.restyle(gd,{x:[xy[0]],y:[xy[1]]},[D.edgeGray.curve]);}"
        "if(D.edgeRed){xy=filtSegs(D.edgeRed.segments,nid);"
        "Plotly.restyle(gd,{x:[xy[0]],y:[xy[1]]},[D.edgeRed.curve]);}"
        "for(t=0;t<D.nodeTraces.length;t++){nt=D.nodeTraces[t];op=[];sz=[];"
        "for(j=0;j<nt.ids.length;j++){op.push(v[nt.ids[j]]?nt.fullOpacity[j]:0);"
        "sz.push(v[nt.ids[j]]?nt.fullSize[j]:0.02);}"
        "Plotly.restyle(gd,{'marker.opacity':[op],'marker.size':[sz]},[nt.curve]);}"
        "if(D.pick){nt=D.pick;op=[];sz=[];"
        "for(j=0;j<nt.ids.length;j++){op.push(v[nt.ids[j]]?nt.fullOpacity[j]:0);"
        "sz.push(v[nt.ids[j]]?nt.fullSize[j]:0.02);}"
        "Plotly.restyle(gd,{'marker.opacity':[op],'marker.size':[sz]},[nt.curve]);}"
        "if(D.supply){nt=D.supply;op=[];sz=[];"
        "for(j=0;j<nt.pairs.length;j++){var on=(nt.pairs[j][0]===nid||nt.pairs[j][1]===nid);"
        "op.push(on?nt.fullOpacity[j]:0);sz.push(on?nt.fullSize[j]:0.02);}"
        "Plotly.restyle(gd,{'marker.opacity':[op],'marker.size':[sz]},[nt.curve]);}"
        "}"
        "function resetFocus(){var t,nt;"
        "if(D.edgeGray)Plotly.restyle(gd,{x:[D.edgeGray.fullX],y:[D.edgeGray.fullY]},[D.edgeGray.curve]);"
        "if(D.edgeRed)Plotly.restyle(gd,{x:[D.edgeRed.fullX],y:[D.edgeRed.fullY]},[D.edgeRed.curve]);"
        "for(t=0;t<D.nodeTraces.length;t++){nt=D.nodeTraces[t];"
        "Plotly.restyle(gd,{'marker.opacity':[nt.fullOpacity],'marker.size':[nt.fullSize]},[nt.curve]);}"
        "if(D.pick)Plotly.restyle(gd,{'marker.opacity':[D.pick.fullOpacity],'marker.size':[D.pick.fullSize]},[D.pick.curve]);"
        "if(D.supply)Plotly.restyle(gd,{'marker.opacity':[D.supply.fullOpacity],'marker.size':[D.supply.fullSize]},[D.supply.curve]);"
        "}"
        "gd.on('plotly_click',function(ev){"
        "if(!ev.points||!ev.points.length)return;"
        "var pt=ev.points[0],cd,m;"
        "if(!pt.data)return;m=pt.data.mode||'';"
        "if(m.indexOf('markers')<0)return;"
        "cd=pt.customdata;if(cd==null||cd==='')return;"
        "if(Array.isArray(cd))cd=cd[0];if(cd==null||cd==='')return;"
        "applyFocus(String(cd));});"
        "gd.on('plotly_doubleclick',function(){resetFocus();});"
        "})();"
    )


def build_figure(
    G: nx.DiGraph,
    pos: dict[Any, tuple[float, float]],
    *,
    max_registry_chars: int,
    show_registry_hover: bool,
    show_supply_chain_edge_hover: bool,
    max_edge_data_chars: int,
    figure_title: str | None = None,
    flagged_nodes: set[str] | None = None,
    flagged_edges: set[tuple[str, str]] | None = None,
    gnn_node_hover_extra: dict[str, str] | None = None,
) -> tuple[go.Figure, dict[str, Any]]:
    fn = flagged_nodes or set()
    fe = flagged_edges or set()
    extra_hover = gnn_node_hover_extra or {}

    adj: dict[str, list[str]] = {}
    for n in G.nodes():
        ns = str(n)
        adj[ns] = sorted({str(x) for x in nx.all_neighbors(G, n)})

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_hi_x: list[float | None] = []
    edge_hi_y: list[float | None] = []
    segments_gray: list[dict[str, Any]] = []
    segments_red: list[dict[str, Any]] = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        segx = (float(x0), float(x1), None)
        segy = (float(y0), float(y1), None)
        su, sv = str(u), str(v)
        seg = {
            "u": su,
            "v": sv,
            "xs": [float(x0), float(x1), None],
            "ys": [float(y0), float(y1), None],
        }
        pair = (su, sv)
        if pair in fe:
            edge_hi_x.extend(segx)
            edge_hi_y.extend(segy)
            segments_red.append(seg)
        else:
            edge_x.extend(segx)
            edge_y.extend(segy)
            segments_gray.append(seg)

    traces: list[go.Scatter] = []
    curve = 0
    edge_gray_curve: int | None = None
    edge_red_curve: int | None = None
    if edge_x:
        edge_gray_curve = curve
        curve += 1
        traces.append(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.4, color="rgba(120,120,120,0.35)"),
                # ``skip`` still participates in ``hovermode=closest`` and can win over
                # nearby markers (hub sink / events) with no tooltip — use ``none``.
                hoverinfo="none",
                showlegend=False,
            )
        )
    if edge_hi_x:
        edge_red_curve = curve
        curve += 1
        traces.append(
            go.Scatter(
                x=edge_hi_x,
                y=edge_hi_y,
                mode="lines",
                line=dict(width=2.8, color="rgba(211,47,47,0.85)"),
                hoverinfo="none",
                name="GNN top anomaly edges",
                showlegend=True,
            )
        )

    # Plot ``event`` (and ``sink``) last so their markers sit on top and receive hovers
    # (default alphabetical order draws material/site/… after event and steals picks).
    kinds_set = {str(G.nodes[n].get("kind", "unknown")) for n in G.nodes()}
    kinds = sorted(k for k in kinds_set if k not in ("event", "sink"))
    if "event" in kinds_set:
        kinds.append("event")
    if "sink" in kinds_set:
        kinds.append("sink")
    node_trace_specs: list[dict[str, Any]] = []
    for kind in kinds:
        nodes_k = [n for n in G.nodes() if str(G.nodes[n].get("kind", "unknown")) == kind]
        xs = [float(pos[n][0]) for n in nodes_k]
        ys = [float(pos[n][1]) for n in nodes_k]
        hover: list[str] = []
        sizes: list[float] = []
        line_widths: list[float] = []
        line_colors: list[str] = []
        ids_k: list[str] = []
        for i, n in enumerate(nodes_k):
            nid = str(nodes_k[i])
            ids_k.append(nid)
            hover.append(
                _node_hover_html(
                    G,
                    n,
                    kind,
                    show_registry_hover=show_registry_hover,
                    max_registry_chars=max_registry_chars,
                    extra_hover=extra_hover,
                )
            )
            if nid in fn:
                sizes.append(15.0)
                line_widths.append(2.5)
                line_colors.append("#b71c1c")
            else:
                sizes.append(9.0)
                line_widths.append(0.5)
                line_colors.append("#263238")
        node_curve = curve
        curve += 1
        node_trace_specs.append(
            {
                "curve": node_curve,
                "ids": ids_k,
                "fullOpacity": [1.0] * len(ids_k),
                "fullSize": list(sizes),
            }
        )
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=kind,
                marker=dict(
                    size=sizes,
                    color=_KIND_COLOR.get(kind, "#546e7a"),
                    line=dict(width=line_widths, color=line_colors),
                ),
                text=hover,
                hoverinfo="text",
                customdata=ids_k,
            )
        )

    supply_ctx: dict[str, Any] | None = None
    if show_supply_chain_edge_hover:
        ex: list[float] = []
        ey: list[float] = []
        ehover: list[str] = []
        pairs_sc: list[list[str]] = []
        for u, v, d in G.edges(data=True):
            if d.get("relation") != "supply_chain":
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            su, sv = str(u), str(v)
            ex.append((float(x0) + float(x1)) / 2)
            ey.append((float(y0) + float(y1)) / 2)
            pairs_sc.append([su, sv])
            payload = d.get("aggregated_links") or []
            body = _json_hover_html(payload, max_edge_data_chars)
            ehover.append(
                f"<b>supply_chain</b><br>{html.escape(su)} → {html.escape(sv)}<br><br>{body}"
            )
        if ex:
            sc_curve = curve
            curve += 1
            sc_n = len(ex)
            supply_ctx = {
                "curve": sc_curve,
                "pairs": pairs_sc,
                "fullOpacity": [1.0] * sc_n,
                "fullSize": [5.0] * sc_n,
            }
            traces.append(
                go.Scatter(
                    x=ex,
                    y=ey,
                    mode="markers",
                    marker=dict(size=5, color="rgba(100,100,100,0.45)", line=dict(width=0)),
                    text=ehover,
                    hoverinfo="text",
                    name="supply_chain (midpoint)",
                    showlegend=False,
                )
            )

    # Sink and events sit on many edges; Plotly still prefers line traces for
    # ``closest`` hover in practice. Extra fully transparent markers (last trace =
    # top layer) widen the hit target with the same tooltip HTML. Use alpha=0 with
    # marker opacity 1 — near-zero opacity traces are ignored for hover in Plotly.js.
    pick_x: list[float] = []
    pick_y: list[float] = []
    pick_text: list[str] = []
    pick_ids: list[str] = []
    for n in G.nodes():
        k = str(G.nodes[n].get("kind", "unknown"))
        if k not in ("event", "sink"):
            continue
        pick_x.append(float(pos[n][0]))
        pick_y.append(float(pos[n][1]))
        pick_ids.append(str(n))
        pick_text.append(
            _node_hover_html(
                G,
                n,
                k,
                show_registry_hover=show_registry_hover,
                max_registry_chars=max_registry_chars,
                extra_hover=extra_hover,
            )
        )
    pick_ctx: dict[str, Any] | None = None
    if pick_x:
        pick_curve = curve
        curve += 1
        pn = len(pick_x)
        pick_ctx = {
            "curve": pick_curve,
            "ids": pick_ids,
            "fullOpacity": [1.0] * pn,
            "fullSize": [48.0] * pn,
        }
        traces.append(
            go.Scatter(
                x=pick_x,
                y=pick_y,
                mode="markers",
                name="",
                marker=dict(
                    size=48,
                    opacity=1,
                    color="rgba(0,0,0,0)",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ),
                text=pick_text,
                hoverinfo="text",
                showlegend=False,
                customdata=pick_ids,
            )
        )

    focus_ctx: dict[str, Any] = {
        "adj": adj,
        "edgeGray": None,
        "edgeRed": None,
        "nodeTraces": node_trace_specs,
        "pick": pick_ctx,
        "supply": supply_ctx,
    }
    if edge_gray_curve is not None:
        focus_ctx["edgeGray"] = {
            "curve": edge_gray_curve,
            "fullX": edge_x,
            "fullY": edge_y,
            "segments": segments_gray,
        }
    if edge_red_curve is not None:
        focus_ctx["edgeRed"] = {
            "curve": edge_red_curve,
            "fullX": edge_hi_x,
            "fullY": edge_hi_y,
            "segments": segments_red,
        }

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=figure_title or "Sink graph (interactive)",
        showlegend=True,
        hoverlabel=dict(align="left", font_size=12),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=24, r=24, t=48, b=24),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#fafafa",
        dragmode="pan",
        hovermode="closest",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig, focus_ctx


def main() -> int:
    if go is None:
        print('Plotly is required. Install with: pip install -e ".[viz]"', file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        description="Interactive Plotly view of a sink graph JSON (NetworkX node-link)."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("sink.json"),
        help="Node-link JSON path (default: sink.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sink_graph.html"),
        help="Output HTML path (default: sink_graph.html)",
    )
    parser.add_argument(
        "--layout",
        choices=("flow", "spring", "kamada_kawai", "circular", "spectral"),
        default="flow",
        help="Layout: flow = sink far right, layers left by hop distance (default); "
        "or force spring / kamada_kawai / circular / spectral",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=120,
        help="Spring-layout iterations (default: 120)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for spring layout (default: 42)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the HTML file in the default browser after writing",
    )
    parser.add_argument(
        "--no-registry-hover",
        action="store_true",
        help="Omit registry JSON from node hovers (faster rendering for huge payloads)",
    )
    parser.add_argument(
        "--max-registry-chars",
        type=int,
        default=12_000,
        metavar="N",
        help="Max characters of registry JSON per node hover (default: 12000)",
    )
    parser.add_argument(
        "--supply-chain-edge-hover",
        action="store_true",
        help="Add invisible markers at supply_chain edge midpoints with aggregated_links JSON",
    )
    parser.add_argument(
        "--max-edge-data-chars",
        type=int,
        default=4_000,
        metavar="N",
        help="Max characters for supply_chain edge hover payload (default: 4000)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    G = load_graph(args.input)
    if G.number_of_nodes() == 0:
        print("Graph has no nodes.", file=sys.stderr)
        return 1

    pos = layout_positions(
        G,
        args.layout,
        seed=args.seed,
        iterations=args.iterations,
    )
    fig, focus_ctx = build_figure(
        G,
        pos,
        max_registry_chars=args.max_registry_chars,
        show_registry_hover=not args.no_registry_hover,
        show_supply_chain_edge_hover=args.supply_chain_edge_hover,
        max_edge_data_chars=args.max_edge_data_chars,
        figure_title=None,
        flagged_nodes=None,
        flagged_edges=None,
        gnn_node_hover_extra=None,
    )
    fig.write_html(
        args.output,
        include_plotlyjs="cdn",
        config=dict(scrollZoom=True, displaylogo=False),
        post_script=_focus_post_script(focus_ctx),
    )
    print(f"Wrote {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {args.output}")

    if args.open:
        webbrowser.open(args.output.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
