"""
Build a NetworkX directed graph from a :class:`~cula.models.Sink`.

The graph merges:

- Registry structure: organisations, sites, ownership, capture site, primary material.
- Material flow: materials ↔ containers (filled).
- LCA: databases → activities; activities ↔ events where referenced.
- Lifecycle: :attr:`~cula.models.Sink.eventGraph` — edges follow **forward** supply-chain
  direction (predecessor event → successor event via ``relation="supply_chain"``).
  API back-links are reversed so chronology reads left-to-right toward the root event,
  which is linked to the sink node with ``relation="to_sink"``.

Nodes use string keys ``"{kind}:{id}"``. Each node carries:

- ``label`` — short display name.
- ``kind`` — ``sink``, ``org``, ``site``, ``material``, ``container``, ``event``,
  ``lca_db``, ``lca_activity``.
- ``registry`` — JSON-serializable dict from the backing Pydantic model
  (:meth:`~pydantic.BaseModel.model_dump` with ``mode="json"``). The sink node’s
  ``registry`` is the full sink payload (including embedded ``eventGraph``). Event
  nodes use the full :class:`~cula.models.EventInfo` record (event payload, links
  metadata, contribution fields). Placeholder event nodes (missing from the API map)
  have no ``registry`` but carry ``placeholder`` / ``missing_event_ref``.

- **Event nodes (full payload)** — additional top-level scalars for quick access:
  ``event_ref``, ``lifecycle_type``, ``interaction_type`` (delivery vs step),
  ``created_ms``, counts (proofs, waypoints, containers), ``emissions_total_kg_co2e``,
  ``emissions_root_id``, site UUIDs, ``step_location`` (GeoJSON-ish dict),
  graph metadata (``contribution_factor``, ``event_graph_completed``,
  ``graph_event_type``, ``step_execution_type``, ``predecessor_link_count``), etc.

Edges carry ``relation`` plus, for ``supply_chain`` edges, ``aggregated_links``:
a list of serialized :class:`~cula.models.AggregatedSinkEventLink` dicts (several
links can share the same predecessor/target pair).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import networkx as nx
from pydantic import BaseModel

from cula.models import Event, EventInfo, MaterialContainer, Site, Sink


def _registry(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def sink_title(sink: Sink) -> str:
    loc = sink.location
    if loc and loc.city:
        return f"Sink — {loc.city}"
    if sink.id:
        return f"Sink — {str(sink.id)[:8]}…"
    return "Sink"


def _container_label(container: MaterialContainer) -> str:
    names: list[str] = []
    for amount in container.content:
        m = amount.material
        if m and m.name:
            names.append(m.name)
    if names:
        text = ", ".join(names)
        return text if len(text) <= 72 else text[:69] + "…"
    return f"Container {str(container.id)[:8]}…"


def _event_label(event: Event) -> str:
    return event.type.value.replace("_", " ").title()


def _event_node_attrs(info: EventInfo) -> dict[str, Any]:
    """
    Top-level, JSON-friendly fields on event nodes (in addition to ``registry``).

    Mirrors the main :class:`~cula.models.Event` / :class:`~cula.models.EventInfo`
    facts so tools can read summaries without unpacking ``registry``.
    """
    ev = info.event
    if ev is None:
        return {"event_payload_missing": True}

    out: dict[str, Any] = {
        "event_ref": ev.eventRef,
        "lifecycle_type": ev.type.value,
        "interaction_type": ev.eventType.value,
        "created_ms": ev.created,
        "proof_count": len(ev.proofs),
        "waypoint_count": len(ev.wayPoints),
        "emissions_total_kg_co2e": float(ev.emissions.value),
        "emissions_root_id": ev.emissions.id,
        "payload_container_count": len(ev.payload or []),
        "input_container_count": len(ev.input or []),
        "output_container_count": len(ev.output or []),
        "has_sink_matrix": ev.matrixType is not None,
        "lca_activity_count": len(ev.lcaActivities or []),
        "predecessor_link_count": len(info.links or []),
    }

    if ev.transportationDistanceInKm is not None:
        out["transportation_distance_km"] = float(ev.transportationDistanceInKm)
    if ev.senderSiteId is not None:
        out["sender_site_id"] = str(ev.senderSiteId)
    if ev.receiverSiteId is not None:
        out["receiver_site_id"] = str(ev.receiverSiteId)
    if ev.siteId is not None:
        out["step_site_id"] = str(ev.siteId)
    if ev.location is not None:
        out["step_location"] = ev.location.model_dump(mode="json")

    if info.contributionFactor is not None:
        out["contribution_factor"] = float(info.contributionFactor)
    if info.isCompleted is not None:
        out["event_graph_completed"] = bool(info.isCompleted)
    if info.eventType is not None:
        out["graph_event_type"] = info.eventType.value
    if info.stepExecutionType is not None:
        out["step_execution_type"] = info.stepExecutionType.value

    return out


def _ensure_site(
    G: nx.DiGraph,
    site_id: UUID,
    site_labels: dict[UUID, str],
    sites_by_ref: dict[UUID, Site],
) -> str:
    key = f"site:{site_id}"
    if key not in G:
        if site_id in sites_by_ref:
            s = sites_by_ref[site_id]
            G.add_node(key, label=s.name, kind="site", registry=_registry(s))
        else:
            label = site_labels.get(site_id, f"Site {str(site_id)[:8]}…")
            G.add_node(key, label=label, kind="site")
    elif site_id in sites_by_ref and G.nodes[key].get("registry") is None:
        G.nodes[key]["registry"] = _registry(sites_by_ref[site_id])
    return key


def _add_or_update_container(G: nx.DiGraph, c: MaterialContainer) -> str:
    ck = f"container:{c.id}"
    reg = _registry(c)
    if ck not in G:
        G.add_node(ck, label=_container_label(c), kind="container", registry=reg)
    elif G.nodes[ck].get("registry") is None:
        G.nodes[ck]["registry"] = reg
    return ck


def _add_supply_chain_edge(G: nx.DiGraph, pred: str, ek: str, link_dump: dict[str, Any]) -> None:
    if G.has_edge(pred, ek):
        edge = G.edges[pred, ek]
        lst = edge.get("aggregated_links")
        if lst is None:
            lst = []
            edge["aggregated_links"] = lst
        lst.append(link_dump)
        edge["relation"] = "supply_chain"
    else:
        G.add_edge(pred, ek, relation="supply_chain", aggregated_links=[link_dump])


def build_entity_graph(sink: Sink) -> nx.DiGraph:
    """
    Construct a :class:`networkx.DiGraph` for *sink*.

    Requires ``sink.id`` to be set. Raises :class:`ValueError` otherwise.
    """
    G = nx.DiGraph()

    if not sink.id:
        raise ValueError("Sink has no id")

    sink_key = f"sink:{sink.id}"
    G.add_node(sink_key, label=sink_title(sink), kind="sink", registry=_registry(sink))

    site_labels: dict[UUID, str] = {}
    sites_by_ref: dict[UUID, Site] = {}
    for s in sink.sites or []:
        site_labels[s.siteRef] = s.name
        sites_by_ref[s.siteRef] = s
        key = f"site:{s.siteRef}"
        G.add_node(key, label=s.name, kind="site", registry=_registry(s))

    org_keys: dict[UUID, str] = {}
    for org in sink.organisations or []:
        key = f"org:{org.id}"
        org_keys[org.id] = key
        G.add_node(key, label=org.name, kind="org", registry=_registry(org))

    for s in sink.sites or []:
        if s.organisationRef and s.organisationRef in org_keys:
            G.add_edge(org_keys[s.organisationRef], f"site:{s.siteRef}", relation="organisation")

    if sink.ownerOrganisationId and sink.ownerOrganisationId in org_keys:
        G.add_edge(org_keys[sink.ownerOrganisationId], sink_key, relation="owns_sink")

    if sink.carbonCaptureSiteId:
        sk = _ensure_site(G, sink.carbonCaptureSiteId, site_labels, sites_by_ref)
        G.add_edge(sk, sink_key, relation="capture_site")

    mat_keys: dict[UUID, str] = {}
    for m in sink.materials or []:
        key = f"material:{m.id}"
        mat_keys[m.id] = key
        G.add_node(key, label=m.name, kind="material", registry=_registry(m))

    if sink.materialId and sink.materialId in mat_keys:
        G.add_edge(mat_keys[sink.materialId], sink_key, relation="sink_material")

    for c in sink.utilizedContainers or []:
        _add_or_update_container(G, c)
        key = f"container:{c.id}"
        for amount in c.content:
            mid = amount.material.id if amount.material else None
            if mid and mid in mat_keys:
                G.add_edge(mat_keys[mid], key, relation="filled")

    for db in sink.lcaDatabases or []:
        key = f"lca_db:{db.id}"
        G.add_node(
            key,
            label=f"{db.type.value} v{db.version}",
            kind="lca_db",
            registry=_registry(db),
        )

    for act in sink.lcaActivities or []:
        key = f"lca_activity:{act.id}"
        title = act.title
        label = title if len(title) <= 56 else title[:53] + "…"
        G.add_node(key, label=label, kind="lca_activity", registry=_registry(act))
        db_key = f"lca_db:{act.lcaDatabaseId}"
        if db_key in G:
            G.add_edge(db_key, key, relation="activity")

    eg = sink.eventGraph
    if not eg or not eg.nodes:
        return G

    for _eid, info in eg.nodes.items():
        ev = info.event
        if ev is None:
            continue
        ek = f"event:{ev.eventRef}"
        G.add_node(
            ek,
            label=_event_label(ev),
            kind="event",
            registry=_registry(info),
            **_event_node_attrs(info),
        )

    for _eid, info in eg.nodes.items():
        ev = info.event
        if ev is None:
            continue
        ek = f"event:{ev.eventRef}"

        for link in info.links or []:
            pred = f"event:{link.eventRef}"
            if pred not in G:
                G.add_node(
                    pred,
                    label=f"Event {link.eventRef[:8]}…",
                    kind="event",
                    placeholder=True,
                    missing_event_ref=link.eventRef,
                )
            _add_supply_chain_edge(G, pred, ek, link.model_dump(mode="json"))

        if ev.senderSiteId:
            sk = _ensure_site(G, ev.senderSiteId, site_labels, sites_by_ref)
            G.add_edge(sk, ek, relation="sender")
        if ev.receiverSiteId:
            sk = _ensure_site(G, ev.receiverSiteId, site_labels, sites_by_ref)
            G.add_edge(sk, ek, relation="receiver")
        if ev.siteId:
            sk = _ensure_site(G, ev.siteId, site_labels, sites_by_ref)
            G.add_edge(sk, ek, relation="site")

        for containers in (ev.payload, ev.input, ev.output):
            if not containers:
                continue
            for c in containers:
                ck = _add_or_update_container(G, c)
                G.add_edge(ek, ck, relation="container")

        for act in ev.lcaActivities or []:
            ak = f"lca_activity:{act.id}"
            if ak not in G:
                label = act.title if len(act.title) <= 56 else act.title[:53] + "…"
                G.add_node(ak, label=label, kind="lca_activity", registry=_registry(act))
            G.add_edge(ak, ek, relation="lca")

    if eg.root:
        rk = f"event:{eg.root}"
        if rk in G:
            G.add_edge(rk, sink_key, relation="to_sink")

    return G


def graph_summary(G: nx.DiGraph) -> dict[str, int | dict[str, int]]:
    """Count nodes by ``kind`` and return edge count."""
    by_kind: dict[str, int] = {}
    for _n, data in G.nodes(data=True):
        k = str(data.get("kind", "unknown"))
        by_kind[k] = by_kind.get(k, 0) + 1
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "nodes_by_kind": dict(sorted(by_kind.items())),
    }
