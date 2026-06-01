# HackHPI 2026 – Salty Peanut Grapes

Team repository for [HackHPI 2026](https://hackhpi.org/). It includes a small Python library that talks to the **Cula** public API using **httpx** and **Pydantic v2** models generated from the OpenAPI document in this repo.

The live API is hosted at `https://api.hack-hpi.cula.earth` (paths still use the `/api/hack-hpi/...` prefix). All endpoints are read-only and unauthenticated. Rate limit: **60 requests per minute** per IP (HTTP 429 when exceeded).

## Requirements 
 
- Python 3.10+

## Install
 
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

For regenerating Pydantic models from the OpenAPI file:

```bash
pip install -e ".[dev]"
```

For the sink graph visualization example:

```bash
pip install -e ".[viz]"
```

## Usage

```python
from datetime import datetime, timezone
from uuid import UUID

from cula import CulaClient
from cula.models import MachineDpRequest

with CulaClient() as client:
    sink_ids = client.list_sinks()
    sink = client.get_sink(sink_ids[0])

    # Optional: time series (needs valid machine / data-point IDs from the sink)
    # machines = client.list_machines(sink.carbonCaptureSiteId)
    # dp_ids = client.list_machine_data_points(machines[0])
    # series = client.get_machine_data([
    #     MachineDpRequest(
    #         source=dp_ids[0],
    #         start=datetime(2024, 1, 1, tzinfo=timezone.utc),
    #         end=datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc),
    #         timeBucket="1 hour",
    #     )
    # ])
```

`CulaClient` methods:

| Method | Description |
|--------|-------------|
| `list_sinks()` | UUIDs of all sinks |
| `get_sink(id)` | Full `Sink` model (graph, sites, materials, …) |
| `download_document(id)` | Raw bytes for a proof file (`cloudStorageId`, etc.) |
| `list_machines(site_id)` | Machine UUIDs for a carbon capture site |
| `list_machine_data_points(machine_id)` | Data-point config UUIDs |
| `get_machine_data_point(config_id)` | Metadata for one data point |
| `get_machine_data(requests)` | POST body: list of `MachineDpRequest` |

Types live in `cula.models` (e.g. `Sink`, `MachineDpRequest`, `MachineDataInRange`).

## Examples and codegen

- **`example/fetch_sink.py`** — Fetches one sink and prints JSON to stdout. With no arguments, uses the first id from `list_sinks()`; optionally pass a sink UUID.

  ```bash
  python example/fetch_sink.py
  python example/fetch_sink.py 10375aa3-b4b0-4543-900c-d83f163babd9
  ```

- **`example/visualize_sink_graph.py`** — Loads a sink and writes a PNG of a **flow layout**: the sink is on the **far right**, and other entities spread **left** in layers by graph distance (with light barycentric ordering to reduce crossings). Node labels use display names (organisations, sites, materials, containers, lifecycle events, LCA entities). Uses NetworkX and Matplotlib (`pip install -e ".[viz]"`).

  ```bash
  python example/visualize_sink_graph.py -o sink_graph.png
  python example/visualize_sink_graph.py 10375aa3-b4b0-4543-900c-d83f163babd9 -o my_sink.png
  ```

- **`example/build_sink_graph.py`** — Builds the same NetworkX graph as `cula.sink_graph.build_entity_graph`, prints node/edge counts by kind, and optionally writes **JSON** (node-link), **GraphML**, or **GEXF** (`-o` / `--format`). Core dependency is NetworkX; no matplotlib required.

  ```bash
  python example/build_sink_graph.py
  python example/build_sink_graph.py -o sink.json
  python example/build_sink_graph.py --format graphml -o sink.graphml
  ```

- **`regenerate_models.sh`** (repository root) — Regenerates `cula/models.py` from `openapi/cula.openapi.json` (requires `pip install -e ".[dev]"` and a working `.venv`). Reapplies a small post-process fix for a datamodel-codegen quirk on `EventInfo.event`.

## OpenAPI

The spec is vendored at **`openapi/cula.openapi.json`**. It was extended with a `manual` value on `sourceConfig.valueSourceType` so responses match production data; if the upstream spec changes, merge carefully and re-run `./regenerate_models.sh`.

## Layout

| Path | Purpose |
|------|---------|
| `cula/` | Package: `CulaClient`, `models.py` |
| `openapi/cula.openapi.json` | OpenAPI 3.0 source |
| `example/` | Example scripts |
| `regenerate_models.sh` | Regenerate `cula/models.py` from the OpenAPI file |
