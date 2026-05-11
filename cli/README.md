# Cula OpenAPI CLI

This CLI is generated dynamically from the OpenAPI spec.
You can override the used spec path with `--spec`.

## Install and run

```bash
pip install -e .
cula --help
```

## Common commands

```bash
# GET /api/hack-hpi/sinks
cula list-sinks

# GET /api/hack-hpi/sinks/{id}
cula get-sink --path-id 5c13033d-fb32-4515-b246-1c9ab6a615eb

# GET /api/hack-hpi/sites/{siteId}/machines
cula list-machines --path-site-id f132b7a0-6989-4242-9df4-1928e06541a9

# POST /api/hack-hpi/machine-data with a body file
cula get-machine-data --body-file requests.json
```

## Inspect sinks workflow

Use this sequence when you want to investigate one sink end-to-end:

```bash
# 1) List available sink IDs
cula list-sinks --output compact

# 2) Inspect one sink payload
cula get-sink --path-id <sink-id> > sink.json

# 3) Re-use sink context:
#    - carbonCaptureSiteId for machine discovery
#    - eventGraph.nodes for lifecycle events
#    - proofs[].fileReference.cloudStorageId for downloadable evidence
```

Machine telemetry drill-down (using IDs from the sink payload):

```bash
# 4) Find machines for the sink's carbon capture site
cula list-machines --path-site-id <carbon-capture-site-id>

# 5) Find data points for one machine
cula list-machine-data-points --path-machine-id <machine-id>

# 6) Inspect a data point config
cula get-machine-data-point --path-machine-dp-config-id <data-point-id>
```

If a sink contains proof file references, you can download them:

```bash
cula download-document --path-id <cloud-storage-id> --download-to proof.bin
```

For direct script execution without installation:

```bash
python cli/cli.py --help
```

## Body input for POST/PUT/PATCH

Pass JSON either as:

- `--body-file path/to/body.json`
- `--body-json '[{"source":"...","start":"...","end":"...","timeBucket":"1 hour"}]'`

## Response output

- JSON responses print prettified by default.
- Use `--output compact` for single-line JSON or raw bytes to stdout.
- Use `--download-to <path>` to write binary content (for `/documents/{id}`).

