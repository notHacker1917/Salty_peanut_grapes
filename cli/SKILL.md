---
name: cula-cli
description: Use when working with the Cula terminal CLI (`cula`) for listing sinks, inspecting sink details, querying machine metadata, fetching time-series data, or downloading proof documents.
---

# Cula CLI Skill

## Overview

`cula` is the packaged terminal CLI for the Cula Hack HPI API. It is generated from an OpenAPI spec and exposes one subcommand per OpenAPI `operationId`.

Primary use cases:
- Explore sink IDs and sink registry payloads
- Traverse site -> machine -> data-point relationships
- Query machine time-series data
- Download binary proof documents

## Setup

Run from the repository root:

```bash
pip install -e .
cula --help
```

The CLI reads OpenAPI from:
- default: `/Users/mikahoppe/Downloads/api-1.json`
- override with `--spec <path>`

## Command Model

Each operation appears as kebab-case:
- `listSinks` -> `cula list-sinks`
- `getSink` -> `cula get-sink`
- `getMachineData` -> `cula get-machine-data`

Argument naming for OpenAPI parameters:
- format: `--<in>-<param-name>`
- path params become `--path-...`
- query params become `--query-...`
- header params become `--header-...`

Examples:
- OpenAPI `{id}` in path -> `--path-id`
- OpenAPI `{siteId}` in path -> `--path-site-id`

## Core Examples

```bash
# List sink IDs
cula list-sinks

# Fetch one sink
cula get-sink --path-id 5c13033d-fb32-4515-b246-1c9ab6a615eb

# List machines by site
cula list-machines --path-site-id f132b7a0-6989-4242-9df4-1928e06541a9

# List data points by machine
cula list-machine-data-points --path-machine-id a05c1f12-c135-4fbd-9e99-06bff6773d2e
```

## How to Inspect Sinks

Use this investigation flow for most sink-debug or analysis tasks:

1. `cula list-sinks` to collect candidate sink IDs.
2. `cula get-sink --path-id <sink-id>` to fetch the complete registry object.
3. Inspect key sink fields:
   - `carbonCaptureSiteId` (entry point to machines)
   - `eventGraph.root` and `eventGraph.nodes` (lifecycle graph)
   - `materials`, `sites`, `organisations` (supply-chain context)
4. From `eventGraph.nodes.*.event.proofs`, extract file references:
   - `cloudStorageId`
   - `blurredCloudStorageId`
   - `thumbnailCloudStorageId`
5. Download evidence with `cula download-document --path-id <cloud-storage-id> --download-to <file>`.

Machine telemetry flow (starting from sink context):

1. `cula list-machines --path-site-id <carbon-capture-site-id>`
2. `cula list-machine-data-points --path-machine-id <machine-id>`
3. `cula get-machine-data-point --path-machine-dp-config-id <dp-config-id>`
4. `cula get-machine-data --body-file requests.json`

Minimal `requests.json` shape:

```json
[
  {
    "source": "<dp-config-id>",
    "start": "2024-01-01T00:00:00.000Z",
    "end": "2024-01-31T23:59:59.000Z",
    "timeBucket": "1 hour"
  }
]
```

## Request Body Operations

For POST/PUT/PATCH commands, pass body as:
- `--body-file <json-file>`
- or `--body-json '<json-string>'`

Example:

```bash
cula get-machine-data --body-file requests.json
```

`requests.json` example:

```json
[
  {
    "source": "00d41556-5c39-4459-b66d-ba64b64bad3e",
    "start": "2024-01-01T00:00:00.000Z",
    "end": "2024-01-31T23:59:59.000Z",
    "timeBucket": "1 hour"
  }
]
```

## Output and Files

- Default output: pretty JSON (`--output pretty`)
- Compact output: `--output compact`
- Save binary responses (documents): `--download-to <path>`

Example:

```bash
cula download-document --path-id 02782e80-23a2-418c-9366-5cb79e362a4d --download-to proof.bin
```

## Operational Guidance

- API is read-only and unauthenticated.
- Respect rate limit: 60 requests/minute/IP (HTTP 429 if exceeded).
- Prefer one deliberate call at a time when exploring sink graphs.
- For reproducible automation, pin `--spec` explicitly in scripts.
