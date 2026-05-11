#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate
datamodel-codegen \
  --input openapi/cula.openapi.json \
  --input-file-type openapi \
  --output cula/models.py \
  --output-model-type pydantic_v2.BaseModel \
  --use-standard-collections \
  --use-schema-description \
  --field-constraints \
  --use-double-quotes \
  --target-python-version 3.10
python3 <<'PY'
from pathlib import Path

path = Path("cula/models.py")
text = path.read_text()
old = """    event: Event | None = Field(
        None,
        description=\"The full event payload for this graph node.\",
        discriminator=\"type\",
    )"""
new = """    event: Event | None = Field(
        None,
        description=\"The full event payload for this graph node.\",
    )"""
if old not in text:
    raise SystemExit(
        "regenerate_models.sh: expected EventInfo.event discriminator block not found; "
        "adjust the script if datamodel-codegen output changed."
    )
path.write_text(text.replace(old, new, 1))
PY
