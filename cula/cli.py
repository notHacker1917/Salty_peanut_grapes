from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx


@dataclass(frozen=True)
class OperationSpec:
    name: str
    method: str
    path: str
    summary: str
    description: str
    parameters: list[dict[str, Any]]
    request_body_required: bool


def _to_cli_name(operation_id: str) -> str:
    chars: list[str] = []
    for idx, char in enumerate(operation_id):
        if char.isupper() and idx > 0:
            chars.append("-")
        chars.append(char.lower())
    return "".join(chars)


def _normalize_arg_name(location: str, param_name: str) -> str:
    normalized: list[str] = []
    previous_is_lower = False
    for char in param_name:
        if char in {"-", " "}:
            normalized.append("_")
            previous_is_lower = False
            continue
        if char.isupper() and previous_is_lower:
            normalized.append("_")
        normalized.append(char.lower())
        previous_is_lower = char.islower()
    compact = "".join(normalized).strip("_")
    return f"{location}_{compact}"


def _load_openapi(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("OpenAPI file must contain a JSON object at the root.")
    if "paths" not in data or not isinstance(data["paths"], dict):
        raise ValueError("OpenAPI file is missing a valid 'paths' object.")
    return data


def _collect_operations(spec: dict[str, Any]) -> list[OperationSpec]:
    operations: list[OperationSpec] = []
    for path, path_item in spec["paths"].items():
        if not isinstance(path_item, dict):
            continue
        common_parameters = path_item.get("parameters", [])
        if not isinstance(common_parameters, list):
            common_parameters = []

        for method in ("get", "post", "put", "patch", "delete"):
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue

            operation_id = operation.get("operationId")
            if not operation_id or not isinstance(operation_id, str):
                continue

            op_parameters = operation.get("parameters", [])
            if not isinstance(op_parameters, list):
                op_parameters = []

            request_body = operation.get("requestBody", {})
            request_body_required = False
            if isinstance(request_body, dict):
                request_body_required = bool(request_body.get("required", False))

            operations.append(
                OperationSpec(
                    name=_to_cli_name(operation_id),
                    method=method.upper(),
                    path=path,
                    summary=str(operation.get("summary", "")).strip(),
                    description=str(operation.get("description", "")).strip(),
                    parameters=[*common_parameters, *op_parameters],
                    request_body_required=request_body_required,
                )
            )
    return operations


def _get_default_base_url(spec: dict[str, Any]) -> str:
    servers = spec.get("servers", [])
    if isinstance(servers, list):
        for server in servers:
            if isinstance(server, dict):
                url = server.get("url")
                if isinstance(url, str) and url:
                    return url.rstrip("/")
    return "https://api.hack-hpi.cula.earth"


def _coerce_parameter_value(raw_value: str, schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "integer":
        return int(raw_value)
    if schema_type == "number":
        return float(raw_value)
    if schema_type == "boolean":
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value '{raw_value}'.")
    return raw_value


def _prepare_body(args: argparse.Namespace) -> Any | None:
    if args.body_file:
        body_path = Path(args.body_file)
        with body_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if args.body_json:
        return json.loads(args.body_json)
    return None


def _print_response(response: httpx.Response, output_mode: str, download_to: str | None) -> None:
    content_type = response.headers.get("content-type", "").lower()
    is_json = "application/json" in content_type

    if download_to:
        out_path = Path(download_to)
        out_path.write_bytes(response.content)
        print(str(out_path))
        return

    if is_json:
        payload = response.json()
        if output_mode == "compact":
            print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if output_mode == "compact":
        sys.stdout.buffer.write(response.content)
        return
    print(response.text)


def _build_parser(spec_path_default: Path, operations: list[OperationSpec]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cula",
        description="Cula API CLI generated from OpenAPI operation IDs.",
    )
    parser.add_argument(
        "--spec",
        default=str(spec_path_default),
        help="Path to OpenAPI JSON file used to derive operations.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override API base URL from OpenAPI servers[0].",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--output",
        choices=("pretty", "compact"),
        default="pretty",
        help="Output mode for responses.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    for operation in operations:
        command_help = operation.summary or operation.description or f"{operation.method} {operation.path}"
        op_parser = subparsers.add_parser(
            operation.name,
            help=command_help,
            description=f"{operation.method} {operation.path}\n\n{operation.description}".strip(),
        )

        seen_names: set[str] = set()
        for parameter in operation.parameters:
            if not isinstance(parameter, dict):
                continue
            param_name = parameter.get("name")
            param_in = parameter.get("in")
            if not isinstance(param_name, str) or not isinstance(param_in, str):
                continue
            arg_name = _normalize_arg_name(param_in, param_name)
            if arg_name in seen_names:
                continue
            seen_names.add(arg_name)

            required = bool(parameter.get("required", False))
            description = str(parameter.get("description", "")).strip()

            option_flag = f"--{arg_name.replace('_', '-')}"
            op_parser.add_argument(
                option_flag,
                dest=arg_name,
                required=required,
                help=f"{param_in} parameter '{param_name}'. {description}".strip(),
            )

        op_parser.add_argument(
            "--download-to",
            help="Optional file path to write response bytes.",
        )

        if operation.method in {"POST", "PUT", "PATCH"}:
            op_parser.add_argument(
                "--body-file",
                help="Path to JSON body file.",
            )
            op_parser.add_argument(
                "--body-json",
                help="Inline JSON body string (alternative to --body-file).",
            )

        op_parser.set_defaults(_operation=operation)
    return parser


def _execute(parsed: argparse.Namespace, base_url: str) -> int:
    operation: OperationSpec = parsed._operation
    path = operation.path
    query_params: dict[str, Any] = {}
    headers: dict[str, Any] = {}

    for parameter in operation.parameters:
        if not isinstance(parameter, dict):
            continue
        name = parameter.get("name")
        location = parameter.get("in")
        if not isinstance(name, str) or not isinstance(location, str):
            continue
        arg_key = _normalize_arg_name(location, name)
        raw_value = getattr(parsed, arg_key, None)
        if raw_value is None:
            continue

        schema = parameter.get("schema", {})
        if not isinstance(schema, dict):
            schema = {}
        value = _coerce_parameter_value(str(raw_value), schema)

        if location == "path":
            path = path.replace("{" + name + "}", str(value))
        elif location == "query":
            query_params[name] = value
        elif location == "header":
            headers[name] = str(value)

    body = None
    if operation.method in {"POST", "PUT", "PATCH"}:
        if operation.request_body_required and not parsed.body_file and not parsed.body_json:
            print("This operation requires --body-file or --body-json.", file=sys.stderr)
            return 2
        try:
            body = _prepare_body(parsed)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to read body: {exc}", file=sys.stderr)
            return 2

    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    try:
        with httpx.Client(timeout=parsed.timeout) as client:
            response = client.request(
                method=operation.method,
                url=url,
                params=query_params or None,
                headers=headers or None,
                json=body,
            )
            response.raise_for_status()
            _print_response(response, parsed.output, parsed.download_to)
            return 0
    except httpx.HTTPStatusError as exc:
        print(
            f"Request failed with HTTP {exc.response.status_code} for {operation.method} {path}",
            file=sys.stderr,
        )
        if exc.response.content:
            print(exc.response.text, file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Invalid argument value: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--spec", default="/Users/mikahoppe/Downloads/api-1.json")
    pre_args, _ = pre_parser.parse_known_args(argv)

    spec_path = Path(pre_args.spec)
    try:
        spec = _load_openapi(spec_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load OpenAPI spec at {spec_path}: {exc}", file=sys.stderr)
        return 2

    operations = _collect_operations(spec)
    if not operations:
        print("No operations with operationId found in spec.", file=sys.stderr)
        return 2

    parser = _build_parser(spec_path, operations)
    parsed = parser.parse_args(argv)
    base_url = parsed.base_url or _get_default_base_url(spec)
    return _execute(parsed, base_url=base_url)

