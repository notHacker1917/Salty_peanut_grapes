from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

import httpx
from pydantic import TypeAdapter

from cula.models import MachineDataInRange, MachineDpConfig, MachineDpRequest, Sink

DEFAULT_BASE_URL = "https://api.hack-hpi.cula.earth"

_uuid_list_adapter: TypeAdapter[list[UUID]] = TypeAdapter(list[UUID])
_machine_data_list_adapter: TypeAdapter[list[MachineDataInRange]] = TypeAdapter(
    list[MachineDataInRange]
)


class CulaClient:
    """Sync HTTP client for the Cula API. Models are Pydantic v2 (`cula.models`)."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=self._base_url, timeout=timeout)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> CulaClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def list_sinks(self) -> list[UUID]:
        response = self._client.get("/api/hack-hpi/sinks")
        response.raise_for_status()
        return _uuid_list_adapter.validate_json(response.content)

    def get_sink(self, sink_id: UUID | str) -> Sink:
        response = self._client.get(f"/api/hack-hpi/sinks/{sink_id}")
        response.raise_for_status()
        return Sink.model_validate_json(response.content)

    def download_document(self, cloud_storage_id: UUID | str) -> bytes:
        response = self._client.get(f"/api/hack-hpi/documents/{cloud_storage_id}")
        response.raise_for_status()
        return response.content

    def list_machines(self, site_id: UUID | str) -> list[UUID]:
        response = self._client.get(f"/api/hack-hpi/sites/{site_id}/machines")
        response.raise_for_status()
        return _uuid_list_adapter.validate_json(response.content)

    def list_machine_data_points(self, machine_id: UUID | str) -> list[UUID]:
        response = self._client.get(f"/api/hack-hpi/machines/{machine_id}/data-points")
        response.raise_for_status()
        return _uuid_list_adapter.validate_json(response.content)

    def get_machine_data_point(self, machine_dp_config_id: UUID | str) -> MachineDpConfig:
        response = self._client.get(
            f"/api/hack-hpi/machine-data-points/{machine_dp_config_id}"
        )
        response.raise_for_status()
        return MachineDpConfig.model_validate_json(response.content)

    def get_machine_data(
        self, body: Sequence[MachineDpRequest]
    ) -> list[MachineDataInRange]:
        payload = [item.model_dump(mode="json") for item in body]
        response = self._client.post("/api/hack-hpi/machine-data", json=payload)
        response.raise_for_status()
        return _machine_data_list_adapter.validate_json(response.content)
