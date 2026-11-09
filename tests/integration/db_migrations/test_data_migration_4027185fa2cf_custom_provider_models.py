"""Data migration test for ``models`` on custom providers."""

import json
from datetime import datetime, timezone
from typing import Any, Literal

from alembic.config import Config
from sqlalchemy import Connection, text
from sqlalchemy.ext.asyncio import AsyncEngine

from . import _down, _get_table_schema_info, _run_async, _TableSchemaInfo, _up

_DOWN = "e724eb2ee5cf"
_UP = "4027185fa2cf"

_SDKS = ("openai", "azure_openai", "anthropic", "google_genai", "aws_bedrock", "typesafe")


def _seed(conn: Connection) -> None:
    for name, sdk in [(sdk, sdk) for sdk in _SDKS] + [("openai in use", "openai")]:
        conn.execute(
            text(
                "INSERT INTO generative_model_custom_providers (name, provider, sdk, config) "
                "VALUES (:name, :provider, :sdk, :config)"
            ),
            {"name": name, "provider": sdk, "sdk": sdk, "config": b"encrypted"},
        )
    now = datetime.now(timezone.utc)
    decision_id = conn.execute(
        text(
            "INSERT INTO decisions (name, metadata, created_at, updated_at) "
            "VALUES ('triage', '{}', :now, :now) RETURNING id"
        ),
        {"now": now},
    ).scalar_one()
    conn.execute(
        text(
            "INSERT INTO decision_versions (decision_id, template_format, state, questions, "
            "model_provider, model_name, metadata, created_at, custom_provider_id) "
            'VALUES (:decision_id, \'MUSTACHE\', \'{"type": "json", "template": "{{t}}"}\', '
            "'[]', 'OPENAI', 'luna', '{}', :now, "
            "(SELECT id FROM generative_model_custom_providers WHERE name = 'openai in use'))"
        ),
        {"decision_id": decision_id, "now": now},
    )
    conn.commit()


def _served(conn: Connection) -> dict[str, Any]:
    rows = conn.execute(text("SELECT name, models FROM generative_model_custom_providers")).all()
    return {name: json.loads(value) if isinstance(value, str) else value for name, value in rows}


def _providers(conn: Connection) -> list[tuple[str, str, bytes]]:
    rows = conn.execute(
        text("SELECT name, sdk, config FROM generative_model_custom_providers ORDER BY id")
    ).all()
    return [(name, sdk, bytes(config)) for name, sdk, config in rows]


async def test_custom_provider_provider_models(
    _engine: AsyncEngine,
    _alembic_config: Config,
    _db_backend: Literal["sqlite", "postgresql"],
    _schema: str,
) -> None:
    async def columns() -> _TableSchemaInfo:
        info = await _run_async(
            _engine,
            lambda conn: _get_table_schema_info(
                conn, "generative_model_custom_providers", _db_backend, _schema
            ),
        )
        assert info is not None
        return info

    await _up(_engine, _alembic_config, _DOWN, _schema)
    await _run_async(_engine, _seed)
    before = await _run_async(_engine, _providers)

    await _up(_engine, _alembic_config, _UP, _schema)
    assert await _run_async(_engine, _served) == {
        "openai": {"chat": {}},
        "azure_openai": {"chat": {}},
        "anthropic": {"chat": {}},
        "google_genai": {"chat": {}},
        "aws_bedrock": {"chat": {}},
        "typesafe": {"decision": {}},
        "openai in use": {"chat": {}, "decision": {}},
    }
    info = await columns()
    assert "models" in info["column_names"]
    assert "models" not in info["nullable_column_names"]
    assert await _run_async(_engine, _providers) == before

    await _down(_engine, _alembic_config, _DOWN, _schema)
    assert "models" not in (await columns())["column_names"]
    assert await _run_async(_engine, _providers) == before
