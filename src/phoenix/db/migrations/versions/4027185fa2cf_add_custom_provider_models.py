"""add models to custom providers

Revision ID: 4027185fa2cf
Revises: e724eb2ee5cf
Create Date: 2026-11-06 12:00:00.000000

"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import JSON
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.compiler import compiles

revision: str = "4027185fa2cf"
down_revision: Union[str, None] = "e724eb2ee5cf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


class JSONB(JSON):
    __visit_name__ = "JSONB"


@compiles(JSONB, "sqlite")
def _(*args: Any, **kwargs: Any) -> str:
    return "JSONB"


JSON_ = JSON().with_variant(postgresql.JSONB(), "postgresql").with_variant(JSONB(), "sqlite")

_providers = sa.table(
    "generative_model_custom_providers",
    sa.column("id", sa.Integer),
    sa.column("sdk", sa.String),
    sa.column("models", JSON_),
)
_decision_versions = sa.table("decision_versions", sa.column("custom_provider_id", sa.Integer))
_decision_tasks = sa.table("experiment_decision_tasks", sa.column("custom_provider_id", sa.Integer))


def upgrade() -> None:
    with op.batch_alter_table("generative_model_custom_providers") as batch_op:
        batch_op.add_column(sa.Column("models", JSON_, nullable=True))
    op.execute(
        _providers.update().where(_providers.c.sdk == "typesafe").values(models={"decision": {}})
    )
    op.execute(
        _providers.update().where(_providers.c.sdk != "typesafe").values(models={"chat": {}})
    )
    in_use = sa.union(
        sa.select(_decision_versions.c.custom_provider_id),
        sa.select(_decision_tasks.c.custom_provider_id),
    )
    op.execute(
        _providers.update()
        .where(_providers.c.sdk == "openai")
        .where(_providers.c.id.in_(in_use))
        .values(models={"chat": {}, "decision": {}})
    )
    with op.batch_alter_table("generative_model_custom_providers") as batch_op:
        batch_op.alter_column("models", existing_type=JSON_, nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("generative_model_custom_providers") as batch_op:
        batch_op.drop_column("models")
