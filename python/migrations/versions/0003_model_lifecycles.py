"""Independent model ownership, deletion operations and immutable cost accounting."""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    from qps.hosted.database import metadata

    for name in (
        "models",
        "artifact_refs",
        "artifact_provenance",
        "operations",
        "tombstones",
        "cost_ledger",
        "tracking_reports",
    ):
        metadata.tables[name].create(op.get_bind(), checkfirst=True)
    op.add_column("deployments", sa.Column("model_version_id", sa.String()))
    op.add_column(
        "allocations", sa.Column("gpu_count", sa.Integer(), nullable=False, server_default="1")
    )
    op.add_column("allocations", sa.Column("worker_group", sa.String()))
    op.execute("UPDATE allocations SET gpu_count=0 WHERE profile='cpu-fixture'")
    op.execute(
        "INSERT INTO cost_ledger (id,usd,gpu_allocations,gpu_seconds,created) SELECT id,spent_usd,CASE WHEN profile='cpu-fixture' THEN 0 ELSE 1 END,COALESCE(ended-created,0)*gpu_count,COALESCE(ended,created) FROM allocations WHERE state='terminated'"
    )


def downgrade():
    raise RuntimeError("Destructive lifecycle migrations require an explicit restore procedure")
