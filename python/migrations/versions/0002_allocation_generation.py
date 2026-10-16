"""Fence duplicate allocations even across independent API replicas."""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_unique_constraint(
        "uq_allocation_generation", "allocations", ["owner_id", "generation"]
    )


def downgrade():
    op.drop_constraint("uq_allocation_generation", "allocations", type_="unique")
