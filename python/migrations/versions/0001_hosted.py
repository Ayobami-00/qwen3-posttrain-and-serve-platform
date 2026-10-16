"""Frozen initial PostgreSQL schema."""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

SQL = [
    "\nCREATE TABLE allocations (\n\tid VARCHAR NOT NULL, \n\towner_id VARCHAR, \n\tkind VARCHAR, \n\tattempt INTEGER, \n\tgeneration INTEGER, \n\tprofile VARCHAR, \n\tstate VARCHAR, \n\tdesired VARCHAR, \n\tpod_id VARCHAR, \n\tgpu_type VARCHAR, \n\thourly_rate FLOAT, \n\treserved_usd FLOAT, \n\tspent_usd FLOAT, \n\tcreated FLOAT, \n\tended FLOAT, \n\tdeadline FLOAT, \n\tlease_until FLOAT, \n\theartbeat FLOAT, \n\tphase VARCHAR, \n\ttelemetry JSON, \n\tresult JSON, \n\tendpoint VARCHAR, \n\tready BOOLEAN, \n\tmanifest_sha VARCHAR, \n\tPRIMARY KEY (id), \n\tUNIQUE (owner_id, generation), \n\tUNIQUE (pod_id)\n)\n\n",
    "\nCREATE TABLE artifacts (\n\tid VARCHAR NOT NULL, \n\trun_id VARCHAR, \n\tattempt INTEGER, \n\tkind VARCHAR, \n\tmanifest_sha VARCHAR, \n\tmanifest JSON, \n\tobjects JSON, \n\tsize FLOAT, \n\tcreated FLOAT, \n\tPRIMARY KEY (id)\n)\n\n",
    "CREATE INDEX ix_artifacts_run_id ON artifacts (run_id)",
    "\nCREATE TABLE attempts (\n\trun_id VARCHAR NOT NULL, \n\tnumber INTEGER NOT NULL, \n\tgeneration INTEGER, \n\tstatus VARCHAR, \n\toutputs JSON, \n\tcheckpoint VARCHAR, \n\tsequence INTEGER, \n\tcreated FLOAT, \n\treason VARCHAR, \n\tPRIMARY KEY (run_id, number)\n)\n\n",
    "\nCREATE TABLE budgets (\n\tid VARCHAR NOT NULL, \n\ttotal FLOAT, \n\tslots INTEGER, \n\tPRIMARY KEY (id)\n)\n\n",
    "\nCREATE TABLE datasets (\n\tid VARCHAR NOT NULL, \n\tidentity JSON, \n\tprepared_key VARCHAR, \n\tartifact_id VARCHAR, \n\tcreated FLOAT, \n\tPRIMARY KEY (id), \n\tUNIQUE (prepared_key)\n)\n\n",
    "\nCREATE TABLE deployments (\n\tid VARCHAR NOT NULL, \n\tname VARCHAR, \n\tidempotency_key VARCHAR, \n\trequest_sha VARCHAR, \n\tspec JSON, \n\trun_id VARCHAR, \n\tstatus VARCHAR, \n\tdesired VARCHAR, \n\tgeneration INTEGER, \n\tcreated FLOAT, \n\tupdated FLOAT, \n\tPRIMARY KEY (id), \n\tUNIQUE (name), \n\tUNIQUE (idempotency_key)\n)\n\n",
    "\nCREATE TABLE events (\n\tid SERIAL NOT NULL, \n\trun_id VARCHAR, \n\tattempt INTEGER, \n\tgeneration INTEGER, \n\tsequence INTEGER, \n\ttime VARCHAR, \n\tkind VARCHAR, \n\tpayload JSON, \n\tPRIMARY KEY (id), \n\tUNIQUE (run_id, attempt, sequence)\n)\n\n",
    "CREATE INDEX ix_events_run_id ON events (run_id)",
    "\nCREATE TABLE outbox (\n\tid SERIAL NOT NULL, \n\tresource VARCHAR, \n\tresource_id VARCHAR, \n\tgeneration INTEGER, \n\taction VARCHAR, \n\tdelivered BOOLEAN, \n\tlocked_until FLOAT, \n\tcreated FLOAT, \n\tPRIMARY KEY (id)\n)\n\n",
    "\nCREATE TABLE runs (\n\tid VARCHAR NOT NULL, \n\tname VARCHAR, \n\tidempotency_key VARCHAR, \n\trequest_sha VARCHAR, \n\tsubmission JSON, \n\tresolved JSON, \n\tstatus VARCHAR, \n\tphase VARCHAR, \n\tattempt INTEGER, \n\tdesired VARCHAR, \n\tcreated FLOAT, \n\tupdated FLOAT, \n\tprojection_time FLOAT, \n\tPRIMARY KEY (id), \n\tUNIQUE (name), \n\tUNIQUE (idempotency_key)\n)\n\n",
    "\nCREATE TABLE sessions (\n\tid VARCHAR NOT NULL, \n\tencrypted VARCHAR, \n\tcsrf VARCHAR, \n\texpires FLOAT, \n\tPRIMARY KEY (id)\n)\n\n",
    "\nCREATE TABLE tracking (\n\trun_id VARCHAR NOT NULL, \n\tattempt INTEGER NOT NULL, \n\tcursor INTEGER, \n\turl VARCHAR, \n\tstate VARCHAR, \n\tPRIMARY KEY (run_id, attempt)\n)\n\n",
    "\nCREATE TABLE uploads (\n\tid VARCHAR NOT NULL, \n\tallocation_id VARCHAR, \n\tmanifest JSON, \n\tmanifest_sha VARCHAR, \n\tstate VARCHAR, \n\tcreated FLOAT, \n\texpires FLOAT, \n\tPRIMARY KEY (id)\n)\n\n",
]


def upgrade():
    for statement in SQL:
        op.execute(statement)


def downgrade():
    op.execute("DROP TABLE uploads")
    op.execute("DROP TABLE tracking")
    op.execute("DROP TABLE sessions")
    op.execute("DROP TABLE runs")
    op.execute("DROP TABLE outbox")
    op.execute("DROP TABLE events")
    op.execute("DROP TABLE deployments")
    op.execute("DROP TABLE datasets")
    op.execute("DROP TABLE budgets")
    op.execute("DROP TABLE attempts")
    op.execute("DROP TABLE artifacts")
    op.execute("DROP TABLE allocations")
