from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import create_engine, select

from cannery.database import Base
from cannery.jobs import ProbeJob, complete_probe, create_probe, fail_probe, pending_probes


def test_duplicate_deliveries_and_terminal_state(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    Base.metadata.create_all(engine)
    job_id = create_probe(engine)
    assert pending_probes(engine) == [str(job_id)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: complete_probe(engine, str(job_id)), range(4)))
    fail_probe(engine, str(job_id))
    with engine.connect() as connection:
        row = connection.execute(select(ProbeJob)).one()
        assert row.status == "complete"
        assert row.effects == 1
    failed_id = create_probe(engine)
    fail_probe(engine, str(failed_id))
    complete_probe(engine, str(failed_id))
    assert pending_probes(engine) == []
    with engine.connect() as connection:
        assert (
            connection.execute(select(ProbeJob.status).where(ProbeJob.id == failed_id)).scalar_one()
            == "failed"
        )
    engine.dispose()
