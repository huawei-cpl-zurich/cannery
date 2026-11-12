import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from test_build_state import candidate
from test_submission_requests import records as records
from typer.testing import CliRunner

from cannery.artifacts import ArtifactRef
from cannery.benchmark.state import claim_job, ensure_job, finish_job
from cannery.builds.state import claim_build, finish_build
from cannery.cli import app
from cannery.numerics.state import NumericalState, claim_numerical, finish_numerical
from cannery.submissions.api import get_info, list_info
from cannery.web.submissions import submission_pending


@pytest.fixture
def report(records):
    engine, _ = records
    identifier = candidate(records)
    artifact = ArtifactRef(
        repository="private", revision="a" * 64, path="raw.json", sha256="a" * 64
    )
    finish_build(engine, claim_build(engine, identifier), artifact=artifact)
    with Session(engine) as db, db.begin():
        db.add(NumericalState(id=identifier))
    finish_numerical(
        engine, claim_numerical(engine, identifier), report={"passed": True, "cases": []}
    )
    ensure_job(engine, "b" * 64, "reference", {})
    ensure_job(
        engine,
        "c" * 64,
        "candidate",
        {"cohort": {"device": "a5"}},
        candidate_id=identifier,
        reference_key="b" * 64,
    )
    summary = {
        "passed": True,
        "protocol": "sync-call-v1",
        "geomean_speedup": 2.0,
        "cases": [
            {
                "id": "small",
                "candidate": {"median_us": 10.0, "p95_us": 12.0, "samples": 100},
                "reference": {"median_us": 20.0, "p95_us": 22.0, "samples": 100},
                "speedup": 2.0,
            },
            {"id": "empty", "excluded": "empty_correctness_only"},
        ],
        "evidence": {"worker": {"target": "private-host"}},
    }
    return engine, identifier, artifact, summary


@pytest.mark.parametrize(
    "state,public",
    [
        ("queued", "benchmark_queued"),
        ("running", "benchmarking"),
        ("ready", "complete"),
        ("failed", "failed"),
        ("blocked", "awaiting_benchmark"),
    ],
)
def test_benchmark_status_and_private_report_boundary(report, state, public):
    engine, identifier, artifact, summary = report
    if state != "queued":
        claim = claim_job(engine, "c" * 64)
        if state in ("ready", "failed", "blocked"):
            finish_job(
                engine,
                claim,
                report=summary,
                artifact=artifact,
                error=None if state == "ready" else "benchmark_unavailable",
                blocked=state == "blocked",
            )
    with Session(engine) as db:
        info = get_info(db, "team", identifier)
        assert info["status"] == public
        assert info["stages"]["correctness"] == "passed" and info["stages"]["build"] == "complete"
        assert "private-host" not in json.dumps(info)
        assert list_info(db, "team") == [info] and list_info(db, "other") == []
        assert (info["benchmark"]["report"] is not None) == (state == "ready")
        assert submission_pending(info, True) == (state in ("queued", "running"))


def test_web_and_cli_render_same_complete_private_report(report, monkeypatch):
    engine, identifier, artifact, summary = report
    finish_job(engine, claim_job(engine, "c" * 64), report=summary, artifact=artifact)
    with Session(engine) as db:
        info = get_info(db, "team", identifier)
    templates = Jinja2Templates(directory=str(Path("src/cannery/web/templates")))
    page = templates.env.get_template("submissions.html").render(
        job=info,
        pages={},
        pending=False,
        title="My Submissions",
        request=SimpleNamespace(url=SimpleNamespace(path="/submissions/" + str(identifier))),
    )
    assert "10.0" in page and "12.0" in page and "100" in page and "20.0" in page
    assert "correctness only" in page and "Geometric mean speedup" in page
    assert 'hx-trigger="every 2s"' not in page and "Results are private" in page
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *a, **k: httpx.Response(200, json=info, request=httpx.Request("GET", "http://test")),
    )
    result = CliRunner().invoke(app, ["submission", str(identifier)])
    assert (
        result.exit_code == 0
        and json.loads(result.stdout)["benchmark"]["report"]["geomean_speedup"] == 2.0
    )


def test_poll_through_discovery_only_when_benchmark_service_enabled():
    info = {"status": "awaiting_benchmark", "numerical": {"status": "passed"}, "benchmark": None}
    assert not submission_pending(info, False) and submission_pending(info, True)
    info["benchmark"] = {"status": "blocked"}
    assert not submission_pending(info, True)
