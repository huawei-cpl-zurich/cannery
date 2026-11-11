from test_baseline_evidence import arguments
from test_freeze import engine as engine
from test_freeze import store as store

from cannery.registry.api import spec_info
from cannery.registry.freeze import freeze_spec


def test_cached_baseline_is_identified_without_exposing_outputs(engine, store):
    args = arguments(engine, store)
    frozen = freeze_spec(engine, *args, store=store, namespace="team", allow_new_version=False)
    value = spec_info(frozen)
    assert value["smoke"]["assurance"] == "cached_cpu_reference"
    assert value["baseline"] == {
        "key": args[2].baseline_key,
        "suite_sha256": args[2].suite_sha256,
        "generator": "sha256-f32-v1",
        "output_sha256": args[2].artifact.sha256,
        "cases": 3,
    }
    assert "outputs" not in value["baseline"] and "artifact" not in value["baseline"]
