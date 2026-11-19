import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from release_fixtures import admission as admission
from release_fixtures import arguments as arguments
from release_fixtures import benchmark as benchmark
from release_fixtures import records as records
from release_fixtures import setup_executor as setup_executor
from test_release_catalog import releases as releases
from test_release_resolution import published as published

from cannery.releases import state
from cannery.releases.worker import process
from cannery.settings import Settings
from cannery.web import install_web
from cannery.web.releases import chart


def web(engine):
    app = FastAPI()
    install_web(app, Settings(release_enabled=True), engine)
    return TestClient(app)


def test_global_best_empty_partition_chart_and_withdrawal(releases):
    engine, add, _, m = releases
    client = web(engine)
    assert "No published results yet" in client.get("/global-best").text
    first, second = add(0.5), add(0.7, "catlass")
    params = {"kernel": m.kernel, "spec_version": "", "cohort": "", "rtol": "", "atol": ""}
    page = client.get("/global-best", params=params)
    assert page.status_code == 200
    assert "Kernel performance over time" in page.text
    assert str(first[0]) in page.text and str(second[0]) in page.text
    assert "0.7×" in page.text and "CATLASS" in page.text
    state.withdraw(engine, second[1], {"namespace": "team", "role": "user"})
    assert str(second[0]) not in client.get("/global-best", params=params).text
    assert client.get(f"/releases/{second[0]}").status_code == 404
    add(0.8, driver="other")
    assert "cohort_required" in client.get("/global-best", params=params).text
    assert (
        client.get(
            "/global-best", params={"kernel": m.kernel, "rtol": "nan", "atol": "0"}
        ).status_code
        == 422
    )


def test_detail_includes_saved_numerical_metrics_without_private_evidence(admission):
    engine, key, store, _ = admission
    process(
        engine, state.request_publication(engine, key, {"namespace": "team", "role": "user"}), store
    )
    from sqlalchemy.orm import Session

    from cannery.releases.models import Publication

    with Session(engine) as db:
        identifier = db.get(Publication, key).release_id
    page = web(engine).get(f"/releases/{identifier}")
    assert page.status_code == 200
    assert (
        "Saved numerical validation" in page.text
        and "Maximum absolute / relative error" in page.text
    )
    assert "private_evidence" not in page.text and "write_token" not in page.text
    assert "Manifest SHA256" in page.text and "Package SHA256" in page.text


def test_chart_uses_publication_times_and_includes_reference_line():
    points = [
        {
            "published_at": f"2026-01-01T00:00:{n:02d}+00:00",
            "geomean_speedup": v,
            "language": "ascendc",
        }
        for n, v in [(0, 0.5), (2, 0.7), (10, 1.2)]
    ]
    value = chart(points)
    assert value["points"][1]["x"] == pytest.approx(190)
    assert value["points"][2]["x"] == pytest.approx(750)
    assert value["line"].count(" H ") == 2 and value["line"].count(" V ") == 2
    assert 0 < value["reference_y"] < 220
    assert chart([])["points"] == []


def test_chart_subsecond_range_matches_axis():
    points = [
        {"published_at": f"2026-01-01T00:00:00.{ms}+00:00", "geomean_speedup": 1.0}
        for ms in ("000", "250", "500")
    ]
    plotted = chart(points)
    assert [p["x"] for p in plotted["points"]] == [tick["x"] for tick in plotted["dates"]]
    assert [tick["time"] for tick in plotted["dates"]] == ["00:00:00", "00:00:00.25", "00:00:00.5"]
    assert " V " not in plotted["line"]


def test_single_publication_is_centered_with_one_utc_date_and_readable_scale():
    result = chart(
        [
            {
                "published_at": "2026-01-01T02:00:00+02:00",
                "geomean_speedup": 0.5,
                "language": "ascendc",
            }
        ]
    )
    assert result["points"][0]["x"] == 400
    assert result["dates"] == [{"x": 400, "date": "01 Jan 2026", "time": "00:00:00"}]
    assert result["ticks"][0] == {"y": 220, "label": "0"}
    assert len(result["ticks"]) == 5
    assert result["best"] == "0.5"


def test_global_best_shows_names_teams_rounded_scores_and_compact_dates(releases):
    engine, add, _, manifest = releases
    release_id, candidate_id, _ = add(1.23456789)
    page = web(engine).get("/global-best", params={"kernel": manifest.kernel})
    assert page.status_code == 200
    assert f"Implementation {str(candidate_id)[:8]}" in page.text
    assert "<th>Team</th>" in page.text and "<td>team</td>" in page.text
    assert "1.235×" in page.text and "1.23456789" not in page.text
    assert ">01 Jan 2026</time>" in page.text
    assert 'datetime="2026-01-01T00:00:01+00:00"' in page.text
    assert f"/releases/{release_id}" in page.text and "Repository</a>" in page.text


def test_release_download_menu_pins_commands_and_gates_python_on_hub(published):
    engine, _, _, _, identifier, _ = published
    app = FastAPI()
    install_web(app, Settings(release_enabled=True, hub_enabled=True, auth_enabled=True), engine)
    page = TestClient(app).get(f"/releases/{identifier}")
    assert page.status_code == 200
    assert "Sign in to download" in page.text and "Download options" in page.text
    assert "Python · Hugging Face" in page.text and f"release-{identifier}" in page.text
    assert f"--release-id {identifier}" in page.text and "--no-detect" in page.text
    disabled = web(engine).get(f"/releases/{identifier}")
    assert "Python · Hugging Face" not in disabled.text and "CLI · download only" in disabled.text


def test_global_best_search_and_device_links(releases, monkeypatch):
    engine, add, _, manifest = releases
    add(0.5)
    add(0.4, device="a3")
    from cannery.releases.catalog import directory as published_directory

    monkeypatch.setattr(
        "cannery.web.releases.catalog.directory",
        lambda db: (
            published_directory(db)
            + [{"kernel": "team/other", "device": "a5", "version": 1, "submission_count": 1}]
        ),
    )
    client = web(engine)
    directory = client.get("/global-best")
    assert "Sort by device ascending" in directory.text
    assert "team/other" in directory.text
    assert '<select name="kernel">' not in directory.text
    page = client.get(
        "/global-best",
        params={"q": manifest.kernel.upper(), "kernel": manifest.kernel, "spec_version": 1},
    )
    assert page.status_code == 200
    assert "team/other" not in page.text
    query = f"kernel={manifest.kernel.replace('/', '%2F')}"
    search = query.upper().replace("KERNEL=", "q=")
    assert f'{query}&amp;device=a3&amp;{search}&amp;spec_version=1#kernel-results"' in page.text
    assert f'{query}&amp;device=a5&amp;{search}&amp;spec_version=1#kernel-results"' in page.text
    assert 'aria-current="page"' in page.text
    assert 'action="/global-best#kernel-results"' in page.text
    for name in ("device", "spec_version", "cohort", "rtol", "atol"):
        assert f'name="{name}"' in page.text
    assert (
        "No kernels match your search."
        in client.get("/global-best", params={"q": "no-such-kernel"}).text
    )


def test_directory_link_opens_older_published_spec(releases):
    from sqlalchemy.orm import Session

    from cannery.registry.models import SpecVersion

    engine, add, _, manifest = releases
    release_id, _, _ = add(0.5, device="a3")
    old_a5, _, _ = add(0.9, device="a5")
    with Session(engine) as db, db.begin():
        frozen = db.get(SpecVersion, manifest.spec_id)
        v2 = SpecVersion(
            kernel_id=frozen.kernel_id,
            version=2,
            content_sha256="f" * 64,
            manifest=frozen.manifest,
            snapshot={},
            file_sha256={},
            smoke={},
        )
        db.add(v2)
        db.flush()
        v2_id = v2.id
    a5_id, _, _ = add(0.6, spec_id=v2_id, spec_version=2)
    client = web(engine)
    url = f"/global-best?kernel={manifest.kernel.replace('/', '%2F')}&device=a3&spec_version=1"
    assert url.replace("&", "&amp;") in client.get("/global-best").text
    failed = client.get(url.removesuffix("&spec_version=1"))
    assert "no_published_release" in failed.text
    assert f'aria-label="{manifest.kernel} on A3" aria-current=' not in failed.text
    assert url.replace("&", "&amp;") in failed.text
    assert str(release_id) in client.get(url).text
    a5_url = url.replace("device=a3&spec_version=1", "device=a5&spec_version=2")
    directory = client.get("/global-best").text
    assert a5_url.replace("&", "&amp;") in directory
    assert str(a5_id) in directory and str(old_a5) not in directory
    assert "<td>2</td></tr>" in directory
    assert str(a5_id) in client.get(a5_url).text
    old = client.get(a5_url.replace("spec_version=2", "spec_version=1")).text
    assert a5_url.replace("&", "&amp;") in old
    assert f'aria-label="{manifest.kernel} on A5" aria-current=' not in old
    assert "Spec v1 · a5" in old


def test_global_best_recovers_if_publication_changes_during_render(releases, monkeypatch):
    from cannery.releases.catalog import SelectionError

    engine, add, _, manifest = releases
    add(0.5)

    def withdrawn(*args):
        raise SelectionError("no_published_release")

    monkeypatch.setattr("cannery.web.releases.catalog.evolution", withdrawn)
    page = web(engine).get("/global-best", params={"kernel": manifest.kernel})
    assert page.status_code == 200
    assert "no_published_release" in page.text
    assert "#kernel-results" in page.text


def test_device_directory_winner_and_slower_commit_evolution(releases):
    engine, add, _, manifest = releases
    winner, _, _ = add(0.9)
    slower, slower_submission, _ = add(0.6)
    add(0.8, device="a3")
    client = web(engine)
    query = {"q": manifest.kernel.upper(), "device_sort": "desc"}
    directory = client.get("/global-best", params=query).text
    assert directory.index(f"{manifest.kernel} on A5") < directory.index(f"{manifest.kernel} on A3")
    assert f"/releases/{winner}" in directory
    assert f"/releases/{slower}" not in directory
    filtered = client.get("/global-best", params=query | {"catalog_device": "a3"}).text
    assert f"{manifest.kernel} on A3" in filtered
    assert f"{manifest.kernel} on A5" not in filtered
    assert 'aria-sort="descending"' in filtered

    route = {"kernel": manifest.kernel, "catalog_device": "a5", "device_sort": "asc"}
    page = client.get("/global-best", params=route)
    assert page.status_code == 200
    assert page.text.count('class="chart-point"') == 2
    assert "<td>2</td></tr>" in page.text
    assert f'class="chart-point" href="/releases/{slower}"' in page.text
    assert manifest.source.commit in page.text
    assert "Chart revisions (2)" in page.text
    assert f'href="/releases/{slower}" title="{manifest.source.commit}"' in page.text
    assert "<strong>0.9×</strong>" in page.text
    state.withdraw(engine, slower_submission, {"namespace": "team", "role": "user"})
    after = client.get("/global-best", params=route).text
    assert after.count('class="chart-point"') == 1
    assert "<td>2</td></tr>" not in after

    add(1.1, driver="other")
    assert "Select cohort" in client.get("/global-best", params=query).text


def test_sortable_directory_columns_preserve_filters_and_place_missing_gains_last(releases):
    engine, add, _, manifest = releases
    a5, _, _ = add(2.0)
    a3, _, _ = add(11.0, device="a3")
    client = web(engine)
    route = {"kernel": manifest.kernel, "q": manifest.kernel, "device": "a5"}
    url = "/global-best"
    for order, expected in [("asc", (a5, a3)), ("desc", (a3, a5))]:
        page = client.get(url, params=route | {"sort": "gain", "direction": order})
        assert page.status_code == 200
        assert page.text.index(f"/releases/{expected[0]}") < page.text.index(
            f"/releases/{expected[1]}"
        )
    assert "<td>11×</td>" in page.text
    filter_query = route | {"catalog_device": "a3", "sort": "gain", "direction": "desc"}
    filtered = client.get(url, params=filter_query).text
    assert 'aria-label="Sort by gain ascending"' in filtered
    assert "catalog_device=a3&amp;sort=implementation&amp;direction=asc" in filtered
    assert 'name="sort" value="gain"' in filtered
    assert 'name="direction" value="desc"' in filtered
    assert f"{manifest.kernel} on A5" not in filtered

    add(3.0, driver="other")
    for order in ("asc", "desc"):
        for column in ("gain", "implementation"):
            page = client.get(url, params={"sort": column, "direction": order}).text
            assert page.index(f"{manifest.kernel} on A3") < page.index(f"{manifest.kernel} on A5")
    descending = client.get(url, params={"sort": "submissions", "direction": "desc"}).text
    ascending = client.get(url, params={"sort": "submissions", "direction": "asc"}).text
    assert descending.index("on A5") < descending.index("on A3")
    assert ascending.index("on A3") < ascending.index("on A5")
