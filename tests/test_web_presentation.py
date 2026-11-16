import pytest

from cannery.web.presentation import number, publication_date, source_links


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "—"),
        (0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (1.23456789, "1.235"),
        (1234.5, "1,234.5"),
        (999.9999, "1,000"),
        (1000000, "1,000,000"),
        (9007199254740993, "9,007,199,254,740,993"),
        (0.001, "0.001"),
        (0.0001, "1e-4"),
        (1.23456789e-8, "1.235e-8"),
        (-1e-12, "-1e-12"),
        (float("nan"), "undefined"),
        (float("inf"), "undefined"),
    ],
)
def test_consistent_numbers_without_false_zero_or_lost_integer_precision(value, expected):
    assert number(value) == expected


def test_publication_dates_are_utc():
    assert publication_date("2026-01-02T00:30:00+02:00") == "01 Jan 2026"


def source(url, path="kernels/add one/#test?"):
    return {"url": url, "path": path, "commit": "a" * 40}


@pytest.mark.parametrize(
    "host,prefix",
    [
        ("github.com", "/tree/"),
        ("gitlab.com", "/-/tree/"),
        ("gitcode.com", "/tree/"),
        ("gitee.com", "/tree/"),
    ],
)
@pytest.mark.parametrize("transport", ["https://", "ssh://git@"])
def test_supported_hosts_link_exact_commit_and_encoded_source(host, prefix, transport):
    links = source_links(source(transport + host + "/team/repo.git"))
    assert links == {
        "repository": "https://" + host + "/team/repo",
        "code": "https://"
        + host
        + "/team/repo"
        + prefix
        + "a" * 40
        + "/kernels/add%20one/%23test%3F",
    }


def test_nested_gitlab_groups_root_and_trailing_clone_slash():
    assert source_links(source("ssh://git@gitlab.com/group/sub/repo.git/", ".")) == {
        "repository": "https://gitlab.com/group/sub/repo",
        "code": "https://gitlab.com/group/sub/repo/-/tree/" + "a" * 40,
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/repo.git",
        "https://github.com.evil.example/repo.git",
        "https://github.com:8443/team/repo.git",
    ],
)
def test_unknown_https_routes_do_not_guess_provider_or_strip_ports(url):
    assert source_links(source(url)) == {"repository": url, "code": None}


def test_unknown_ssh_is_not_a_browser_link():
    assert source_links(source("ssh://git@example.com/team/repo.git")) is None
