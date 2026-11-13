import pytest

from cannery.settings import Settings


@pytest.mark.parametrize(
    "origin,secure",
    [
        ("http://public.invalid", True),
        ("https://public.invalid", False),
        ("https://user:secret@public.invalid", True),
        ("https://public.invalid/path", True),
        ("https://public.invalid?token=secret", True),
    ],
)
def test_unsafe_account_configuration_is_rejected(origin, secure):
    with pytest.raises(ValueError):
        Settings(auth_enabled=True, auth_origin=origin, auth_cookie_secure=secure)


def test_loopback_requires_explicit_cookie_opt_out_and_modes_are_exclusive():
    assert Settings(auth_enabled=True).auth_cookie_secure
    assert not Settings(auth_enabled=True, auth_cookie_secure=False).auth_cookie_secure
    assert Settings(auth_enabled=True, auth_origin="https://public.invalid").auth_enabled
    with pytest.raises(ValueError, match="cannot be combined"):
        Settings(auth_enabled=True, dev_identity_enabled=True)
