from test_account_authorization import setup as setup
from test_account_authorization import signed


def test_account_navigation_and_private_pages(setup):
    engine, client, _ = setup
    login = client.get("/login")
    assert login.status_code == 200 and 'autocomplete="current-password"' in login.text
    assert "/static/accounts.js" in login.text and 'href="/login"' in login.text
    assert client.get("/account").status_code == 401
    navigation = client.get("/account", headers={"Accept": "text/html"}, follow_redirects=False)
    assert navigation.status_code == 303 and navigation.headers["location"] == "/login"
    signed(client, engine, "alice")
    account = client.get("/account")
    assert account.status_code == 200 and account.headers["cache-control"] == "no-store"
    assert 'href="/account"' in account.text and 'href="/login"' not in account.text
    assert "Change password and sign out everywhere" in account.text
    assert "cannery_session=" not in account.text
