from html.parser import HTMLParser

from test_account_authorization import setup as setup
from test_account_authorization import signed


class Forms(HTMLParser):
    def __init__(self):
        super().__init__()
        self.methods = []

    def handle_starttag(self, tag, attrs):
        if tag == "form":
            self.methods.append(dict(attrs).get("method", "get").lower())


def password_forms_never_use_get(html):
    parser = Forms()
    parser.feed(html)
    assert parser.methods and set(parser.methods) == {"post"}


def test_account_navigation_and_private_pages(setup):
    engine, client, _ = setup
    login = client.get("/login")
    password_forms_never_use_get(login.text)
    assert login.status_code == 200 and 'autocomplete="current-password"' in login.text
    assert 'name="namespace"' not in login.text
    assert "/static/accounts.js" in login.text and 'href="/login"' in login.text
    assert client.get("/account").status_code == 401
    navigation = client.get("/account", headers={"Accept": "text/html"}, follow_redirects=False)
    assert navigation.status_code == 303 and navigation.headers["location"] == "/login"
    signed(client, engine, "alice")
    account = client.get("/account")
    password_forms_never_use_get(account.text)
    assert account.status_code == 200 and account.headers["cache-control"] == "no-store"
    assert 'href="/account"' in account.text and 'href="/login"' not in account.text
    assert "Change password and sign out everywhere" in account.text
    assert "cannery_session=" not in account.text
