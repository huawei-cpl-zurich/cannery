from test_account_authorization import setup as setup
from test_account_authorization import signed

from cannery.accounts import operator


def test_admin_summaries_require_current_role_and_hide_credentials(setup):
    engine, client, _ = setup
    assert client.get("/api/v1/admin/summary").status_code == 401
    signed(client, engine, "alice")
    assert client.get("/api/v1/admin/summary").status_code == 403
    signed(client, engine, "operator")
    response = client.get("/api/v1/admin/summary")
    assert response.status_code == 200
    assert response.json()["queues"]["registration"] == {"queued": 1}
    assert response.json()["frozen_specifications"] == 0
    assert client.get("/admin").status_code == 200
    for value in ["password", "token", "ssh", "127.0.0.1", "/var/lib"]:
        assert value not in response.text
    operator.configure(engine, "operator", role="user")
    assert client.get("/api/v1/admin/summary").status_code == 401
    signed(client, engine, "operator")
    assert client.get("/api/v1/admin/summary").status_code == 403
