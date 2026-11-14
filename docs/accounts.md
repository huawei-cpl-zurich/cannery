# Local accounts and service credentials

M7c adds operator-managed local accounts. Published rankings, release metadata
and history remain public. Private submissions, changes and binary downloads
require sign-in. Members share their assigned namespace; administrators may
select any namespace and access the read-only administration page. Usernames are
globally unique across all teams. Sign in with just your username and password;
your first assigned namespace is selected automatically. Switch teams after
signing in through Account. Administrators without explicit memberships start
in `team`; their existing access to any namespace is unchanged.

## Operator provisioning

Run account commands on a trusted host with the service database configured:

```sh
cannery accounts create admin --admin
cannery accounts grant admin team
cannery accounts create alice
cannery accounts grant alice team
cannery accounts set-role alice user
cannery accounts disable alice
cannery accounts enable alice
cannery accounts revoke alice team
cannery accounts reset-password alice
```

Create/reset prompts hide and confirm passwords. A user-owned regular UTF-8
file with mode 0600 can be supplied with `--password-file`; symlinks are rejected.
Passwords must contain 12–1024 characters and are hashed with Argon2id. There is
no public signup, email recovery or default account/password. Account and role
changes revoke all credentials; namespace removal revokes affected credentials.
Keep database access restricted to trusted operators.

## Browser and CLI

Visit `/login`, then `/account` to switch namespaces, change your password or
create/revoke a namespace-bound CLI token. Sessions expire after 12 hours; CLI
tokens expire after 30 days. Password changes sign out all sessions and tokens.
Neither sessions nor tokens are stored in browser local/session storage.

```sh
cannery login --url http://127.0.0.1:8095 --username alice --namespace team
cannery submissions --url http://127.0.0.1:8095
cannery get team/helloworld --url http://127.0.0.1:8095
cannery logout --url http://127.0.0.1:8095
```

CLI login creates a token in a private file under `~/.config/cannery/tokens/`.
Credentials are bound to the exact service origin, including its port. Use the
same URL for login and subsequent commands. HTTPS is required except for
loopback HTTP. CLI connections ignore ambient HTTP proxy settings to keep
credentials on the selected service connection. Logout revokes the token before deleting its local file; network
failure preserves it for retry. Tokens can also be revoked from the browser.

API clients send `Authorization: Bearer <token>`. Cookie-authenticated mutations
send `X-CSRF-Token` from `/api/v1/auth/me`. Login requires `X-Cannery-Login: 1`
and, for browser requests, the configured same origin. Cookies are HttpOnly,
SameSite=Strict and Secure by default. Tokens cannot issue additional tokens or
change their bound namespace. Authentication errors do not reveal passwords,
storage credentials or account existence.

## Deployment

Back up PostgreSQL, snapshot repositories, artifact data, durable worker journals
and operator configuration using the [maintenance procedure](maintenance.md).
Drain and stop writers before replacing a worker/controller image. Apply the
additive account migration and create the administrator plus its `team`
membership before changing API identity settings:

```sh
CANNERY_AUTH_ENABLED=true
CANNERY_DEV_IDENTITY_ENABLED=false
CANNERY_AUTH_ORIGIN=https://cannery.example.org
CANNERY_AUTH_COOKIE_SECURE=true
```

For this loopback-only development deployment, set the origin to the exact
browser address, for example `http://127.0.0.1:8095`, and explicitly set
`CANNERY_AUTH_COOKIE_SECURE=false`. Never use insecure cookies on a public host.
Authentication and development identity cannot be enabled together. Workers do
not receive account identity configuration or user tokens.

Login is limited per username and client address over five-minute windows.
Redis failure temporarily blocks login with 503 while existing database-backed
sessions continue. Database failure blocks protected access with bounded 503;
recovery reuses existing sessions without rerunning kernel jobs.

## Artifact credential rotation

Only the artifact service accepts `CANNERY_ARTIFACT_PREVIOUS_READ_TOKEN` and
`CANNERY_ARTIFACT_PREVIOUS_WRITE_TOKEN`. At most one previous value per capability
is accepted. Every configured value must be distinct and at least 32 characters.
Read credentials never authorize writes; write credentials never authorize reads.

1. Generate new private read/write credentials and back up operator configuration.
2. Configure old values as previous and new values as current on the artifact
   service; restart it. Old consumers continue working during this overlap.
3. Restart API/coordinators with their needed new current credentials after
   draining writers. Test authenticated download and artifact write/read.
4. Clear both previous values and restart the artifact service. Verify old values
   are rejected. Do not leave the overlap enabled indefinitely.

The API receives only an artifact reader. Trusted build/validation/benchmark/
publication coordinators receive the read/write capabilities they need. Git
credentials remain with the registrar; remote transport configuration remains
with trusted remote coordinators. Submitted-code containers receive none of
these control-plane or account credentials. Rollback must restore a compatible
verified backup and operator configuration; never silently enable development
identity as a fallback.
