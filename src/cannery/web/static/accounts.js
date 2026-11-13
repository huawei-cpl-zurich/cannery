/* Same-origin requests only. No session or API credentials in browser storage. */
(() => {
  let csrf = null;
  function error(message) {
    let node = document.getElementById('account-error');
    if (!node) {
      node = document.createElement('p'); node.setAttribute('role', 'alert');
      document.getElementById('content').prepend(node);
    }
    node.textContent = message;
  }
  async function request(path, method = 'GET', data = null) {
    const headers = {};
    if (method !== 'GET') {
      if (!csrf && path !== '/api/v1/auth/login') {
        csrf = (await request('/api/v1/auth/me')).csrf_token;
      }
      headers['X-CSRF-Token'] = csrf || '';
      if (path === '/api/v1/auth/login') headers['X-Cannery-Login'] = '1';
    }
    if (data !== null) headers['Content-Type'] = 'application/json';
    const reply = await fetch(path, {method, headers, credentials: 'same-origin',
      body: data === null ? undefined : JSON.stringify(data)});
    if (!reply.ok) {
      if (reply.status === 401 && path !== '/api/v1/auth/login') location.assign('/login');
      throw new Error(reply.status === 429 ? 'Too many attempts. Try again in five minutes.' :
        reply.status === 503 ? 'Service temporarily unavailable. Try again.' :
        reply.status === 401 ? 'Invalid username or password.' : 'Request denied. Check your input and permissions.');
    }
    return reply.status === 204 ? null : reply.json();
  }
  async function account() {
    if (!document.getElementById('account-identity')) return;
    const user = await request('/api/v1/auth/me'); csrf = user.csrf_token;
    document.getElementById('account-identity').textContent = `${user.username} · ${user.role} · ${user.namespace}`;
    const list = document.getElementById('account-namespaces'); list.replaceChildren();
    user.namespaces.forEach(name => { const node = document.createElement('option'); node.value = name; list.append(node); });
    document.querySelectorAll('#account-namespace input, #account-token input[name="namespace"]').forEach(node => { node.value = user.namespace; });
    const tokens = await request('/api/v1/auth/tokens');
    const output = document.getElementById('account-tokens'); output.replaceChildren();
    tokens.forEach(token => {
      const item = document.createElement('li');
      item.textContent = `${token.label} · ${token.namespace} · expires ${token.expires_at} `;
      const button = document.createElement('button'); button.textContent = 'Revoke';
      button.type = 'button'; button.onclick = () => request(`/api/v1/auth/tokens/${token.id}`, 'DELETE').then(account).catch(e => error(e.message));
      item.append(button); output.append(item);
    });
  }
  document.addEventListener('submit', async event => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const action = new URL(form.action, location.href);
    if (action.origin !== location.origin) return;
    const isAccount = form.id.startsWith('account-');
    if (!isAccount && form.method.toLowerCase() !== 'post') return;
    event.preventDefault();
    try {
      const data = Object.fromEntries(new FormData(form));
      if (isAccount) {
        const operation = form.id.slice(8);
        if (operation === 'login' && !data.namespace) delete data.namespace;
        const value = await request(`/api/v1/auth/${operation === 'token' ? 'tokens' : operation}`, 'POST', data);
        if (operation === 'login') { csrf = value.csrf_token; location.assign('/my-submissions'); }
        else if (operation === 'password') location.assign('/login');
        else if (operation === 'token') document.getElementById('account-new-token').textContent = value.token;
        if (operation !== 'login' && operation !== 'password') await account();
      } else {
        if (!csrf) csrf = (await request('/api/v1/auth/me')).csrf_token;
        const reply = await fetch(action, {method: 'POST', credentials: 'same-origin',
          headers: {'X-CSRF-Token': csrf}, body: new URLSearchParams(new FormData(form))});
        if (reply.redirected) location.assign(reply.url);
        else if (reply.ok) {
          const parsed = new DOMParser().parseFromString(await reply.text(), 'text/html');
          const content = parsed.getElementById('content');
          if (content) { document.getElementById('content').replaceWith(content); if (window.htmx) htmx.process(content); }
          else error('Unexpected response. Reload and try again.');
        } else error('Request failed. Reload and check your permissions.');
      }
    } catch (e) { error(e.message); }
  });
  document.addEventListener('click', event => {
    if (event.target.id === 'account-logout') request('/api/v1/auth/logout', 'POST').then(() => location.assign('/login')).catch(e => error(e.message));
  });
  document.addEventListener('htmx:afterSwap', () => account().catch(e => error(e.message)));
  account().catch(e => error(e.message));
})();
