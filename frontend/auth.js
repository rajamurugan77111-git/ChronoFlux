// ChronoFlux console: session, CSRF, user menu, admin panels and live-sensor status.
// Loaded before app.js. Wraps fetch so every same-origin state-changing request
// carries the CSRF token, and any 401 sends the user back to the sign-in page.
(function () {
  const CF = (window.CF_AUTH = { user: null, role: 'analyst', csrf: sessionStorage.getItem('cf_csrf') || '', enabled: true });
  const nativeFetch = window.fetch.bind(window);

  window.fetch = async (input, init = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    const sameOrigin = url.startsWith('/') || url.startsWith(location.origin);
    const method = (init.method || 'GET').toUpperCase();
    if (sameOrigin && method !== 'GET' && CF.csrf) {
      const headers = new Headers(init.headers || {});
      headers.set('X-CSRF-Token', CF.csrf);
      init = { ...init, headers };
    }
    const resp = await nativeFetch(input, init);
    if (sameOrigin && resp.status === 401 && !url.includes('/api/auth/') && !url.includes('/api/sensor/live')) {
      location.href = '/login';
    }
    return resp;
  };

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const json = async (path, opts) => { const r = await fetch(path, opts); return [r, await r.json().catch(() => ({}))]; };

  async function loadMe() {
    const [r, me] = await json('/api/auth/me');
    if (!r.ok) return;
    Object.assign(CF, { user: me.user, role: me.role, enabled: me.auth_enabled });
    if (me.csrf) { CF.csrf = me.csrf; sessionStorage.setItem('cf_csrf', me.csrf); }
    document.body.classList.toggle('is-admin', me.role === 'admin');
    const chip = $('userChip');
    if (!chip) return;
    chip.innerHTML = me.auth_enabled
      ? `<strong>${esc(me.user)}</strong><span class="role">${esc(me.role)}</span><button id="cfLogout">Sign out</button>`
      : `<span class="role">auth off</span>`;
    const lo = $('cfLogout');
    if (lo) lo.onclick = async () => { await fetch('/api/auth/logout', { method: 'POST' }); sessionStorage.removeItem('cf_csrf'); location.href = '/login'; };
  }

  async function loadSensor() {
    const box = $('sensorStatus'), chip = $('sensorChip');
    if (!box) return;
    const [r, s] = await json('/api/sensor/status');
    if (!r.ok) return;
    chip.textContent = s.connected ? 'ONLINE' : 'NO LIVE SENSOR';
    chip.style.color = s.connected ? '#0d9488' : '';
    box.innerHTML = s.sensors.length
      ? s.sensors.map((x) => `<div class="urow"><span><b>${esc(x.sensor_id)}</b> · ${x.online ? '🟢 online' : '⚪ offline'}</span>
          <span>last window ${x.age_seconds != null ? Math.round(x.age_seconds) + 's ago' : '?'} · ${x.packets_last_window ?? 0} packets · ${esc(x.last_stage || '-')}</span></div>`).join('')
      : `No live sensor has connected yet.${s.sensor_key_required ? ' Sensors must send CHRONOFLUX_SENSOR_KEY.' : ''}`;
  }

  async function loadAdmin() {
    if (CF.role !== 'admin') return;
    const [, u] = await json('/api/users');
    $('usersList').innerHTML = (u.users || []).map((x) => `<div class="urow"><span>${esc(x.username)} <span class="role" style="font-size:10px">${esc(x.role)}</span></span>
      ${x.username === CF.user ? '<span>you</span>' : `<button class="secondary-button" data-del="${esc(x.username)}">Remove</button>`}</div>`).join('');
    $('usersList').querySelectorAll('[data-del]').forEach((b) => (b.onclick = async () => {
      const [r, body] = await json('/api/users/' + encodeURIComponent(b.dataset.del), { method: 'DELETE' });
      $('usersMsg').textContent = r.ok ? 'Removed' : body.detail; loadAdmin();
    }));
    const [, a] = await json('/api/alerts');
    $('alertsInfo').innerHTML = (a.enabled
      ? `Channels: <b>${a.channels.join(', ')}</b> · risk ≥ ${a.min_score} · cool-down ${Math.round(a.cooldown_seconds / 60)} min`
      : 'No channel configured. Set CHRONOFLUX_SMTP_* or CHRONOFLUX_TELEGRAM_* in .env.') +
      (a.recent || []).slice(0, 5).map((x) => `<div class="urow"><span>${esc(x.ip_address)} · ${esc(x.stage || '')}</span>
        <span>${esc(x.severity)} · ${new Date(x.created_at * 1000).toLocaleTimeString()}</span></div>`).join('');
  }

  document.addEventListener('DOMContentLoaded', () => {
    loadMe().then(() => { loadSensor(); loadAdmin(); });
    setInterval(loadSensor, 10000);
    const add = $('nuAdd');
    if (add) add.onclick = async () => {
      const [r, body] = await json('/api/users', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: $('nuName').value, password: $('nuPass').value, role: $('nuRole').value }) });
      $('usersMsg').textContent = r.ok ? 'User added ✓' : (Array.isArray(body.detail) ? 'Invalid input' : body.detail);
      if (r.ok) { $('nuName').value = ''; $('nuPass').value = ''; loadAdmin(); }
    };
    const t = $('alertTest');
    if (t) t.onclick = async () => {
      const [r, body] = await json('/api/alerts/test', { method: 'POST' });
      $('alertMsg').textContent = r.ok ? 'Result: ' + JSON.stringify(body.results) : body.detail; loadAdmin();
    };
  });
})();
