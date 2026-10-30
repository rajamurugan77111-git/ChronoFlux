const state = {
  latest: null,
  history: [],
  system: null,
  evidence: {verification: {valid:true, blocks:0}, items: []},
  devices: {items: [], total: 0, new: 0, elevated: 0},
  forecastEval: {verified_forecasts: 0, correct_forecasts: 0},
  selectedFile: null,
  activePage: 'overview',
  offlineAbort: null,
};

const pageMeta = {
  overview: ['Overview', ''],
  simulation: ['Simulation', ''],
  live: ['Traffic', ''],
  devices: ['Devices', ''],
  offline: ['Offline', ''],
  forecast: ['Forecast', ''],
  alerts: ['Alerts', ''],
  evidence: ['Evidence', ''],
  system: ['System', ''],
};

const $ = (id) => document.getElementById(id);
const fmtPct = (value) => value == null ? '--%' : `${(Number(value) <= 1 ? Number(value) * 100 : Number(value)).toFixed(1)}%`;
const fmtTime = (value) => {
  if (!value) return '--:--:--';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? String(value).slice(-8) : d.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
};
const safe = (value, fallback='--') => value == null || value === '' ? fallback : value;
const stageClass = (stage='') => {
  const s = stage.toLowerCase();
  if (s === 'normal') return 'normal';
  if (s.includes('suspicious') || s.includes('recon') || s.includes('fuzz') || s.includes('analysis')) return 'medium';
  if (s.includes('backdoor') || s.includes('shell') || s.includes('worm') || s.includes('command') || s.includes('exfil')) return 'critical';
  return 'high';
};
const riskClass = (risk='') => risk.toLowerCase();

function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => el.classList.remove('show'), 2600);
}

async function request(path, options={}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function navigate(page) {
  if (!pageMeta[page]) return;
  const pageEl = $(`page-${page}`);
  if (!pageEl) return;
  // Leaving Simulation should pause auto-play so it doesn't keep advancing in the background
  if (state.activePage === 'simulation' && page !== 'simulation') stopSimulationTimer();
  state.activePage = page;
  document.body.classList.toggle('sim-page-active', page === 'simulation');
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.toggle('active', btn.dataset.page === page));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  pageEl.classList.add('active');
  $('pageTitle').textContent = pageMeta[page][0];
  if ($('pageSubtitle')) $('pageSubtitle').textContent = pageMeta[page][1] || '';
  $('sidebar').classList.remove('open');
  window.scrollTo(0, 0);
  if (page === 'system') loadSystem();
  if (page === 'alerts') renderAlerts();
  if (page === 'forecast') renderForecastPage();
  if (page === 'evidence') loadEvidence();
  if (page === 'devices') renderDevices();
  if (page === 'simulation') startSimulationPage();
  if (page === 'overview') renderOverview();
  if (page === 'live') {
    renderHistoryTable();
    renderCharts();
  }
}

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    navigate(btn.dataset.page);
  });
});

document.querySelectorAll('[data-goto]').forEach(el => {
  el.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    navigate(el.dataset.goto);
  });
});

$('menuButton').addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  $('sidebar').classList.toggle('open');
});

function showDemoReadyBanner(message) {
  const banner = $('demoReadyBanner');
  if (banner) {
    banner.classList.remove('hidden');
    if ($('demoReadyText') && message) $('demoReadyText').textContent = message;
  }
  const simBanner = $('simDemoBanner');
  if (simBanner && state.activePage === 'simulation') simBanner.classList.remove('hidden');
}

$('dismissDemoBanner')?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  $('demoReadyBanner')?.classList.add('hidden');
});
$('dismissSimBanner')?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  $('simDemoBanner')?.classList.add('hidden');
});
function applyTheme() {
  const theme = localStorage.getItem('cf-theme') || 'light';
  document.body.classList.toggle('dark', theme === 'dark');
}
$('themeToggle').addEventListener('click', () => {
  const next = document.body.classList.contains('dark') ? 'light' : 'dark';
  localStorage.setItem('cf-theme', next);
  applyTheme();
  renderCharts();
});
applyTheme();

function setBackendStatus(online) {
  $('backendStatus').textContent = online ? 'Backend online' : 'Backend offline';
  document.querySelector('.pulse-dot').classList.toggle('online', online);
}

function renderTimeline() {
  const container = $('timelineGrid');
  const recent = state.history.slice(-4);
  const items = Array(Math.max(0, 4 - recent.length)).fill(null).concat(recent);
  container.innerHTML = items.map((item, idx) => {
    if (!item) return `<div class="timeline-card"><div class="timeline-label">T${idx+1}</div><div class="timeline-state muted">Waiting</div><div class="timeline-meta">No window received</div></div>`;
    return `<div class="timeline-card">
      <div class="timeline-label">T${idx+1}${idx===3 ? ' · CURRENT' : ''}</div>
      <div class="timeline-state"><span class="badge ${stageClass(item.current_stage)}">${safe(item.current_stage)}</span></div>
      <div class="timeline-meta">${fmtTime(item.timestamp)}<br>${Number(item.packet_count||0)} packets · anomaly ${Number(item.anomaly_score||0).toFixed(0)}%</div>
    </div>`;
  }).join('');
}

function renderProgression(forecast) {
  const progression = forecast?.progression || [];
  $('progression').innerHTML = progression.length ? progression.map((s, i) => `${i ? '<span class="arrow">→</span>' : ''}<span class="node">${s}</span>`).join('') : '<span class="muted tiny">No progression generated yet</span>';
}

function renderOverview() {
  const latest = state.latest;
  const pred = latest?.prediction;
  const telemetry = latest?.telemetry || {};
  const forecast = pred?.forecast || {};

  if (!pred) {
    $('currentState').textContent = 'Waiting';
    $('currentStateMeta').textContent = 'Click Run demo or upload CSV/PCAP in Offline';
    $('sidebarSensorText').textContent = 'Waiting';
    $('sidebarSensorDot').classList.remove('online');
    if ($('sidebarSensorMeta')) $('sidebarSensorMeta').textContent = 'No traffic yet — use Run demo';
    const banner = $('statusBanner');
    if (banner) banner.className = 'status-banner';
    renderTimeline();
    renderCharts();
    return;
  }

  $('currentState').textContent = pred.current_stage;
  $('currentStateMeta').textContent = `ML: ${safe(pred.ml_stage)} · ${Number(telemetry.packet_count||0)} packets · ${safe(pred.risk_level)} risk`;
  const banner = $('statusBanner');
  if (banner) banner.className = `status-banner risk-${riskClass(pred.risk_level || 'low')}`;
  $('anomalyScore').textContent = `${Number(pred.anomaly_score||0).toFixed(0)}%`;
  $('anomalyLabel').textContent = safe(pred.anomaly_label);
  $('riskLevel').textContent = safe(pred.risk_level);
  $('riskLevel').className = `metric-value risk-tone ${riskClass(pred.risk_level || '')}`;
  $('riskScore').textContent = Number(pred.risk_score||0).toFixed(0);
  $('confidenceValue').textContent = fmtPct(pred.current_stage_confidence);
  $('confidenceMeta').textContent = `Attack probability ${fmtPct(pred.attack_probability)}`;
  $('healthScore').textContent = Math.max(0, 100 - Number(pred.risk_score||0)).toFixed(0);
  $('healthNote').textContent = pred.risk_level === 'LOW' ? 'Behaviour within baseline' : `${pred.risk_level} attention level`;

  $('forecastNext').textContent = safe(forecast.next_stage, '—');
  $('forecastConfidence').textContent = forecast.next_stage == null ? 'Need more windows' : `${fmtPct(forecast.confidence)} confidence`;
  const modelReadyOverview = !!(state.system?.models?.world_model || forecast.world_model_ready || forecast.model_name || forecast.mode === 'lstm_world_model');
  const modeLabel = forecast.status === 'active'
    ? 'ACTIVE'
    : modelReadyOverview
      ? (state.history.length < 4 ? 'NEED T1–T4' : 'LOW CONF')
      : 'NOT TRAINED';
  $('forecastMode').textContent = modeLabel;
  $('forecastEvidence').textContent = safe(forecast.evidence, 'Waiting for sequence evidence.');
  renderProgression(forecast);

  $('mlStage').textContent = safe(pred.ml_stage);
  $('mlConfidence').textContent = fmtPct(pred.ml_confidence);
  $('engineAnomaly').textContent = `${Number(pred.anomaly_score||0).toFixed(0)}%`;
  $('engineAnomalyLabel').textContent = safe(pred.anomaly_label);
  const signatures = pred.signatures || [];
  $('signatureStatus').textContent = signatures.length ? signatures[0].name.replaceAll('_',' ') : 'No active rule';
  $('signatureCount').textContent = `${signatures.length} signal${signatures.length===1?'':'s'}`;
  $('behaviourStage').textContent = safe(pred.current_stage);
  $('riskFormula').textContent = safe(pred.risk_formula, '0.50*ML + 0.30*Anomaly + 0.20*Signature');
  const rc = pred.risk_components || {};
  $('riskComponents').textContent = `ML ${Number(rc.ml||0).toFixed(0)} · IF ${Number(rc.anomaly||0).toFixed(0)} · Rules ${Number(rc.signature||0).toFixed(0)}`;
  $('attackCategory').textContent = `Category ${safe(pred.current_attack_category || pred.ml_stage)}`;
  $('mitreTactics').textContent = (pred.mitre_tactics || []).length ? pred.mitre_tactics.join(' · ') : 'No tactic mapped';

  $('whyList').innerHTML = (pred.explanations || ['No explanation available.']).map(x => `<li>${x}</li>`).join('');
  $('actionList').innerHTML = (pred.recommended_actions || ['Continue monitoring.']).map(x => `<li>${x}</li>`).join('');

  $('sidebarSensorText').textContent = 'Traffic active';
  $('sidebarSensorDot').classList.add('online');
  $('sidebarSensorMeta').textContent = `${safe(latest.sensor_id, 'demo-lab')} · ${fmtTime(latest.timestamp)}`;

  $('livePackets').textContent = Number(telemetry.packet_count || 0).toLocaleString();
  $('liveRate').textContent = Number(telemetry.packet_rate || 0).toFixed(1);
  $('livePorts').textContent = Number(telemetry.unique_dst_ports || 0);
  $('liveSyn').textContent = Number(telemetry.syn_count || 0);

  const meta = {
    'Source feed': latest.sensor_id,
    'Source IP': telemetry.source_ip,
    'Source MAC': telemetry.source_mac,
    'Source port': telemetry.source_port,
    'Primary destination': telemetry.primary_destination,
    'Destination port': telemetry.destination_port,
    'Protocol': String(safe(telemetry.protocol, latest.features?.proto || '--')).toUpperCase(),
    'Service': String(safe(telemetry.service, latest.features?.service || '--')).toUpperCase(),
    'Unique destination IPs': telemetry.unique_dst_ips,
    'Unique destination ports': telemetry.unique_dst_ports,
    'SYN packets': telemetry.syn_count,
    'RST packets': telemetry.rst_count,
    'ICMP packets': telemetry.icmp_count,
    'ARP packets': telemetry.arp_count,
    'Outbound bytes': Number(telemetry.outbound_bytes||0).toLocaleString(),
    'Inbound bytes': Number(telemetry.inbound_bytes||0).toLocaleString(),
  };
  $('sensorMeta').innerHTML = Object.entries(meta).map(([k,v]) => `<div class="key-value-row"><span>${k}</span><strong>${safe(v)}</strong></div>`).join('');

  // Deep flow intelligence (metadata only)
  $('overviewSource').textContent = safe(telemetry.source_ip);
  $('overviewSourcePort').textContent = `Port ${safe(telemetry.source_port)}`;
  $('overviewDestination').textContent = safe(telemetry.primary_destination);
  $('overviewDestinationPort').textContent = `Port ${safe(telemetry.destination_port)}`;
  $('overviewProtocol').textContent = String(safe(telemetry.protocol, latest.features?.proto || '--')).toUpperCase();
  const flags = Array.isArray(telemetry.tcp_flags) ? telemetry.tcp_flags.join(', ') : safe(telemetry.tcp_flags);
  $('overviewFlags').textContent = `Flags ${flags || '--'}`;
  const services = Array.isArray(telemetry.services_seen) && telemetry.services_seen.length ? telemetry.services_seen : [safe(telemetry.service, latest.features?.service || '--')];
  $('overviewServices').innerHTML = services.map(x => `<span>${String(x).toUpperCase()}</span>`).join('');

  $('connSource').textContent = safe(telemetry.source_ip);
  $('connSourcePort').textContent = `Source port ${safe(telemetry.source_port)}`;
  $('connDestination').textContent = safe(telemetry.primary_destination);
  $('connDestinationPort').textContent = `Destination port ${safe(telemetry.destination_port)}`;
  $('connProtocol').textContent = String(safe(telemetry.protocol, latest.features?.proto || '--')).toUpperCase();
  $('connService').textContent = `Service ${String(safe(telemetry.service, latest.features?.service || '--')).toUpperCase()}`;
  $('connFlags').textContent = flags || '--';
  $('connPortSpread').textContent = `${Number(telemetry.unique_dst_ports || 0)} destination ports`;
  const portList = Array.isArray(telemetry.dst_ports_sample) ? telemetry.dst_ports_sample : [];
  $('connPortList').innerHTML = portList.length ? portList.map(p => `<span>PORT ${p}</span>`).join('') : '<span>No destination ports observed</span>';

  renderEvidenceSummary();
  renderTimeline();
  renderHistoryTable();
  renderCharts();
  if (state.activePage === 'forecast') renderForecastPage();
}

function renderHistoryTable() {
  const rows = state.history.slice(-20).reverse();
  $('historyTable').innerHTML = rows.length ? rows.map(item => `<tr>
    <td>${fmtTime(item.timestamp)}</td>
    <td><span class="badge ${stageClass(item.current_stage)}">${item.current_stage}</span></td>
    <td><span class="badge ${riskClass(item.risk_level)}">${item.risk_level}</span></td>
    <td>${Number(item.anomaly_score||0).toFixed(0)}%</td>
    <td>${Number(item.packet_count||0)}</td>
    <td>${safe(item.forecast?.next_stage, 'Monitoring')}</td>
  </tr>`).join('') : '<tr><td colspan="6" class="muted">No live traffic windows yet.</td></tr>';
}

function renderAlerts() {
  const alerts = state.history.filter(x => ['MEDIUM','HIGH','CRITICAL'].includes(x.risk_level)).slice(-50).reverse();
  $('alertsTable').innerHTML = alerts.map(item => `<tr>
    <td>${fmtTime(item.timestamp)}</td>
    <td>${item.current_stage}</td>
    <td><span class="badge ${riskClass(item.risk_level)}">${item.risk_level}</span></td>
    <td>${Number(item.risk_score||0).toFixed(0)}</td>
    <td>${item.signatures?.[0]?.name?.replaceAll('_',' ') || 'ML / anomaly'}</td>
    <td>${safe(item.forecast?.next_stage, 'Monitoring')}</td>
  </tr>`).join('');
  $('alertsEmpty').classList.toggle('hidden', alerts.length > 0);
}


function shortHash(value) {
  if (!value) return 'No evidence block yet';
  const v = String(value);
  return v.length > 24 ? `${v.slice(0,12)}…${v.slice(-10)}` : v;
}

function renderEvidenceSummary() {
  const verification = state.evidence?.verification || {};
  const items = state.evidence?.items || [];
  const latestBlock = items[0];
  const valid = verification.valid !== false;
  $('overviewChainState').textContent = valid ? 'VERIFIED' : 'CHECK FAILED';
  $('overviewChainState').classList.toggle('green-chip', valid);
  $('overviewBlockCount').textContent = Number(verification.blocks || 0).toLocaleString();
  $('overviewLatestEvent').textContent = latestBlock?.event?.event_id || 'None';
  $('overviewLatestHash').textContent = shortHash(verification.latest_block_hash || latestBlock?.block_hash);
}

function renderEvidencePage() {
  const verification = state.evidence?.verification || {};
  const items = state.evidence?.items || [];
  const valid = verification.valid !== false;

  $('chainIntegrity').textContent = valid ? 'VERIFIED' : 'FAILED';
  $('chainIntegrityNote').textContent = safe(verification.reason, valid ? 'All hashes and links verified' : 'Integrity check failed');
  $('chainBlocks').textContent = Number(verification.blocks || 0).toLocaleString();
  $('chainLatestEvent').textContent = verification.latest_event_id || items[0]?.event?.event_id || 'None';

  $('evidenceTable').innerHTML = items.length ? items.map(block => {
    const event = block.event || {};
    return `<tr>
      <td><strong>${safe(event.event_id)}</strong><br><span class="muted tiny">Block #${Number(block.index||0)+1}</span></td>
      <td>${fmtTime(event.event_time || block.created_at)}</td>
      <td><span class="badge ${stageClass(event.stage)}">${safe(event.stage)}</span></td>
      <td><span class="badge ${riskClass(event.risk_level)}">${safe(event.risk_level)}</span></td>
      <td>${safe(event.source_ip)}${event.source_port != null ? ':'+event.source_port : ''}</td>
      <td>${safe(event.destination_ip)}${event.destination_port != null ? ':'+event.destination_port : ''}</td>
      <td>${String(safe(event.protocol)).toUpperCase()}</td>
      <td title="${safe(block.block_hash)}"><code>${shortHash(block.block_hash)}</code></td>
    </tr>`;
  }).join('') : '<tr><td colspan="8" class="muted">No elevated evidence events committed yet.</td></tr>';

  $('evidenceEmpty').classList.toggle('hidden', items.length > 0);
}

async function loadEvidence() {
  try {
    state.evidence = await request('/api/evidence?limit=100');
    renderEvidenceSummary();
    renderEvidencePage();
  } catch (err) {
    toast(`Evidence chain error: ${err.message}`);
  }
}

function drawLineChart(canvas, values, strokeColor='#2563eb') {
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(400, rect.width * ratio);
  canvas.height = 280 * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const w = rect.width;
  const h = 280;
  const cs = getComputedStyle(document.body);
  const line = cs.getPropertyValue('--line').trim() || '#e5eaf1';
  const muted = cs.getPropertyValue('--muted').trim() || '#667085';
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = line;
  ctx.lineWidth = 1;
  for (let i=0;i<5;i++) {
    const y = 20 + i * 55;
    ctx.beginPath(); ctx.moveTo(38,y); ctx.lineTo(w-12,y); ctx.stroke();
  }
  if (!values.length) {
    ctx.fillStyle = muted; ctx.font = '12px system-ui'; ctx.fillText('Waiting for live traffic windows', 48, 140); return;
  }
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const span = Math.max(max-min, 1);
  const points = values.map((v,i) => ({x: 42 + (i/(Math.max(values.length-1,1))) * (w-70), y: 245 - ((v-min)/span)*190}));
  ctx.strokeStyle = strokeColor; ctx.lineWidth = 2.4; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  ctx.beginPath(); points.forEach((p,i) => i ? ctx.lineTo(p.x,p.y) : ctx.moveTo(p.x,p.y)); ctx.stroke();
  ctx.fillStyle = strokeColor; points.forEach(p => {ctx.beginPath(); ctx.arc(p.x,p.y,2.8,0,Math.PI*2); ctx.fill();});
  ctx.fillStyle = muted; ctx.font = '10px system-ui'; ctx.fillText(`${max.toFixed(1)}`, 3, 28); ctx.fillText(`${min.toFixed(1)}`, 3, 248);
}

function renderCharts() {
  const rates = state.history.slice(-30).map(x => Number(x.packet_rate || 0));
  const packets = state.history.slice(-30).map(x => Number(x.packet_count || 0));
  drawLineChart($('trafficChart'), rates, '#2563eb');
  drawLineChart($('liveChart'), packets, '#7c3aed');
  $('trafficChip').textContent = `${state.history.length} window${state.history.length===1?'':'s'}`;
}

function renderForecastPage() {
  const recent = state.history.slice(-4);
  if ($('forecastFlow')) {
    $('forecastFlow').innerHTML = recent.length
      ? recent.map((item, i) => `<div class="forecast-hist-card"><small>T${i + 1}</small><strong>${safe(item.current_stage)}</strong></div>`).join('')
      : '<div class="muted forecast-empty">No history yet — click Load forecast sample.</div>';
  }

  const forecast = state.latest?.prediction?.forecast || {};
  const temporal = state.system?.temporal || {};
  const horizons = forecast.horizons || [];
  const byH = Object.fromEntries(horizons.map(h => [Number(h.horizon), h]));

  if ($('forecastPageNext')) $('forecastPageNext').textContent = safe(forecast.next_stage, '—');
  if ($('forecastPageMeta')) {
    $('forecastPageMeta').textContent = forecast.next_stage == null
      ? (recent.length < 4 ? 'Need T1–T4 — load forecast sample' : 'Waiting for confident forecast')
      : `${fmtPct(forecast.confidence)} confidence · ${safe(forecast.model_name, 'LSTM')}`;
  }
  if ($('forecastInfiltrationMeta')) {
    $('forecastInfiltrationMeta').textContent = forecast.infiltration_probability == null
      ? ''
      : `Infiltration ${fmtPct(forecast.infiltration_probability)}`;
  }
  if ($('forecastPageEvidence')) {
    $('forecastPageEvidence').textContent = safe(
      forecast.evidence,
      'Load forecast sample to generate T5–T7 from the lab attack sequence.'
    );
  }

  const modelReady = !!(state.system?.models?.world_model || temporal.ready || forecast.world_model_ready || forecast.model_name);
  if ($('forecastModelName')) {
    $('forecastModelName').textContent = modelReady ? (forecast.model_name || 'LSTM') : 'Not ready';
  }
  if ($('forecastModelMeta')) {
    $('forecastModelMeta').textContent = recent.length
      ? `${recent.length}/4 windows loaded`
      : 'Click Load forecast sample (10.0.2.15 lab path)';
  }

  [1, 2, 3].forEach((h) => {
    const item = byH[h];
    const label = `forecastT${h + 4}`;
    const meta = `forecastT${h + 4}Meta`;
    if ($(label)) $(label).textContent = item && item.accepted !== false ? item.stage : '—';
    if ($(meta)) {
      $(meta).textContent = item && item.accepted !== false
        ? `${fmtPct(item.confidence)} confidence`
        : `+${h} window${h === 1 ? '' : 's'}`;
    }
  });

  const driving = forecast.driving_features || [];
  if ($('forecastDrivingFeatures')) {
    $('forecastDrivingFeatures').innerHTML = driving.length
      ? driving.slice(0, 5).map(d => `<div class="key-value-row"><span>${safe(d.label, d.feature)}</span><strong>${fmtPct(d.attention_weight)}</strong></div>`).join('')
      : '<div class="muted">Drivers appear after the sample loads.</div>';
  }
}

function renderDevices() {
  const data = state.devices || {items:[]};
  const items = data.items || [];
  $('deviceTotal').textContent = Number(data.total || items.length).toLocaleString();
  $('deviceNew').textContent = Number(data.new || 0).toLocaleString();
  $('deviceElevated').textContent = Number(data.elevated || 0).toLocaleString();
  $('deviceLatest').textContent = items.length ? safe(items[0].ip || items[0].mac) : 'None';
  $('devicesTable').innerHTML = items.length ? items.map(d => `<tr>
    <td><strong>${safe(d.ip)}</strong><br><span class="muted tiny">${safe(d.mac, 'MAC unavailable')}</span></td>
    <td><span class="badge ${d.status === 'NEW' ? 'medium' : 'low'}">${safe(d.status)}</span></td>
    <td>${fmtTime(d.first_seen)}</td>
    <td>${fmtTime(d.last_seen)}</td>
    <td><span class="badge ${stageClass(d.current_stage)}">${safe(d.current_stage)}</span></td>
    <td><span class="badge ${riskClass(d.risk_level)}">${safe(d.risk_level)}</span></td>
    <td>${(d.ports || []).slice(-8).join(', ') || '--'}</td>
    <td>${(d.destinations || []).slice(-4).join(', ') || '--'}</td>
  </tr>`).join('') : '<tr><td colspan="8" class="muted">No devices observed yet.</td></tr>';
  $('devicesEmpty').classList.toggle('hidden', items.length > 0);
}

async function refreshLive() {
  try {
    const [latest, history, evidence, devices, forecastEval] = await Promise.all([
      request('/api/latest'),
      request('/api/history?limit=80'),
      request('/api/evidence?limit=40'),
      request('/api/devices'),
      request('/api/forecast/evaluation'),
    ]);
    setBackendStatus(true);
    state.latest = latest;
    state.history = history.items || [];
    state.evidence = evidence;
    state.devices = devices;
    state.forecastEval = forecastEval;
    $('lastSync').textContent = new Date().toLocaleTimeString();
    renderOverview();
    if (state.activePage === 'alerts') renderAlerts();
    if (state.activePage === 'devices') renderDevices();
    if (state.activePage === 'forecast') renderForecastPage();
  } catch (err) {
    setBackendStatus(false);
    $('sidebarSensorText').textContent = 'Backend offline';
    $('sidebarSensorDot').classList.remove('online');
  }
}

async function loadArena() {
  const box = $('arenaBody');
  if (!box) return;
  try {
    const arena = await request('/api/models/arena');
    if (!arena.available) { box.innerHTML = `<div class="muted tiny">${escapeArena(arena.hint || arena.error || 'Leaderboard not generated yet')}</div>`; return; }
    const titles = { detection: 'Detection — Normal vs Attack', stage: 'Attack category (stage)' };
    box.innerHTML = Object.values(arena.tasks).map((t) => {
      const metric = t.ranking_metric;
      const rows = t.leaderboard.map((e) => `<tr class="${e.key === t.winner ? 'winner' : ''}">
          <td title="${escapeArena(e.model)}">${e.key === t.winner ? '🏆 ' : ''}${escapeArena(ARENA_SHORT[e.key] || e.model)}</td>
          <td>${(e[metric] * 100).toFixed(2)}%</td><td>${(e.accuracy * 100).toFixed(2)}%</td>
          <td>${e.size_mb} MB</td></tr>`).join('');
      const maxImp = Math.max(...t.feature_importance.map((f) => f.importance), 1e-9);
      const feats = t.feature_importance.slice(0, 8).map((f) => `<div class="arena-feat"><span>${escapeArena(f.feature)}</span>
          <span class="track"><span class="fill" style="display:block;width:${Math.max(0, f.importance / maxImp * 100)}%"></span></span>
          <span class="muted">${f.importance.toFixed(3)}</span></div>`).join('');
      const delta = t.improvement_vs_reference;
      return `<div class="arena-task">
        <h4>${titles[t.task] || t.task}</h4>
        <span class="muted tiny">Ranked by ${metric.replace('_', '-')} · ${t.validation_rows.toLocaleString()} validation rows ·
          <span class="badge ${delta > 0 ? 'low' : 'medium'}">${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(2)} pts vs v7 model</span></span>
        <table class="arena-table"><thead><tr><th>Model</th><th>${metric.replace('_', '-')}</th><th>Accuracy</th><th>Size</th></tr></thead><tbody>${rows}</tbody></table>
        <div class="muted tiny">Top drivers (permutation importance)</div>${feats}
      </div>`;
    }).join('');
  } catch (err) {
    box.innerHTML = `<div class="muted tiny">Leaderboard unavailable: ${escapeArena(err.message)}</div>`;
  }
}

const ARENA_SHORT = {
  hist_gradient_boost: 'Gradient Boosting (HGB)',
  random_forest: 'Random Forest (v7 reference)',
  logistic_regression: 'Logistic Regression',
};

function escapeArena(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

async function loadSystem() {
  try {
    state.system = await request('/api/system');
    const models = state.system.models || {};
    const prefix = [
      ['Current-state detector', models.detection && models.stage, 'Binary + attack category models (selected by AI model arena)'],
      ['Normal anomaly model', models.anomaly, 'Isolation Forest normal baseline'],
      ['Behaviour signatures', true, 'Port/SYN/RST/rate/volume rules'],
      ['Device intelligence', true, `${Number(state.system?.devices || 0)} observed source devices`],
      ['T1–T4 history', true, 'Chronological state buffer from demo / offline windows'],
    ];
    const cards = [
      ['LSTM world model', models.world_model, models.world_model ? 'P(S_t+1|history) + K-step infiltration rollout' : 'Train via Forecasting page', models.world_model ? 'READY' : 'NOT TRAINED'],
      ['Logistic regression baseline', !!(state.system?.metrics?.baseline || state.system?.temporal?.baseline_ready), 'Same-feature baseline for F1 comparison'],
      ['Sequence RF ablation', models.temporal_ml, 'Optional non-recurrent T1–T4 classifier', models.temporal_ml ? 'READY' : 'OPTIONAL'],
      ['T5/T6/T7 horizons', true, 'Multi-horizon future-behaviour forecasts with confidence gating'],
      ['Attention explainability', true, 'SHAP-style driving flags/ports/flow stats'],
      ['MITRE interpretation', true, 'Inferred behaviour/tactic context kept separate from dataset labels'],
      ['PCAP + CSV offline', true, 'Dual flow/packet feature extraction'],
      ['Demo traffic seed', true, 'Curated attack progression without a live gateway'],
      ['Tamper-evident evidence chain', state.system?.evidence_chain?.valid !== false, 'SHA-256 event + previous-hash linkage; not a distributed-consensus ledger'],
      ['Web console', true, 'FastAPI + responsive browser UI'],
    ];
    const allCards = [...prefix, ...cards];
    $('systemGrid').innerHTML = allCards.map((row) => {
      const [name, ready, note, label] = row;
      const badge = label || (ready ? 'READY' : 'NOT READY');
      const tone = ready || badge === 'OPTIONAL' ? (badge === 'OPTIONAL' ? 'medium' : 'low') : 'high';
      return `<div class="system-card"><div class="eyebrow">COMPONENT</div><strong>${name}</strong><div class="status-line"><span class="muted tiny">${note}</span><span class="badge ${tone}">${badge}</span></div></div>`;
    }).join('');
    renderForecastPage();
    loadArena();
  } catch (err) {
    toast(`System status error: ${err.message}`);
  }
}

$('clearHistoryButton').addEventListener('click', async () => {
  try { await request('/api/history', {method:'DELETE'}); state.history=[]; state.latest=null; renderOverview(); toast('Live history cleared'); }
  catch (err) { toast(err.message); }
});

async function seedDemoSession(mode = 'attack', { stay = true } = {}) {
  const result = await request(`/api/demo/seed?mode=${encodeURIComponent(mode)}`, { method: 'POST' });
  await loadSystem();
  await refreshLive();
  if (state.activePage === 'forecast') renderForecastPage();
  if (state.activePage === 'overview') renderOverview();
  const fc = result.forecast || {};
  if (mode === 'normal') {
    toast(`Normal demo · stage ${result.latest_stage || 'Normal'} · LOW baseline`);
    showDemoReadyBanner('Normal/no-attack sample loaded. Overview should show Normal / LOW risk.');
  } else {
    toast(
      result.world_model_ready
        ? `Attack demo ready · next: ${fc.next_stage || fc.status || 'active'}`
        : `Attack demo loaded · train world model if needed`
    );
    showDemoReadyBanner('Attack sample loaded. Open Forecast for T5–T7.');
  }
  return result;
}

$('seedDemoButton') && $('seedDemoButton').addEventListener('click', async (e) => {
  e.preventDefault();
  e.stopPropagation();
  const btn = $('seedDemoButton');
  btn.disabled = true; btn.textContent = 'Loading…';
  try {
    await seedDemoSession('attack');
  } catch (err) {
    toast(`Demo seed failed: ${err.message}`);
  } finally {
    btn.disabled = false; btn.textContent = '▶ Attack demo';
  }
});

$('seedNormalButton')?.addEventListener('click', async (e) => {
  e.preventDefault();
  e.stopPropagation();
  const btn = $('seedNormalButton');
  btn.disabled = true; btn.textContent = 'Loading…';
  try {
    await seedDemoSession('normal');
  } catch (err) {
    toast(`Normal demo failed: ${err.message}`);
  } finally {
    btn.disabled = false; btn.textContent = 'Normal demo';
  }
});


$('verifyChainButton').addEventListener('click', async () => {
  try {
    const verification = await request('/api/evidence/verify');
    state.evidence.verification = verification;
    renderEvidenceSummary();
    renderEvidencePage();
    toast(verification.valid ? `Evidence chain verified: ${verification.blocks} block(s)` : `Evidence verification failed at block ${verification.failed_index}`);
  } catch (err) {
    toast(`Verification failed: ${err.message}`);
  }
});

$('trainTemporalButton').addEventListener('click', async () => {
  const btn = $('trainTemporalButton');
  btn.disabled = true; btn.textContent = 'Training…';
  try {
    const result = await request('/api/temporal/train', {method:'POST'});
    toast(result.ready ? 'LSTM world model trained/refreshed successfully' : 'World-model training needs more chronological evidence');
    await loadSystem();
    await refreshLive();
  } catch (err) { toast(`World-model training: ${err.message}`); }
  finally { btn.disabled = false; btn.textContent = 'Train model'; }
});

$('forecastSampleButton')?.addEventListener('click', async (e) => {
  e.preventDefault();
  e.stopPropagation();
  const btn = $('forecastSampleButton');
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    const result = await seedDemoSession('attack');
    const fc = result.forecast || {};
    toast(`Attack sample · T5: ${fc.next_stage || fc.status || 'ready'}`);
    renderForecastPage();
    window.scrollTo(0, 0);
  } catch (err) {
    toast(`Attack sample failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Attack sample';
  }
});

$('forecastNormalButton')?.addEventListener('click', async (e) => {
  e.preventDefault();
  e.stopPropagation();
  const btn = $('forecastNormalButton');
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    await seedDemoSession('normal');
    renderForecastPage();
    window.scrollTo(0, 0);
  } catch (err) {
    toast(`Normal sample failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Normal sample';
  }
});

$('clearDevicesButton').addEventListener('click', async () => {
  try { await request('/api/devices', {method:'DELETE'}); state.devices={items:[],total:0,new:0,elevated:0}; renderDevices(); toast('Device registry cleared'); }
  catch (err) { toast(err.message); }
});

const uploadZone = $('uploadZone');
const fileInput = $('csvFile');
$('chooseFileButton').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
['dragenter','dragover'].forEach(evt => uploadZone.addEventListener(evt, e => {e.preventDefault(); uploadZone.classList.add('dragover');}));
['dragleave','drop'].forEach(evt => uploadZone.addEventListener(evt, e => {e.preventDefault(); uploadZone.classList.remove('dragover');}));
uploadZone.addEventListener('drop', e => selectFile(e.dataTransfer.files[0]));
function selectFile(file) {
  if (!file) return;
  const name = file.name.toLowerCase();
  const ok = name.endsWith('.csv') || name.endsWith('.pcap') || name.endsWith('.pcapng') || name.endsWith('.cap');
  if (!ok) { toast('Please choose a CSV or PCAP file.'); return; }
  state.selectedFile = file;
  $('fileName').textContent = `${file.name} · ${(file.size/1024/1024).toFixed(2)} MB`;
  $('analyzeFileButton').disabled = false;
  if ($('analyzeSampleButton')) $('analyzeSampleButton').disabled = false;
}

async function runOfflineAnalysis({ mode = 'all', maxRows = 0, batchSize = 2000 } = {}) {
  if (!state.selectedFile) return;
  const btn = $('analyzeFileButton');
  const sampleBtn = $('analyzeSampleButton');
  const cancelBtn = $('offlineCancelButton');
  const name = state.selectedFile.name.toLowerCase();
  const isPcap = name.endsWith('.pcap') || name.endsWith('.pcapng') || name.endsWith('.cap');
  btn.disabled = true;
  if (sampleBtn) sampleBtn.disabled = true;
  btn.textContent = isPcap ? 'Analysing…' : (mode === 'all' ? 'Batch scanning…' : 'Sampling…');
  state.offlineAbort = null;

  try {
    if (isPcap) {
      const form = new FormData(); form.append('file', state.selectedFile);
      const data = await request('/api/offline/analyze-pcap?window_seconds=5', {method:'POST', body:form});
      $('offlineResults').classList.remove('hidden');
      $('offlineProgress')?.classList.add('hidden');
      $('offlineTotal').textContent = Number(data.windows || 0).toLocaleString();
      $('offlineAnalysed').textContent = Number(data.windows || 0).toLocaleString();
      const infil = data.forecast?.infiltration_probability;
      $('offlineAnomaly').textContent = infil == null ? 'N/A' : fmtPct(infil);
      $('offlineAccuracy').textContent = safe(data.forecast?.next_stage, 'No forecast');
      if ($('offlineAccuracyFoot')) $('offlineAccuracyFoot').textContent = safe(data.forecast?.model_name, 'World-model forecast stage');
      if ($('offlineBinaryAcc')) $('offlineBinaryAcc').textContent = 'N/A';
      $('offlineForecastSummary').textContent = data.forecast?.evidence || data.note || 'PCAP analysed.';
      const driving = data.forecast?.driving_features || [];
      $('offlineDrivingFeatures').innerHTML = driving.length
        ? driving.map(d => `<div class="key-value-row"><span>${safe(d.label, d.feature)}</span><strong>${fmtPct(d.attention_weight)}</strong></div>`).join('')
        : '<div class="muted">Train the world model for driving-feature attribution.</div>';
      renderBars('offlineStageBars', data.stage_distribution || {});
      renderBars('offlineRiskBars', data.risk_distribution || {});
      toast(`Parsed ${Number(data.windows||0)} PCAP windows`);
      return;
    }

    await runCsvBatchScan({ mode, maxRows, batchSize });
  } catch (err) {
    if (err.name === 'AbortError') toast('Batch scan stopped');
    else toast(`Analysis failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyse all';
    if (sampleBtn) {
      sampleBtn.disabled = !state.selectedFile;
      sampleBtn.textContent = 'Balanced sample';
    }
    if (cancelBtn) {
      cancelBtn.classList.add('hidden');
      cancelBtn.disabled = false;
    }
    state.offlineAbort = null;
  }
}

function applyOfflineCsvResult(data, { live = false } = {}) {
  $('offlineResults').classList.remove('hidden');
  $('offlineTotal').textContent = Number(data.total_rows || 0).toLocaleString();
  $('offlineAnalysed').textContent = Number(data.analyzed_rows || 0).toLocaleString();
  $('offlineAnomaly').textContent = `${Number(data.average_anomaly||0).toFixed(1)}%`;
  $('offlineAccuracy').textContent = data.sample_accuracy == null ? 'N/A' : fmtPct(data.sample_accuracy);
  if ($('offlineAccuracyFoot')) {
    const batchBit = data.batch_index
      ? ` · batch ${data.batch_index}/${data.batch_total}`
      : '';
    $('offlineAccuracyFoot').textContent = `Category match · ${safe(data.sample_mode, 'all')}${batchBit}`;
  }
  if ($('offlineBinaryAcc')) {
    $('offlineBinaryAcc').textContent = data.binary_accuracy == null ? 'N/A' : fmtPct(data.binary_accuracy);
  }
  const truth = data.truth_distribution || data.label_counts_in_sample || {};
  const truthLine = Object.keys(truth).length
    ? `Truth so far: ${Object.entries(truth).map(([k,v]) => `${k}=${v}`).join(', ')}.`
    : '';
  $('offlineForecastSummary').textContent = `${data.note || ''} ${truthLine} ${live ? 'Scanning…' : 'CSV detection complete. Upload PCAP for T5–T7 forecast.'}`.trim();
  $('offlineDrivingFeatures').innerHTML = '';
  renderBars('offlineStageBars', data.stage_distribution || {});
  renderBars('offlineRiskBars', data.risk_distribution || {});
}

function setOfflineProgress({ batchIndex = 0, batchTotal = 0, analyzed = 0, planned = 0, meta = '' } = {}) {
  const wrap = $('offlineProgress');
  if (!wrap) return;
  wrap.classList.remove('hidden');
  const pct = batchTotal > 0 ? Math.min(100, Math.round((batchIndex / batchTotal) * 100)) : 0;
  if ($('offlineBatchLabel')) $('offlineBatchLabel').textContent = `Batch ${batchIndex} / ${batchTotal}`;
  if ($('offlineBatchPct')) $('offlineBatchPct').textContent = `${pct}%`;
  if ($('offlineBatchBar')) $('offlineBatchBar').style.width = `${pct}%`;
  if ($('offlineBatchMeta')) {
    $('offlineBatchMeta').textContent = meta || `${Number(analyzed).toLocaleString()} / ${Number(planned).toLocaleString()} rows`;
  }
}

async function runCsvBatchScan({ mode = 'all', maxRows = 0, batchSize = 2000 } = {}) {
  const cancelBtn = $('offlineCancelButton');
  const controller = new AbortController();
  state.offlineAbort = controller;
  if (cancelBtn) {
    cancelBtn.classList.remove('hidden');
    cancelBtn.onclick = () => controller.abort();
  }

  const form = new FormData();
  form.append('file', state.selectedFile);
  const url = `/api/offline/analyze-csv-stream?mode=${encodeURIComponent(mode)}&max_rows=${maxRows}&batch_size=${batchSize}`;
  const response = await fetch(url, { method: 'POST', body: form, signal: controller.signal });
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let last = null;
  let planned = 0;

  $('offlineResults').classList.remove('hidden');
  setOfflineProgress({ batchIndex: 0, batchTotal: 0, analyzed: 0, planned: 0, meta: 'Starting batch scan…' });

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (!line.trim()) continue;
      let msg;
      try { msg = JSON.parse(line); } catch (_) { continue; }
      if (msg.event === 'error') throw new Error(msg.detail || 'Batch scan failed');
      if (msg.event === 'start') {
        planned = Number(msg.planned_rows || 0);
        setOfflineProgress({
          batchIndex: 0,
          batchTotal: Number(msg.batch_total || 0),
          analyzed: 0,
          planned,
          meta: `Preparing ${planned.toLocaleString()} rows in ${msg.batch_total} batches…`,
        });
        $('offlineTotal').textContent = Number(msg.total_rows || 0).toLocaleString();
        continue;
      }
      if (msg.event === 'batch') {
        last = msg;
        applyOfflineCsvResult(msg, { live: !msg.done });
        setOfflineProgress({
          batchIndex: Number(msg.batch_index || 0),
          batchTotal: Number(msg.batch_total || 0),
          analyzed: Number(msg.analyzed_rows || 0),
          planned: Number(msg.planned_rows || planned),
          meta: msg.note || '',
        });
        if (msg.done) {
          toast(`Batch scan complete · ${Number(msg.analyzed_rows||0).toLocaleString()} rows`);
        }
      }
    }
  }
  if (buffer.trim()) {
    try {
      const msg = JSON.parse(buffer);
      if (msg.event === 'batch') {
        last = msg;
        applyOfflineCsvResult(msg, { live: false });
      }
    } catch (_) {}
  }
  if (!last) throw new Error('No batch results received');
  applyOfflineCsvResult(last, { live: false });
  if ($('offlineBatchMeta')) $('offlineBatchMeta').textContent = `Complete · ${Number(last.analyzed_rows||0).toLocaleString()} rows`;
}

$('analyzeFileButton').addEventListener('click', () => runOfflineAnalysis({ mode: 'all', maxRows: 0, batchSize: 2000 }));
$('analyzeSampleButton')?.addEventListener('click', () => runOfflineAnalysis({ mode: 'stratified', maxRows: 5000, batchSize: 1000 }));

function renderBars(id, obj) {
  const entries = Object.entries(obj).sort((a,b)=>b[1]-a[1]);
  const max = Math.max(...entries.map(x=>x[1]),1);
  $(id).innerHTML = entries.length ? entries.map(([k,v]) => `<div class="bar-row"><strong>${k}</strong><div class="bar-track"><div class="bar-fill" style="width:${(v/max*100).toFixed(1)}%"></div></div><span class="muted">${v}</span></div>`).join('') : '<div class="muted">No results</div>';
}

/* —— Simulation: concrete worked example —— */
const simExampleWindows = [
  { stage: 'Normal', detail: 'web browse · 42 pkt/s' },
  { stage: 'Reconnaissance', detail: '48 dst ports · high SYN' },
  { stage: 'Reconnaissance', detail: 'scan continues · SSH focus' },
  { stage: 'Initial Access', detail: 'auth attempts · risk↑' },
];

const simSteps = [
  {
    title: 'Capture the session',
    phase: 'CAPTURE',
    narration: 'Project start: ChronoFlux ingests a chronological traffic session (demo seed, CSV, or PCAP). Example host 10.0.2.15 → gateway 10.0.2.1. Metadata only — no payload inspection.',
    bullets: [
      'Problem: detect now AND forecast next attack stage',
      'Input path: Run demo / Offline CSV / Offline PCAP',
      'Source 10.0.2.15 → destination 10.0.2.1 (L3/L4 only)',
      'Pillars: Detect → Forecast → Observe → Verify',
    ],
    checks: [0, 9],
    stage: 'Normal',
    anomaly: 8, risk: 12, infil: 5,
    windows: 1,
    flow: { srcPort: ':49152', proto: 'TCP', flags: 'ACK', dstPort: ':443' },
    forecast: ['—', '—', '—'],
  },
  {
    title: 'Build dual features',
    phase: 'FEATURES',
    narration: 'For this window ChronoFlux builds a dual state vector: flow stats + packet timing/flags. Example: unique_dst_ports jumps from 3 → 48.',
    bullets: [
      'Flow: ports=48, outbound bytes↑, bi-dir ratio skewed',
      'Packet: IAT variance↑, SYN count↑, TTL stable',
      'Port-scan score rises (many destinations/ports)',
      'No deep packet inspection — metadata only',
    ],
    checks: [1],
    stage: 'Normal',
    anomaly: 22, risk: 24, infil: 10,
    windows: 1,
    flow: { srcPort: ':49152', proto: 'TCP', flags: 'SYN↑', dstPort: ':1–1024' },
    forecast: ['—', '—', '—'],
  },
  {
    title: 'Hybrid detect = Recon',
    phase: 'DETECT',
    narration: 'Hybrid engine flags current stage as Reconnaissance. Example risk mix: ML 62 · Anomaly 55 · Signature 70 → fused risk ≈ 61.',
    bullets: [
      'ML classifier → Reconnaissance / scan-like category',
      'Isolation Forest anomaly ≈ 55–65%',
      'Signature: high SYN + many unique destination ports',
      'Risk = 0.50 ML + 0.30 IF + 0.20 Rules',
    ],
    checks: [2],
    stage: 'Reconnaissance',
    anomaly: 58, risk: 61, infil: 28,
    windows: 2,
    flow: { srcPort: ':49152', proto: 'TCP', flags: 'SYN flood', dstPort: ':22 / many' },
    forecast: ['—', '—', '—'],
  },
  {
    title: 'Fill T1–T4 history',
    phase: 'HISTORY',
    narration: 'ChronoFlux keeps four chronological windows for 10.0.2.15. Example buffer: Normal → Recon → Recon → Initial Access.',
    bullets: [
      'T1 Normal — ordinary HTTPS browsing',
      'T2 Recon — port sweep begins',
      'T3 Recon — scan concentrates on SSH',
      'T4 Initial Access — login-style attempts appear',
    ],
    checks: [3],
    stage: 'Initial Access',
    anomaly: 66, risk: 70, infil: 42,
    windows: 4,
    flow: { srcPort: ':49188', proto: 'TCP', flags: 'PSH/ACK', dstPort: ':22' },
    forecast: ['—', '—', '—'],
  },
  {
    title: 'LSTM world model',
    phase: 'WORLD MODEL',
    narration: 'The LSTM reads T1–T4 and estimates P(next stage | history). Example: after Recon→Access, probability mass shifts toward Execution / C2.',
    bullets: [
      'Input: 4 dual state vectors from 10.0.2.15',
      'Learns transition dynamics, not one-shot classification',
      'Beats logistic regression on same holdout features',
      'Attention focuses on SYN count, ports, byte ratio',
    ],
    checks: [4, 7],
    stage: 'Initial Access',
    anomaly: 70, risk: 74, infil: 55,
    windows: 4,
    flow: { srcPort: ':49188', proto: 'TCP', flags: 'PSH/ACK', dstPort: ':22' },
    forecast: ['Execution?', 'C2?', 'Exfil?'],
  },
  {
    title: 'Forecast T5 → T6 → T7',
    phase: 'FORECAST',
    narration: 'K-step rollout for this host: T5 Execution → T6 Command & Control → T7 Exfiltration. Infiltration probability rises above 70%.',
    bullets: [
      'T5 (next window): Execution',
      'T6 (+2): Command & Control',
      'T7 (+3): Exfiltration',
      'Defender can act before data leaves the lab',
    ],
    checks: [5],
    stage: 'Execution',
    anomaly: 78, risk: 84, infil: 76,
    windows: 4,
    flow: { srcPort: ':49201', proto: 'TCP', flags: 'PSH', dstPort: ':4444' },
    forecast: ['Execution', 'Command & Control', 'Exfiltration'],
  },
  {
    title: 'Explain why + MITRE',
    phase: 'EXPLAIN',
    narration: 'Example drivers: unique_dst_ports, syn_count, outbound_bytes, iat_variance. MITRE map: Recon → Initial Access → Execution → C2 → Exfiltration.',
    bullets: [
      'Top drivers: port-scan score, SYN count, outbound bytes',
      'MITRE: Discovery / Initial Access / Execution / C2',
      'Why: scan then auth then beacon-like traffic',
      'Action: isolate 10.0.2.15, block :4444, review SSH logs',
    ],
    checks: [6],
    stage: 'Command & Control',
    anomaly: 84, risk: 89, infil: 86,
    windows: 4,
    flow: { srcPort: ':49201', proto: 'TCP', flags: 'beacon', dstPort: ':4444' },
    forecast: ['Execution', 'Command & Control', 'Exfiltration'],
  },
  {
    title: 'Commit evidence block',
    phase: 'VERIFY',
    narration: 'Because risk is elevated, ChronoFlux writes a SHA-256 evidence block for this event and links it to the previous hash — tamper-evident record.',
    bullets: [
      'Event: HIGH risk · stage Exfiltration forecast path',
      'Hash links to previous block (private chain)',
      'Verify later on Evidence page',
      'Not a cryptocurrency / public blockchain',
    ],
    checks: [8],
    stage: 'Exfiltration',
    anomaly: 90, risk: 94, infil: 92,
    windows: 4,
    flow: { srcPort: ':49220', proto: 'TCP', flags: 'bulk out', dstPort: ':443' },
    forecast: ['Execution', 'Command & Control', 'Exfiltration'],
  },
];

const simState = { step: -1, timer: null, playing: false, covered: new Set() };

function setSimMeters(anomaly, risk, infil) {
  if ($('simAnomalyMeter')) $('simAnomalyMeter').style.width = `${anomaly}%`;
  if ($('simRiskMeter')) $('simRiskMeter').style.width = `${risk}%`;
  if ($('simInfilMeter')) $('simInfilMeter').style.width = `${infil}%`;
  if ($('simAnomalyText')) $('simAnomalyText').textContent = `${anomaly}%`;
  if ($('simRiskText')) $('simRiskText').textContent = `${risk}`;
  if ($('simInfilText')) $('simInfilText').textContent = `${infil}%`;
}

function renderSimChecklist() {
  const labels = document.querySelectorAll('#simChecklist label');
  labels.forEach((label) => {
    const input = label.querySelector('input');
    const idx = Number(input.dataset.check);
    const on = simState.covered.has(idx);
    input.checked = on;
    label.classList.toggle('on', on);
  });
  if ($('simCoverageChip')) $('simCoverageChip').textContent = `${simState.covered.size} / 10`;
}

function renderSimExample(step) {
  const count = Number(step.windows || 0);
  document.querySelectorAll('#simWindowGrid .sim-win').forEach((el) => {
    const w = Number(el.dataset.w);
    const filled = w <= count;
    el.classList.toggle('on', filled);
    el.classList.toggle('current', filled && w === count);
    if (!filled) {
      el.querySelector('strong').textContent = '—';
      el.querySelector('small').textContent = 'waiting';
      return;
    }
    const win = simExampleWindows[w - 1];
    el.querySelector('strong').textContent = win.stage;
    el.querySelector('small').textContent = win.detail;
  });

  const fc = step.forecast || ['—', '—', '—'];
  if ($('simT5')) $('simT5').textContent = fc[0];
  if ($('simT6')) $('simT6').textContent = fc[1];
  if ($('simT7')) $('simT7').textContent = fc[2];
  document.querySelectorAll('#simForecastRow .sim-fc').forEach((el, i) => {
    el.classList.toggle('on', fc[i] && fc[i] !== '—' && !String(fc[i]).endsWith('?'));
    el.classList.toggle('soft', String(fc[i] || '').endsWith('?'));
  });

  const flow = step.flow || {};
  if ($('simSrcPort')) $('simSrcPort').textContent = flow.srcPort || ':—';
  if ($('simProto')) $('simProto').textContent = flow.proto || 'TCP';
  if ($('simFlags')) $('simFlags').textContent = flow.flags || '—';
  if ($('simDstPort')) $('simDstPort').textContent = flow.dstPort || ':—';
}

function applySimStep(index) {
  const step = simSteps[index];
  if (!step) return;
  simState.step = index;
  $('simStepIndex').textContent = `Step ${index + 1} / ${simSteps.length}`;
  $('simStepTitle').textContent = step.title;
  $('simNarration').textContent = step.narration;
  $('simPhaseChip').textContent = step.phase;
  $('simProgressBar').style.width = `${((index + 1) / simSteps.length) * 100}%`;
  $('simBullets').innerHTML = step.bullets.map((b, i) => `<li class="${i === 0 ? 'flash' : ''}">${b}</li>`).join('');

  document.querySelectorAll('.sim-node').forEach((node) => {
    const n = Number(node.dataset.sim);
    node.classList.toggle('active', n === index);
    node.classList.toggle('done', n < index);
  });

  // Scroll only inside the pipeline strip — never jump the whole page
  const pipeline = $('simPipeline');
  const activeNode = document.querySelector(`.sim-node[data-sim="${index}"]`);
  if (pipeline && activeNode) {
    const left = activeNode.offsetLeft - pipeline.clientWidth / 2 + activeNode.clientWidth / 2;
    pipeline.scrollTo({ left: Math.max(0, left), behavior: 'smooth' });
  }

  document.querySelectorAll('.sim-attack-step').forEach((el) => {
    const order = ['Normal', 'Reconnaissance', 'Initial Access', 'Execution', 'Command & Control', 'Exfiltration'];
    const cur = order.indexOf(step.stage);
    const mine = order.indexOf(el.dataset.stage);
    el.classList.toggle('on', mine <= cur);
    el.classList.toggle('hot', mine === cur && cur >= 4);
  });
  $('simAttackChip').textContent = step.stage.toUpperCase();
  setSimMeters(step.anomaly, step.risk, step.infil);
  renderSimExample(step);

  step.checks.forEach((c) => simState.covered.add(c));
  renderSimChecklist();
  updateSimNavButtons();

  document.querySelector('.sim-packets')?.classList.toggle('running', simState.playing);
}

function updateSimNavButtons() {
  if ($('simPrevButton')) $('simPrevButton').disabled = simState.step < 0;
  if ($('simNextButton')) {
    $('simNextButton').disabled = false;
    $('simNextButton').textContent = simState.step >= simSteps.length - 1
      ? 'Done — Run real demo'
      : 'Next step →';
  }
}

function stopSimulationTimer() {
  if (simState.timer) clearInterval(simState.timer);
  simState.timer = null;
  simState.playing = false;
  document.querySelector('.sim-packets')?.classList.remove('running');
  if ($('simPlayButton')) $('simPlayButton').textContent = '▶ Auto-play';
}

function advanceSimulation() {
  const next = simState.step + 1;
  if (next >= simSteps.length) {
    stopSimulationTimer();
    $('simNarration').textContent = 'Story complete. ChronoFlux Detect → Forecast → Explain → Verify on host 10.0.2.15. Click Run real demo next — you stay on this page.';
    $('simPhaseChip').textContent = 'COMPLETE';
    $('simHint').textContent = 'Run real demo loads the LSTM into memory. Then open Overview / Forecast yourself.';
    [0,1,2,3,4,5,6,7,8,9].forEach((i) => simState.covered.add(i));
    renderSimChecklist();
    updateSimNavButtons();
    toast('Example complete — Run real demo when ready');
    return;
  }
  applySimStep(next);
}

function playSimulation() {
  if (simState.playing) return;
  if (simState.step >= simSteps.length - 1) resetSimulation();
  if (simState.step < 0) applySimStep(0);
  if (simState.timer) clearInterval(simState.timer);
  simState.playing = true;
  document.querySelector('.sim-packets')?.classList.add('running');
  if ($('simPlayButton')) $('simPlayButton').textContent = 'Playing…';
  simState.timer = setInterval(advanceSimulation, 3800);
}

function resetSimulation() {
  stopSimulationTimer();
  simState.step = -1;
  simState.covered = new Set();
  renderSimChecklist();
  $('simStepIndex').textContent = `Step 0 / ${simSteps.length}`;
  $('simStepTitle').textContent = 'Ready — press Next step';
  $('simNarration').textContent = 'Use Next step to move one stage at a time. Pipeline nodes are also clickable. Nothing will jump to another page while you learn.';
  $('simPhaseChip').textContent = 'IDLE';
  $('simProgressBar').style.width = '0%';
  $('simBullets').innerHTML = '<li>Press <strong>Next step</strong> to start. Each click stays on Simulation.</li>';
  $('simAttackChip').textContent = 'NORMAL';
  setSimMeters(0, 0, 0);
  document.querySelectorAll('.sim-node').forEach((n) => n.classList.remove('active', 'done'));
  document.querySelectorAll('.sim-attack-step').forEach((el) => el.classList.remove('on', 'hot'));
  if ($('simT5')) $('simT5').textContent = '—';
  if ($('simT6')) $('simT6').textContent = '—';
  if ($('simT7')) $('simT7').textContent = '—';
  document.querySelectorAll('#simForecastRow .sim-fc').forEach((el) => el.classList.remove('on', 'soft'));
  document.querySelectorAll('#simWindowGrid .sim-win').forEach((el) => {
    el.classList.remove('on', 'current');
    el.querySelector('strong').textContent = '—';
    el.querySelector('small').textContent = 'waiting';
  });
  if ($('simSrcPort')) $('simSrcPort').textContent = ':49152';
  if ($('simProto')) $('simProto').textContent = 'TCP';
  if ($('simFlags')) $('simFlags').textContent = 'SYN flood';
  if ($('simDstPort')) $('simDstPort').textContent = ':22 / many';
  $('simHint').textContent = 'After step 8, click Run real demo. You choose when to open Overview / Forecast.';
  updateSimNavButtons();
}

function startSimulationPage() {
  if (simState.step < 0) resetSimulation();
  else {
    applySimStep(simState.step);
    stopSimulationTimer();
  }
}

$('simPrevButton')?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  stopSimulationTimer();
  if (simState.step <= 0) {
    resetSimulation();
    return;
  }
  applySimStep(simState.step - 1);
});

$('simNextButton')?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  stopSimulationTimer();
  if (simState.step >= simSteps.length - 1) {
    toast('Story complete — click Run real demo');
    return;
  }
  if (simState.step < 0) applySimStep(0);
  else advanceSimulation();
});

$('simPlayButton')?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  if (simState.playing) {
    stopSimulationTimer();
    toast('Auto-play paused');
    return;
  }
  if (simState.step >= simSteps.length - 1) resetSimulation();
  playSimulation();
});

$('simPauseButton')?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  stopSimulationTimer();
  toast('Paused — use Next / Prev to continue');
});

$('simResetButton')?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  resetSimulation();
  toast('Simulation restarted');
});

document.querySelectorAll('.sim-node').forEach((node) => {
  node.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    stopSimulationTimer();
    const idx = Number(node.dataset.sim);
    if (Number.isFinite(idx)) applySimStep(idx);
  });
});

$('simSeedButton')?.addEventListener('click', async (e) => {
  e.preventDefault();
  e.stopPropagation();
  const btn = $('simSeedButton');
  btn.disabled = true;
  btn.textContent = 'Seeding…';
  try {
    const result = await request('/api/demo/seed', { method: 'POST' });
    toast(`Real demo seeded (${result.windows} windows) — stay on Simulation`);
    await loadSystem();
    await refreshLive();
    [0,1,2,3,4,5,6,7,8,9].forEach((i) => simState.covered.add(i));
    renderSimChecklist();
    $('simDemoBanner')?.classList.remove('hidden');
    showDemoReadyBanner('Real demo loaded. Open Overview or Forecast when you are ready.');
  } catch (err) {
    toast(`Demo failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run real demo';
  }
});

window.addEventListener('resize', renderCharts);
loadSystem();
loadEvidence();
refreshLive();
setInterval(refreshLive, 2500);
