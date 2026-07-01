/* LoRA Maker - Main App */

const API = '';
const WS_BASE = `ws://${location.host}`;

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  projects: [],
  currentProjectId: null,
  activeTab: 'progress',
  wizard: { step: 1, lora_type: null, name: '', trigger_word: '', base_model: '', gpu_mode: 'local', vastai_offer_id: null },
  ws: null,
  lossHistory: [],
  uploadedImages: [],
  checkpoints: [],
  vastaiInstances: [],
};

// ── DOM Refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Persistent state — initialize early so all functions can use safely ──
if (!window._genState) window._genState = { gallery: [], running: false, cancelled: false, currentGenId: null, queue: [], queueCounter: 0 };
if (!window._genForm)  window._genForm  = {
  baseModel: '', loraFile: '', prompt: 'masterpiece, best quality, portrait',
  neg: '', trigger: '', scheduler: 'dpm++_2m', steps: 20, cfg: 7,
  seed: -1, count: 1, loraScale: 1.0, denoising: 0.75,
  res_1024: true, res_832x1216: false, res_1216x832: false, res_custom: false,
  customW: 512, customH: 512, galleryHTML: null,
};
if (!window._valForm)  window._valForm  = {
  loraPath: '', baseModel: '', prompt: 'masterpiece, best quality, 1girl, portrait, detailed',
  neg: '', trigger: '', scheduler: 'dpm++_2m', steps: 20, cfg: 7,
  seed: -1, loraScale: 1.0, denoising: 0.75, resultsHTML: null,
};
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; return e; };

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadProjects();
  bindEvents();
  setInterval(refreshCurrentProject, 3000);
});

// ── API Helpers ─────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return r.json();
}

async function loadProjects() {
  try {
    state.projects = await api('GET', '/api/projects');
    renderProjectList();
  } catch (e) { toast('프로젝트 로드 실패', 'error'); }
}

async function refreshCurrentProject() {
  if (!state.currentProjectId) return;
  try {
    const p = await api('GET', `/api/projects/${state.currentProjectId}`);
    const idx = state.projects.findIndex(x => x.id === p.id);
    if (idx >= 0) state.projects[idx] = p;
    updateDetailHeader(p);
    renderSidebarItem(p);
  } catch (e) {}
}

// ── Project List ────────────────────────────────────────────────────────────
function renderProjectList() {
  const list = $('projectList');
  list.innerHTML = '';

  if (!state.projects.length) {
    list.innerHTML = `<div style="padding:24px 16px;text-align:center;color:var(--text-dim);font-size:12px;">
      프로젝트가 없습니다.<br>새 프로젝트를 만들어보세요.
    </div>`;
    showEmptyState();
    return;
  }

  state.projects.forEach(p => {
    list.appendChild(buildProjectItem(p));
  });

  // Show first project if none selected
  if (!state.currentProjectId && state.projects.length) {
    selectProject(state.projects[0].id);
  }
}

function buildProjectItem(p) {
  const item = el('div', 'project-item');
  item.dataset.id = p.id;
  const emoji = typeEmoji(p.lora_type);
  const bg = typeColor(p.lora_type);

  item.innerHTML = `
    <div class="project-item-icon" style="background:${bg}20">${emoji}</div>
    <div class="project-item-info">
      <div class="project-item-name">${escHtml(p.name)}</div>
      <div class="project-item-meta">${typeLabel(p.lora_type)} · ${p.image_count || 0}장</div>
    </div>
    <div class="status-dot ${p.status}"></div>
  `;
  item.onclick = () => selectProject(p.id);
  return item;
}

function renderSidebarItem(p) {
  const existing = document.querySelector(`.project-item[data-id="${p.id}"]`);
  if (existing) {
    const newItem = buildProjectItem(p);
    newItem.classList.toggle('active', p.id === state.currentProjectId);
    existing.replaceWith(newItem);
  }
}

async function selectProject(id) {
  _saveGenForm();
  _saveValForm();
  state.currentProjectId = id;
  state.lossHistory = [];
  state.checkpoints = [];

  // Update sidebar active
  document.querySelectorAll('.project-item').forEach(i => {
    i.classList.toggle('active', i.dataset.id === id);
  });

  const p = state.projects.find(x => x.id === id);
  if (!p) return;

  hideEmptyState();
  renderProjectDetail(p);
  connectWebSocket(id);
  switchTab('progress');

  // Load checkpoints
  await loadCheckpoints(id);
}

// ── Project Detail ──────────────────────────────────────────────────────────
function renderProjectDetail(p) {
  const main = $('mainContent');
  main.innerHTML = `
    <div class="project-detail">
      <div class="detail-header">
        <div class="detail-header-top">
          <div class="detail-title-area">
            <div class="detail-title">
              <span>${typeEmoji(p.lora_type)}</span>
              <span id="detailName">${escHtml(p.name)}</span>
              <span id="statusBadge" class="status-badge ${p.status}">${statusLabel(p.status)}</span>
            </div>
            <div class="detail-meta">
              <span>${typeLabel(p.lora_type)} LoRA</span>
              <span>·</span>
              <span>트리거: <strong>${escHtml(p.trigger_word)}</strong></span>
              <span>·</span>
              <span id="imageCountMeta">${p.image_count || 0}장</span>
              <span>·</span>
              <span>${p.gpu_mode === 'vastai' ? '☁️ Vast.ai' : '💻 로컬'}</span>
            </div>
          </div>
          <div class="detail-actions" id="detailActions"></div>
        </div>
        <div class="tabs">
          <button class="tab-btn active" onclick="switchTab('progress')" id="tab-progress">📊 학습 현황</button>
          <button class="tab-btn" onclick="switchTab('checkpoints')" id="tab-checkpoints">💾 체크포인트</button>
          <button class="tab-btn" onclick="switchTab('images')" id="tab-images">🖼 이미지</button>
          <button class="tab-btn" onclick="switchTab('settings')" id="tab-settings">⚙️ 설정</button>
        </div>
      </div>

      <div class="tab-content">
        <!-- Progress Tab -->
        <div class="tab-panel active" id="panel-progress">
          <div class="progress-card" id="trainingProgress">
            <div class="progress-card-title">학습 진행</div>
            <div class="progress-bar-wrap"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
            <div id="progressText" style="font-size:11px;color:var(--text-muted);margin-bottom:12px;">대기 중...</div>
            <div class="progress-stats">
              <div class="progress-stat">
                <div class="progress-stat-label">에폭</div>
                <div class="progress-stat-value" id="statEpoch">0 / ${p.total_epochs || '—'}</div>
              </div>
              <div class="progress-stat">
                <div class="progress-stat-label">스텝</div>
                <div class="progress-stat-value" id="statStep">0</div>
              </div>
              <div class="progress-stat">
                <div class="progress-stat-label">손실값</div>
                <div class="progress-stat-value" id="statLoss">—</div>
              </div>
              <div class="progress-stat">
                <div class="progress-stat-label">남은 시간</div>
                <div class="progress-stat-value" id="statEta">—</div>
              </div>
            </div>
          </div>

          <div class="chart-container">
            <div class="section-header">
              <div class="section-title">손실값 곡선</div>
            </div>
            <canvas id="lossChart"></canvas>
          </div>

          <div>
            <div class="section-header">
              <div class="section-title">학습 로그</div>
              <button class="btn btn-ghost btn-sm" onclick="clearLog()">지우기</button>
            </div>
            <div class="log-container" id="trainingLog">
              <div class="log-line" style="color:var(--text-dim)">로그가 여기에 표시됩니다...</div>
            </div>
          </div>
        </div>

        <!-- Checkpoints Tab -->
        <div class="tab-panel" id="panel-checkpoints">
          <div class="section-header">
            <div class="section-title">저장된 체크포인트</div>
            <button class="btn btn-ghost btn-sm" onclick="loadCheckpoints('${p.id}')">새로고침</button>
          </div>
          <div id="checkpointGrid" class="checkpoint-grid">
            <div class="empty-checkpoints">
              <div class="empty-checkpoints-icon">💾</div>
              <p>학습이 완료되면 에폭마다<br>체크포인트가 저장됩니다</p>
            </div>
          </div>
          <div id="checkpointDownloadAll" class="hidden" style="margin-top:8px;">
            <a href="/api/projects/${p.id}/checkpoints/final/download" class="btn btn-primary">
              ⬇️ 최종 LoRA 다운로드
            </a>
            <a href="/api/projects/${p.id}/checkpoints/best/download" class="btn btn-ghost" style="margin-left:8px;">
              ⭐ 최고 체크포인트 다운로드
            </a>
          </div>
        </div>

        <!-- Images Tab -->
        <div class="tab-panel" id="panel-images">
          <div id="imageUploadArea">
            <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
              <div class="upload-zone-icon">📁</div>
              <div class="upload-zone-title">이미지 추가</div>
              <div class="upload-zone-hint">클릭하거나 파일을 드래그하세요 (JPG, PNG, WEBP)</div>
            </div>
            <input type="file" id="fileInput" multiple accept=".jpg,.jpeg,.png,.webp" style="display:none" onchange="handleFileSelect(event)">
          </div>
          <div id="imageGrid" class="image-grid" style="margin-top:16px;"></div>
          <div id="imageCountInfo" style="margin-top:12px;font-size:12px;color:var(--text-muted);"></div>
        </div>

        <!-- Settings Tab -->
        <div class="tab-panel" id="panel-settings">
          <div id="settingsContent"></div>
        </div>
      </div>
    </div>
  `;

  updateDetailHeader(p);
  initLossChart();
  loadImages(p.id);
  loadSettings(p.id);
  setupUploadZone(p.id);
}

function updateDetailHeader(p) {
  const badge = $('statusBadge');
  if (badge) {
    badge.className = `status-badge ${p.status}`;
    badge.textContent = statusLabel(p.status);
  }
  const meta = $('imageCountMeta');
  if (meta) meta.textContent = `${p.image_count || 0}장`;

  renderDetailActions(p);

  // Update progress if we have data
  if (p.current_step) {
    const pct = p.total_steps ? Math.round(p.current_step / p.total_steps * 100) : 0;
    updateProgress({
      step: p.current_step, total_steps: p.total_steps,
      epoch: p.current_epoch, total_epochs: p.total_epochs,
      loss: p.current_loss, eta_seconds: p.eta_seconds,
    });
  }
}

function renderDetailActions(p) {
  const actions = $('detailActions');
  if (!actions) return;
  actions.innerHTML = '';

  const canStart = ['pending', 'cancelled', 'failed'].includes(p.status);
  const isRunning = ['running', 'preprocessing', 'captioning', 'training'].includes(p.status);
  const isDone = p.status === 'completed';

  if (canStart) {
    const btn = el('button', 'btn btn-success');
    btn.innerHTML = '▶ 학습 시작';
    btn.onclick = () => startTraining(p.id);
    actions.appendChild(btn);
  }
  if (isRunning) {
    const cancelBtn = el('button', 'btn btn-danger');
    cancelBtn.innerHTML = '⏹ 중단';
    cancelBtn.onclick = () => cancelTraining(p.id);
    actions.appendChild(cancelBtn);
  }
  if (isDone) {
    const a = document.createElement('a');
    a.href = `/api/projects/${p.id}/checkpoints/final/download`;
    a.className = 'btn btn-primary';
    a.innerHTML = '⬇️ LoRA 다운로드';
    actions.appendChild(a);
  }

  const delBtn = el('button', 'btn btn-ghost');
  delBtn.innerHTML = '🗑 삭제';
  delBtn.onclick = () => deleteProject(p.id);
  actions.appendChild(delBtn);
}

// ── Training Control ─────────────────────────────────────────────────────────
async function startTraining(id) {
  try {
    const result = await api('POST', `/api/projects/${id}/run`);
    toast(`학습 시작! ${result.image_count}장 × ${result.num_repeats}회 반복 = ${result.total_steps}스텝`, 'success');
    await loadProjects();
    switchTab('progress');
    addLog('학습 파이프라인 시작...', 'success');
  } catch (e) {
    toast(e.message, 'error');
  }
}

async function cancelTraining(id) {
  if (!confirm('학습을 중단하시겠습니까?')) return;
  try {
    await api('POST', `/api/projects/${id}/cancel`);
    toast('학습 중단 요청됨', 'info');
  } catch (e) { toast(e.message, 'error'); }
}

async function deleteProject(id) {
  if (!confirm('프로젝트를 삭제하시겠습니까? 모든 파일이 삭제됩니다.')) return;
  try {
    await api('DELETE', `/api/projects/${id}`);
    toast('프로젝트 삭제됨', 'info');
    state.currentProjectId = null;
    disconnectWebSocket();
    await loadProjects();
    if (!state.projects.length) showEmptyState();
  } catch (e) { toast(e.message, 'error'); }
}

// ── WebSocket ────────────────────────────────────────────────────────────────
function connectWebSocket(projectId) {
  disconnectWebSocket();
  const ws = new WebSocket(`${WS_BASE}/ws/${projectId}`);
  ws.onmessage = e => handleWsMessage(JSON.parse(e.data));
  ws.onerror = () => {};
  state.ws = ws;
}

function disconnectWebSocket() {
  if (state.ws) { try { state.ws.close(); } catch (e) {} state.ws = null; }
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'status':
      updateStatus(msg.status);
      break;
    case 'progress':
    case 'step':
      updateProgress(msg);
      break;
    case 'epoch_end':
      addLog(`에폭 ${msg.epoch}/${msg.total_epochs} 완료 - 평균 손실: ${(msg.avg_loss||0).toFixed(4)}`, 'success');
      break;
    case 'log':
      addLog(msg.message);
      break;
    case 'error':
      addLog(`오류: ${msg.message}`, 'error');
      toast(`학습 오류: ${msg.message}`, 'error');
      updateStatus('failed');
      break;
    case 'checkpoint_saved':
      const chk = msg.checkpoint;
      addLog(`💾 체크포인트 저장: 에폭 ${chk.epoch}`, 'success');
      addCheckpointCard(chk);
      break;
    case 'training_complete':
      toast('✅ 학습 완료!', 'success');
      updateStatus('completed');
      loadCheckpoints(state.currentProjectId);
      break;
  }
}

// ── Progress Updates ──────────────────────────────────────────────────────────
function updateStatus(status) {
  const badge = $('statusBadge');
  if (badge) {
    badge.className = `status-badge ${status}`;
    badge.textContent = statusLabel(status);
  }
  // Update sidebar dot
  const item = document.querySelector(`.project-item[data-id="${state.currentProjectId}"] .status-dot`);
  if (item) { item.className = `status-dot ${status}`; }

  // Update action buttons
  const p = state.projects.find(x => x.id === state.currentProjectId);
  if (p) { p.status = status; renderDetailActions(p); }
}

function updateProgress(data) {
  const step = data.step || 0;
  const total = data.total_steps || 1;
  const epoch = data.epoch || 0;
  const totalEpochs = data.total_epochs || 0;
  const loss = data.loss;
  const eta = data.eta_seconds;

  const pct = total ? Math.round(step / total * 100) : 0;
  const bar = $('progressBar');
  const txt = $('progressText');
  if (bar) bar.style.width = pct + '%';
  if (txt) txt.textContent = `${pct}% 완료 (스텝 ${step} / ${total})`;

  const se = $('statEpoch'); if (se) se.textContent = totalEpochs ? `${epoch} / ${totalEpochs}` : epoch;
  const ss = $('statStep'); if (ss) ss.textContent = step.toLocaleString();
  const sl = $('statLoss');
  if (sl && loss != null) {
    sl.textContent = loss.toFixed(4);
    state.lossHistory.push({ step, loss });
    updateLossChart();
  }
  const seta = $('statEta');
  if (seta && eta != null) seta.textContent = formatEta(eta);
}

// ── Loss Chart ─────────────────────────────────────────────────────────────
let lossChartInstance = null;

function initLossChart() {
  const canvas = $('lossChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  lossChartInstance = {
    ctx, data: [],
    draw() {
      const { ctx, data } = this;
      const W = canvas.width = canvas.offsetWidth * devicePixelRatio;
      const H = canvas.height = 180 * devicePixelRatio;
      ctx.clearRect(0, 0, W, H);

      if (data.length < 2) {
        ctx.fillStyle = '#484f58';
        ctx.font = `${12 * devicePixelRatio}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.fillText('학습 시작 후 그래프가 표시됩니다', W/2, H/2);
        return;
      }

      const pad = { t: 16*devicePixelRatio, r: 16*devicePixelRatio, b: 28*devicePixelRatio, l: 52*devicePixelRatio };
      const gW = W - pad.l - pad.r;
      const gH = H - pad.t - pad.b;

      const losses = data.map(d => d.loss);
      const minL = Math.min(...losses) * .95;
      const maxL = Math.max(...losses) * 1.05;
      const minS = data[0].step;
      const maxS = data[data.length-1].step;

      const sx = s => pad.l + (s - minS) / (maxS - minS) * gW;
      const sy = l => pad.t + gH - (l - minL) / (maxL - minL) * gH;

      // Grid
      ctx.strokeStyle = '#30363d';
      ctx.lineWidth = devicePixelRatio;
      for (let i = 0; i <= 4; i++) {
        const y = pad.t + gH * i / 4;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
        const val = maxL - (maxL - minL) * i / 4;
        ctx.fillStyle = '#484f58';
        ctx.font = `${10 * devicePixelRatio}px sans-serif`;
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(4), pad.l - 6*devicePixelRatio, y + 4*devicePixelRatio);
      }

      // Line
      const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + gH);
      grad.addColorStop(0, 'rgba(124,58,237,.3)');
      grad.addColorStop(1, 'rgba(124,58,237,0)');

      ctx.beginPath();
      ctx.moveTo(sx(data[0].step), sy(data[0].loss));
      for (let i = 1; i < data.length; i++) {
        ctx.lineTo(sx(data[i].step), sy(data[i].loss));
      }
      const lineEnd = ctx.currentX;
      ctx.strokeStyle = '#7c3aed';
      ctx.lineWidth = 2 * devicePixelRatio;
      ctx.stroke();

      // Fill under
      ctx.lineTo(sx(data[data.length-1].step), pad.t + gH);
      ctx.lineTo(sx(data[0].step), pad.t + gH);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }
  };
}

function updateLossChart() {
  if (!lossChartInstance) return;
  lossChartInstance.data = state.lossHistory.slice(-300);
  lossChartInstance.draw();
}

// ── Log ───────────────────────────────────────────────────────────────────
function addLog(msg, type = '') {
  const log = $('trainingLog');
  if (!log) return;
  const line = el('div', `log-line ${type}`);
  const time = new Date().toLocaleTimeString('ko', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  line.textContent = `[${time}] ${msg}`;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;

  // Trim old lines
  while (log.children.length > 500) log.removeChild(log.firstChild);
}

function clearLog() {
  const log = $('trainingLog');
  if (log) log.innerHTML = '<div class="log-line" style="color:var(--text-dim)">로그가 지워졌습니다.</div>';
}

// ── Checkpoints ────────────────────────────────────────────────────────────
async function loadCheckpoints(projectId) {
  try {
    state.checkpoints = await api('GET', `/api/projects/${projectId}/checkpoints`);
    renderCheckpoints();
  } catch (e) {}
}

function renderCheckpoints() {
  const grid = $('checkpointGrid');
  if (!grid) return;

  if (!state.checkpoints.length) {
    grid.innerHTML = `<div class="empty-checkpoints">
      <div class="empty-checkpoints-icon">💾</div>
      <p>학습이 완료되면 에폭마다<br>체크포인트가 저장됩니다</p>
    </div>`;
    return;
  }

  grid.innerHTML = '';
  state.checkpoints.forEach(chk => grid.appendChild(buildCheckpointCard(chk)));

  const dl = $('checkpointDownloadAll');
  if (dl) dl.classList.remove('hidden');
}

function buildCheckpointCard(chk) {
  const card = el('div', `checkpoint-card${chk.is_best ? ' best' : ''}`);
  const projectId = state.currentProjectId;
  card.dataset.epoch = chk.epoch;
  card.innerHTML = `
    ${chk.is_best ? '<div class="best-badge">⭐ Best</div>' : ''}
    <div class="checkpoint-epoch">Ep ${chk.epoch}</div>
    <div class="checkpoint-label">에폭 체크포인트</div>
    ${chk.loss != null ? `
      <div class="checkpoint-loss-label" style="margin-top:8px;">손실값</div>
      <div class="checkpoint-loss">${chk.loss.toFixed(4)}</div>
    ` : ''}
    <div id="weightGrade-${chk.epoch}" class="weight-grade-badge hidden"></div>
    <div class="checkpoint-actions">
      <button class="btn btn-accent btn-sm" onclick="openValidation(${chk.epoch})">🔬 검증</button>
      <a href="/api/projects/${projectId}/checkpoints/${chk.epoch}/download"
         class="btn btn-ghost btn-sm">⬇ 다운로드</a>
    </div>
  `;
  return card;
}

function addCheckpointCard(chk) {
  const grid = $('checkpointGrid');
  if (!grid) return;

  // Remove empty state
  const empty = grid.querySelector('.empty-checkpoints');
  if (empty) empty.remove();

  // Add card
  const existing = grid.querySelector(`[data-epoch="${chk.epoch}"]`);
  const card = buildCheckpointCard(chk);
  card.dataset.epoch = chk.epoch;
  if (existing) existing.replaceWith(card);
  else grid.appendChild(card);

  const dl = $('checkpointDownloadAll');
  if (dl) dl.classList.remove('hidden');
}

// ── Images ────────────────────────────────────────────────────────────────
async function loadImages(projectId) {
  try {
    const images = await api('GET', `/api/projects/${projectId}/images`);
    state.uploadedImages = images;
    renderImageGrid();
  } catch (e) {}
}

function renderImageGrid() {
  const grid = $('imageGrid');
  if (!grid) return;
  grid.innerHTML = '';
  state.uploadedImages.forEach(img => grid.appendChild(buildImageThumb(img)));
  updateImageCount();
}

function buildImageThumb(img) {
  const thumb = el('div', 'image-thumb');
  thumb.innerHTML = `
    <img src="${img.thumbnail_url || ''}" alt="${img.filename}" loading="lazy"
         onerror="this.parentElement.style.background='var(--surface3)'">
    <button class="image-thumb-del" onclick="deleteImage('${img.filename}', this)">✕</button>
  `;
  return thumb;
}

function updateImageCount() {
  const info = $('imageCountInfo');
  if (info) {
    const n = state.uploadedImages.length;
    info.textContent = n ? `${n}장 업로드됨` : '';
  }
}

async function handleFileSelect(e) {
  const files = Array.from(e.target.files);
  if (!files.length) return;
  await uploadFiles(files);
  e.target.value = '';
}

async function uploadFiles(files) {
  const id = state.currentProjectId;
  if (!id) return;

  const fd = new FormData();
  files.forEach(f => fd.append('files', f));

  try {
    const r = await fetch(`/api/projects/${id}/images`, { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail);
    const results = await r.json();
    results.forEach(img => {
      if (!state.uploadedImages.find(x => x.filename === img.filename)) {
        state.uploadedImages.push(img);
      }
    });
    renderImageGrid();
    toast(`${files.length}장 업로드됨`, 'success');
  } catch (e) {
    toast(`업로드 실패: ${e.message}`, 'error');
  }
}

async function deleteImage(filename, btn) {
  const id = state.currentProjectId;
  try {
    await api('DELETE', `/api/projects/${id}/images/${encodeURIComponent(filename)}`);
    state.uploadedImages = state.uploadedImages.filter(x => x.filename !== filename);
    btn.closest('.image-thumb').remove();
    updateImageCount();
  } catch (e) { toast(e.message, 'error'); }
}

function setupUploadZone(projectId) {
  const zone = $('uploadZone');
  if (!zone) return;

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', async e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files).filter(f => /\.(jpg|jpeg|png|webp)$/i.test(f.name));
    if (files.length) await uploadFiles(files);
  });
}

// ── Settings ───────────────────────────────────────────────────────────────
async function loadSettings(projectId) {
  try {
    const config = await api('GET', `/api/projects/${projectId}/config`);
    renderSettings(config);
  } catch (e) {}
}

function renderSettings(config) {
  const content = $('settingsContent');
  if (!content) return;
  const t = config.training || {};
  const m = config.model || {};

  content.innerHTML = `
    <div class="settings-section">
      <div class="settings-section-header">🤖 모델</div>
      <div class="settings-section-body">
        <div class="form-group">
          <label class="form-label">베이스 모델 경로</label>
          <div class="setting-value" style="word-break:break-all;font-size:12px;color:var(--text-muted)">${m.base_model || '—'}</div>
        </div>
      </div>
    </div>

    <div class="settings-section">
      <div class="settings-section-header">📐 LoRA 구조</div>
      <div class="settings-section-body">
        <div class="settings-grid">
          <div class="setting-item"><div class="setting-label">Rank (Dim)</div><div class="setting-value">${t.lora_rank || t.network_dim || '—'}</div></div>
          <div class="setting-item"><div class="setting-label">Alpha</div><div class="setting-value">${t.lora_alpha || t.network_alpha || '—'}</div></div>
        </div>
      </div>
    </div>

    <div class="settings-section">
      <div class="settings-section-header">📈 학습률</div>
      <div class="settings-section-body">
        <div class="settings-grid">
          <div class="setting-item"><div class="setting-label">UNet LR</div><div class="setting-value">${t.unet_lr || '—'}</div></div>
          <div class="setting-item"><div class="setting-label">Text Encoder LR</div><div class="setting-value">${t.network_train_unet_only ? '<span style="color:var(--text-dim)">끔</span>' : (Array.isArray(t.text_encoder_lr) ? t.text_encoder_lr[0] : (t.text_encoder_lr || '—'))}</div></div>
          <div class="setting-item"><div class="setting-label">스케줄러</div><div class="setting-value">${t.lr_scheduler || '—'}</div></div>
          <div class="setting-item"><div class="setting-label">사이클 수</div><div class="setting-value">${t.lr_scheduler_num_cycles || 1}</div></div>
        </div>
      </div>
    </div>

    <div class="settings-section">
      <div class="settings-section-header">⚙️ 학습 설정</div>
      <div class="settings-section-body">
        <div class="settings-grid">
          <div class="setting-item"><div class="setting-label">에폭</div><div class="setting-value">${t.num_epochs || '—'}</div></div>
          <div class="setting-item"><div class="setting-label">반복 횟수</div><div class="setting-value">${(()=>{const ic=state.project?.image_count||0;const ts=t.target_steps||2000;const ep=t.num_epochs||10;if(t.auto_num_repeats!==false&&ic>0&&ep>0){return Math.max(1,Math.min(Math.ceil(ts/(ic*ep)),50))+'<span style="font-size:11px;color:var(--text-dim)"> (자동계산)</span>';}return t.num_repeats||'—';})()}</div></div>
          <div class="setting-item"><div class="setting-label">Min SNR Gamma</div><div class="setting-value">${t.min_snr_gamma || '—'}</div></div>
          <div class="setting-item"><div class="setting-label">Noise Offset</div><div class="setting-value">${t.noise_offset || '—'}</div></div>
          <div class="setting-item"><div class="setting-label">해상도</div><div class="setting-value">${t.resolution || 1024}×${t.resolution || 1024}</div></div>
          <div class="setting-item"><div class="setting-label">Mixed Precision</div><div class="setting-value">${t.mixed_precision || 'fp16'}</div></div>
        </div>
      </div>
    </div>
  `;
}

// ── Wizard ─────────────────────────────────────────────────────────────────
function openWizard() {
  state.wizard = { step: 1, lora_type: null, name: '', trigger_word: '', base_model: '', gpu_mode: 'local' };
  state.uploadedImages = [];
  $('wizardModal').classList.add('open');
  renderWizardStep(1);
}

function closeWizard() {
  $('wizardModal').classList.remove('open');
}

function renderWizardStep(step) {
  state.wizard.step = step;
  updateWizardSteps(step);

  const body = $('wizardBody');
  switch (step) {
    case 1: body.innerHTML = buildStep1(); break;
    case 2: body.innerHTML = buildStep2(); break;
    case 3: body.innerHTML = buildStep3(); break;
    case 4: body.innerHTML = buildStep4(); break;
  }

  // Restore wizard state
  if (step === 2) {
    const nameEl = $('wizardName'); if (nameEl && state.wizard.name) nameEl.value = state.wizard.name;
    const trigEl = $('wizardTrigger'); if (trigEl && state.wizard.trigger_word) trigEl.value = state.wizard.trigger_word;
    const modEl = $('wizardModel'); if (modEl && state.wizard.base_model) modEl.value = state.wizard.base_model;
  }
  if (step === 3) setupWizardUpload();
  if (step === 4) {
    const localCard = $('gpuLocal'); if (localCard) localCard.classList.toggle('selected', state.wizard.gpu_mode === 'local');
    const vastCard = $('gpuVastai'); if (vastCard) vastCard.classList.toggle('selected', state.wizard.gpu_mode === 'vastai');
  }

  updateWizardButtons(step);
}

function buildStep1() {
  const types = [
    { key: 'style', emoji: '🎨', title: '그림체', desc: '화풍·색감·선 스타일 복사. 프리셋 선택 가능', specs: ['LoCon', 'TE 끔', '최소 30장+'] },
    { key: 'character', emoji: '👤', title: '캐릭터', desc: '특정 캐릭터의 외형과 의상을 학습', specs: ['Rank 32', 'TE 포함', '10 에폭', '최소 20장+'] },
    { key: 'face', emoji: '😊', title: '얼굴', desc: '특정 인물의 얼굴 특징을 세밀하게 학습', specs: ['Rank 16', '얼굴 크롭', '10 에폭', '최소 15장+'] },
    { key: 'object', emoji: '📦', title: '사물/개념', desc: '특정 오브젝트, 아이템, 개념을 학습', specs: ['Rank 32', '전체 이미지', '10 에폭', '최소 10장+'] },
  ];

  const stylePresets = [
    {
      key: 'style_balanced',
      emoji: '⚖️',
      title: '균형형',
      desc: '스타일 강함 + 포즈·구도 제어 가능',
      detail: 'LoCon · Rank 32 / Alpha 16 · conv 8 · TE 끔 · α/r=0.5',
      tags: ['권장'],
    },
    {
      key: 'style_copy',
      emoji: '🔥',
      title: '복사형',
      desc: '체크포인트 영향 최소화, 최강 그림체 복사',
      detail: 'LoCon · Rank 32 / Alpha 32 · conv 16 · TE 포함(저LR) · α/r=1.0',
      tags: ['강력'],
    },
    {
      key: 'style',
      emoji: '🎨',
      title: '기본형',
      desc: '기존 표준 설정 (TE 포함, Rank 64)',
      detail: 'networks.lora · Rank 64 / Alpha 32 · TE 포함',
      tags: [],
    },
    {
      key: 'style_custom',
      emoji: '⚙️',
      title: '커스텀',
      desc: '모든 파라미터 직접 설정',
      detail: '다음 단계에서 직접 입력',
      tags: [],
    },
  ];

  const isStyleSel = ['style','style_balanced','style_copy','style_custom'].includes(state.wizard.lora_type);

  return `
    <div style="margin-bottom:16px;">
      <div style="font-size:15px;font-weight:600;margin-bottom:6px;">LoRA 목적을 선택하세요</div>
      <div style="font-size:12px;color:var(--text-muted);">목적에 따라 최적의 학습 설정이 자동으로 적용됩니다</div>
    </div>
    <div class="purpose-grid">
      ${types.map(t => `
        <div class="purpose-card${(t.key === 'style' ? isStyleSel : state.wizard.lora_type === t.key) ? ' selected' : ''}"
             onclick="selectPurpose('${t.key}', this)">
          <div class="purpose-card-emoji">${t.emoji}</div>
          <div class="purpose-card-title">${t.title}</div>
          <div class="purpose-card-desc">${t.desc}</div>
          <div class="purpose-card-specs">
            ${t.specs.map(s => `<span class="spec-tag">${s}</span>`).join('')}
          </div>
        </div>
      `).join('')}
    </div>

    <!-- 그림체 프리셋 (style 선택 시 표시) -->
    <div id="stylePresetPanel" style="display:${isStyleSel ? 'block' : 'none'};margin-top:16px;">
      <div style="font-size:12px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px;">🎨 그림체 학습 방식 선택</div>
      <div class="style-preset-grid">
        ${stylePresets.map(p => `
          <div class="style-preset-card${state.wizard.lora_type === p.key ? ' selected' : ''}"
               onclick="selectStylePreset('${p.key}', this)">
            <div class="style-preset-top">
              <span class="style-preset-emoji">${p.emoji}</span>
              <span class="style-preset-title">${p.title}</span>
              ${p.tags.map(tag => `<span class="style-preset-tag">${tag}</span>`).join('')}
            </div>
            <div class="style-preset-desc">${p.desc}</div>
            <div class="style-preset-detail">${p.detail}</div>
          </div>
        `).join('')}
      </div>
      ${state.wizard.lora_type === 'style_custom' ? `
      <div class="style-custom-fields">
        <div style="font-size:12px;font-weight:600;color:var(--text-muted);margin-bottom:10px;">커스텀 파라미터</div>
        <div class="gen-params-grid" style="grid-template-columns:repeat(2,1fr);">
          <div class="form-group">
            <label class="form-label">Network Module</label>
            <select class="form-input form-select" id="customNetModule" onchange="_saveCustomPreset()">
              <option value="networks.lora">networks.lora (기본)</option>
              <option value="lycoris.kohya">lycoris.kohya (LoCon/LoKr)</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">LyCORIS Algo</label>
            <select class="form-input form-select" id="customAlgo" onchange="_saveCustomPreset()">
              <option value="locon">locon (권장)</option>
              <option value="lokr">lokr</option>
              <option value="lora">lora</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Rank (Dim)</label>
            <input class="form-input" id="customRank" type="number" min="1" max="512" value="${state.wizard.customPreset?.rank||128}" oninput="_saveCustomPreset()">
          </div>
          <div class="form-group">
            <label class="form-label">Alpha</label>
            <input class="form-input" id="customAlpha" type="number" min="1" max="512" value="${state.wizard.customPreset?.alpha||64}" oninput="_saveCustomPreset()">
          </div>
          <div class="form-group">
            <label class="form-label">Conv Dim</label>
            <input class="form-input" id="customConvDim" type="number" min="0" max="512" value="${state.wizard.customPreset?.convDim||32}" oninput="_saveCustomPreset()">
          </div>
          <div class="form-group">
            <label class="form-label">Conv Alpha</label>
            <input class="form-input" id="customConvAlpha" type="number" min="0" max="512" value="${state.wizard.customPreset?.convAlpha||32}" oninput="_saveCustomPreset()">
          </div>
          <div class="form-group" style="grid-column:1/-1;">
            <label class="form-label">
              <input type="checkbox" id="customTeOff" ${state.wizard.customPreset?.teOff ? 'checked' : ''} onchange="_saveCustomPreset()">
              &nbsp;Text Encoder 학습 끔 (포즈/구도 제어력 유지)
            </label>
          </div>
          <div class="form-group">
            <label class="form-label">UNet LR</label>
            <input class="form-input" id="customUnetLr" type="number" step="0.00001" value="${state.wizard.customPreset?.unetLr||0.0005}" oninput="_saveCustomPreset()">
          </div>
          <div class="form-group">
            <label class="form-label">에폭</label>
            <input class="form-input" id="customEpochs" type="number" min="1" max="100" value="${state.wizard.customPreset?.epochs||20}" oninput="_saveCustomPreset()">
          </div>
          <div class="form-group">
            <label class="form-label">Caption Dropout</label>
            <input class="form-input" id="customDropout" type="number" min="0" max="1" step="0.05" value="${state.wizard.customPreset?.dropout||0.1}" oninput="_saveCustomPreset()">
          </div>
        </div>
      </div>` : ''}
    </div>
  `;
}

function buildStep2() {
  return `
    <div class="form-group">
      <label class="form-label">프로젝트 이름</label>
      <input class="form-input" id="wizardName" placeholder="내 그림체 LoRA" value="${escHtml(state.wizard.name)}">
    </div>
    <div class="form-group">
      <label class="form-label">트리거 단어</label>
      <input class="form-input" id="wizardTrigger" placeholder="mystyle" value="${escHtml(state.wizard.trigger_word)}">
      <div class="form-hint">이 단어를 프롬프트에 넣으면 LoRA가 활성화됩니다. 영문, 특수문자 없이 짧게 설정하세요.</div>
    </div>
    <div class="form-group">
      <label class="form-label">베이스 모델 경로</label>
      <div class="path-input-row">
        <input class="form-input" id="wizardModel" placeholder="D:/Models/aMixIllustrious.safetensors" value="${escHtml(state.wizard.base_model)}">
        <button class="btn-browse" onclick="browsePath('wizardModel','.safetensors,.ckpt,.pt')" title="파일 찾아보기">📁</button>
      </div>
      <div class="form-hint">학습에 사용할 .safetensors 또는 .ckpt 파일의 전체 경로를 입력하세요.</div>
    </div>
  `;
}

function buildStep3() {
  return `
    <div style="margin-bottom:16px;">
      <div style="font-size:15px;font-weight:600;margin-bottom:4px;">학습 이미지 업로드</div>
      <div style="font-size:12px;color:var(--text-muted);">JPG, PNG, WEBP 파일을 업로드하세요. ${{style:'최소 50장+ 권장 (그림체)',character:'최소 20장+ 권장 (캐릭터)',face:'최소 15장+ 권장 (얼굴)',object:'최소 10장+ 권장 (사물)'}[state.wizard.lora_type]||'최소 20장 이상 권장'}</div>
    </div>
    <div class="upload-zone" id="wizardUploadZone" onclick="$('wizardFileInput').click()">
      <div class="upload-zone-icon">📁</div>
      <div class="upload-zone-title">이미지 드래그 또는 클릭</div>
      <div class="upload-zone-hint">JPG, PNG, WEBP · 여러 장 한 번에 가능</div>
    </div>
    <input type="file" id="wizardFileInput" multiple accept=".jpg,.jpeg,.png,.webp" style="display:none" onchange="handleWizardFiles(event)">
    <div id="wizardImageGrid" class="image-grid" style="margin-top:12px;"></div>
    <div id="wizardImageCount" style="margin-top:8px;font-size:12px;color:var(--text-muted);"></div>
  `;
}

function buildStep4() {
  return `
    <div style="margin-bottom:16px;">
      <div style="font-size:15px;font-weight:600;margin-bottom:4px;">GPU 선택</div>
      <div style="font-size:12px;color:var(--text-muted);">학습을 실행할 GPU 환경을 선택하세요</div>
    </div>
    <div class="gpu-toggle">
      <div class="gpu-card${state.wizard.gpu_mode === 'local' ? ' selected' : ''}" id="gpuLocal" onclick="selectGpuMode('local')">
        <div class="gpu-card-header">💻 로컬 GPU</div>
        <div class="gpu-card-desc">내 컴퓨터의 GPU로 학습합니다. CUDA가 설치되어 있어야 합니다.</div>
      </div>
      <div class="gpu-card${state.wizard.gpu_mode === 'vastai' ? ' selected' : ''}" id="gpuVastai" onclick="selectGpuMode('vastai')">
        <div class="gpu-card-header">☁️ Vast.ai (클라우드)</div>
        <div class="gpu-card-desc">Vast.ai의 GPU 인스턴스를 빌려 학습합니다. 고성능 GPU 사용 가능.</div>
      </div>
    </div>
    <div class="vastai-setup${state.wizard.gpu_mode === 'vastai' ? ' visible' : ''}" id="vastaiSetup">
      <div class="form-group">
        <label class="form-label">Vast.ai API Key</label>
        <input class="form-input" id="vastaiKey" placeholder="API 키를 입력하세요">
        <div class="form-hint">Vast.ai 콘솔 → Account → API Keys에서 발급받을 수 있습니다.</div>
      </div>
      <div class="form-group" style="margin-bottom:0">
        <label class="form-label">SSH 공개 키 경로</label>
        <input class="form-input" id="vastaiSshKey" placeholder="C:/Users/YourName/.ssh/id_rsa">
        <div class="form-hint">원격 인스턴스에 접속하기 위한 SSH 키 경로 (확장자 없는 개인 키).</div>
      </div>
    </div>
    <div style="margin-top:16px;">
      <div class="info-box">
        <strong>📋 학습 요약</strong><br>
        • 목적: <strong>${typeLabel(state.wizard.lora_type)}</strong><br>
        • 프로젝트: <strong>${escHtml(state.wizard.name)}</strong><br>
        • 트리거: <strong>${escHtml(state.wizard.trigger_word)}</strong><br>
        • 이미지: <strong>${state.uploadedImages.length}장</strong>
      </div>
    </div>
  `;
}

function selectPurpose(key, el) {
  const wasStyle = ['style','style_balanced','style_copy','style_custom'].includes(state.wizard.lora_type);
  // style 목적 카드 클릭 시 기존 style 프리셋 유지 or 초기화
  if (key === 'style') {
    const isAlreadyStyle = ['style','style_balanced','style_copy','style_custom'].includes(state.wizard.lora_type);
    if (!isAlreadyStyle) state.wizard.lora_type = 'style_balanced'; // 기본 프리셋
  } else {
    state.wizard.lora_type = key;
  }
  document.querySelectorAll('.purpose-card').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  // 그림체 프리셋 패널 토글
  const panel = $('stylePresetPanel');
  const isStyle = ['style','style_balanced','style_copy','style_custom'].includes(state.wizard.lora_type);
  if (panel) panel.style.display = isStyle ? 'block' : 'none';
}

function selectStylePreset(key, el) {
  const prev = state.wizard.lora_type;
  state.wizard.lora_type = key;
  // 커스텀 ON/OFF 전환 시 step 재렌더로 커스텀 필드 표시/숨김
  if (key === 'style_custom' || prev === 'style_custom') {
    renderWizardStep(1);
  } else {
    document.querySelectorAll('.style-preset-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
  }
}

function _saveCustomPreset() {
  state.wizard.customPreset = {
    netModule:  $('customNetModule')?.value || 'lycoris.kohya',
    algo:       $('customAlgo')?.value || 'locon',
    rank:       parseInt($('customRank')?.value) || 128,
    alpha:      parseInt($('customAlpha')?.value) || 64,
    convDim:    parseInt($('customConvDim')?.value) || 32,
    convAlpha:  parseInt($('customConvAlpha')?.value) || 32,
    teOff:      $('customTeOff')?.checked ?? true,
    unetLr:     parseFloat($('customUnetLr')?.value) || 0.0005,
    epochs:     parseInt($('customEpochs')?.value) || 20,
    dropout:    parseFloat($('customDropout')?.value) || 0.1,
  };
}

function selectGpuMode(mode) {
  state.wizard.gpu_mode = mode;
  document.querySelectorAll('.gpu-card').forEach(c => c.classList.remove('selected'));
  $(`gpu${mode === 'local' ? 'Local' : 'Vastai'}`)?.classList.add('selected');
  const setup = $('vastaiSetup');
  if (setup) setup.classList.toggle('visible', mode === 'vastai');
}

function setupWizardUpload() {
  const zone = $('wizardUploadZone');
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', async e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files).filter(f => /\.(jpg|jpeg|png|webp)$/i.test(f.name));
    if (files.length) { state.wizard._pendingFiles = (state.wizard._pendingFiles || []).concat(files); renderWizardImages(); }
  });
  // Re-render existing
  renderWizardImages();
}

async function handleWizardFiles(e) {
  const files = Array.from(e.target.files);
  state.wizard._pendingFiles = (state.wizard._pendingFiles || []).concat(files);
  renderWizardImages();
  e.target.value = '';
}

function renderWizardImages() {
  const grid = $('wizardImageGrid');
  const count = $('wizardImageCount');
  if (!grid) return;
  const files = state.wizard._pendingFiles || [];
  grid.innerHTML = '';
  files.slice(0, 40).forEach((file, i) => {
    const thumb = el('div', 'image-thumb');
    const url = URL.createObjectURL(file);
    thumb.innerHTML = `<img src="${url}" alt="${file.name}">
      <button class="image-thumb-del" onclick="removeWizardFile(${i})">✕</button>`;
    grid.appendChild(thumb);
  });
  if (files.length > 40) {
    const more = el('div', 'image-thumb', `<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:12px;color:var(--text-muted)">+${files.length-40}장</div>`);
    grid.appendChild(more);
  }
  if (count) count.textContent = files.length ? `${files.length}장 선택됨 (최소 20장 권장)` : '';
  // Sync to state
  state.uploadedImages = files.map(f => ({ filename: f.name }));
}

function removeWizardFile(i) {
  (state.wizard._pendingFiles || []).splice(i, 1);
  renderWizardImages();
}

function updateWizardSteps(active) {
  document.querySelectorAll('.wizard-step-item').forEach((item, i) => {
    const num = i + 1;
    item.classList.toggle('active', num === active);
    item.classList.toggle('done', num < active);
  });
}

function updateWizardButtons(step) {
  const footer = $('wizardFooter');
  if (!footer) return;
  footer.innerHTML = `
    ${step > 1 ? `<button class="btn btn-ghost" onclick="renderWizardStep(${step-1})">← 이전</button>` : '<div></div>'}
    <div style="display:flex;gap:8px;">
      <button class="btn btn-ghost" onclick="closeWizard()">취소</button>
      ${step < 4
        ? `<button class="btn btn-primary" onclick="wizardNext(${step})">다음 →</button>`
        : `<button class="btn btn-success" onclick="createProject()">✅ 프로젝트 생성</button>`}
    </div>
  `;
}

function wizardNext(step) {
  if (step === 1) {
    if (!state.wizard.lora_type) { toast('목적을 선택하세요', 'error'); return; }
  }
  if (step === 2) {
    const name = $('wizardName')?.value.trim();
    const trigger = $('wizardTrigger')?.value.trim();
    const model = $('wizardModel')?.value.trim();
    if (!name) { toast('프로젝트 이름을 입력하세요', 'error'); return; }
    if (!trigger) { toast('트리거 단어를 입력하세요', 'error'); return; }
    if (!model) { toast('베이스 모델 경로를 입력하세요', 'error'); return; }
    state.wizard.name = name;
    state.wizard.trigger_word = trigger;
    state.wizard.base_model = model;
  }
  if (step === 3) {
    const files = state.wizard._pendingFiles || [];
    if (!files.length) { toast('이미지를 최소 1장 업로드하세요', 'error'); return; }
  }
  renderWizardStep(step + 1);
}

async function createProject() {
  // Save Vast.ai settings if needed
  if (state.wizard.gpu_mode === 'vastai') {
    const key = $('vastaiKey')?.value.trim();
    const ssh = $('vastaiSshKey')?.value.trim();
    if (!key) { toast('Vast.ai API 키를 입력하세요', 'error'); return; }
    try {
      await api('POST', '/api/vastai/settings', { api_key: key, ssh_key_path: ssh || '' });
    } catch (e) {}
  }

  // Create project
  let project;
  try {
    const customP = state.wizard.customPreset;
    project = await api('POST', '/api/projects', {
      name: state.wizard.name,
      lora_type: state.wizard.lora_type,
      trigger_word: state.wizard.trigger_word,
      base_model: state.wizard.base_model,
      gpu_mode: state.wizard.gpu_mode,
      ...(state.wizard.lora_type === 'style_custom' && customP ? {
        custom_overrides: {
          training: {
            network_module: customP.netModule,
            network_args: customP.netModule === 'lycoris.kohya'
              ? [`algo=${customP.algo}`, `conv_dim=${customP.convDim}`, `conv_alpha=${customP.convAlpha}`]
              : [],
            lora_rank: customP.rank,
            lora_alpha: customP.alpha,
            unet_lr: customP.unetLr,
            network_train_unet_only: customP.teOff,
            num_epochs: customP.epochs,
            caption_dropout_rate: customP.dropout,
          }
        }
      } : {}),
    });
    toast('프로젝트 생성됨!', 'success');
  } catch (e) {
    toast(`생성 실패: ${e.message}`, 'error');
    return;
  }

  // Upload images
  const files = state.wizard._pendingFiles || [];
  if (files.length) {
    const fd = new FormData();
    files.forEach(f => fd.append('files', f));
    try {
      await fetch(`/api/projects/${project.id}/images`, { method: 'POST', body: fd });
    } catch (e) { toast('이미지 업로드 실패', 'error'); }
  }

  closeWizard();
  state.wizard._pendingFiles = [];
  await loadProjects();
  selectProject(project.id);
}

// ── Validation ────────────────────────────────────────────────────────────────
async function openValidation(epoch) {
  const modal = $('validationModal');
  if (!modal) return;
  modal.classList.add('open');

  $('valEpoch').textContent = epoch;
  $('valWeightResult').innerHTML = '<div class="val-loading">⏳ 가중치 분석 중...</div>';
  $('valInferenceSection').classList.add('hidden');
  $('valRunInference').dataset.epoch = epoch;

  // Run weight check immediately
  try {
    const r = await api('GET', `/api/projects/${state.currentProjectId}/validate/weight/${epoch}`);
    renderWeightResult(r, epoch);
  } catch (e) {
    $('valWeightResult').innerHTML = `<div class="val-error">❌ 분석 실패: ${e.message}</div>`;
  }
}

function closeValidation() {
  const modal = $('validationModal');
  if (modal) modal.classList.remove('open');
}

function renderWeightResult(r, epoch) {
  if (r.error) {
    $('valWeightResult').innerHTML = `<div class="val-error">❌ ${r.error}</div>`;
    return;
  }

  const gradeColor = { A: '#22c55e', B: '#84cc16', C: '#f59e0b', F: '#ef4444' }[r.grade] || '#888';
  const issueHtml = (r.issues || []).map(i => `
    <div class="val-issue val-issue-${i.level}">
      ${i.level === 'error' ? '❌' : '⚠️'} ${i.msg}
    </div>
  `).join('');

  const statsHtml = Object.entries(r.stats || {}).map(([k, v]) => `
    <div class="val-stat">
      <span class="val-stat-key">${k}</span>
      <span class="val-stat-val">${Array.isArray(v) ? v.join(', ') : v}</span>
    </div>
  `).join('');

  $('valWeightResult').innerHTML = `
    <div class="val-grade-row">
      <div class="val-grade" style="color:${gradeColor}">Grade ${r.grade}</div>
      <div class="val-filesize">${r.file_size_mb} MB</div>
    </div>
    ${issueHtml || '<div class="val-ok">✅ 이상 없음 — 가중치 정상</div>'}
    <div class="val-stats">${statsHtml}</div>
  `;

  // Update grade badge on the checkpoint card
  const badge = $(`weightGrade-${epoch}`);
  if (badge) {
    badge.textContent = `Grade ${r.grade}`;
    badge.className = `weight-grade-badge grade-${r.grade}`;
    badge.classList.remove('hidden');
  }

  $('valInferenceSection').classList.remove('hidden');
}

async function runInferenceTest() {
  const btn = $('valRunInference');
  const epoch = parseInt(btn.dataset.epoch);
  const baseModel = $('valBaseModel').value.trim();
  const prompt = $('valPrompt').value.trim();

  if (!baseModel) { toast('베이스 모델 경로를 입력하세요', 'error'); return; }

  $('valImages').innerHTML = '<div class="val-loading">🖼️ 이미지 생성 중... (30~120초 소요)</div>';
  btn.disabled = true;
  btn.textContent = '생성 중...';

  try {
    const r = await fetch(`/api/projects/${state.currentProjectId}/validate/inference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epoch, base_model: baseModel, prompt }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || '실패');

    $('valImages').innerHTML = `
      <div class="val-compare">
        <div class="val-compare-item">
          <div class="val-compare-label">LoRA 없음 (베이스)</div>
          <img src="data:image/png;base64,${data.before}" class="val-img">
        </div>
        <div class="val-compare-arrow">→</div>
        <div class="val-compare-item">
          <div class="val-compare-label">LoRA 적용 후</div>
          <img src="data:image/png;base64,${data.after}" class="val-img">
        </div>
      </div>
      <div style="font-size:11px;color:var(--text-muted);margin-top:8px;">
        프롬프트: ${escHtml(data.prompt_used)}
      </div>
    `;
  } catch (e) {
    $('valImages').innerHTML = `<div class="val-error">❌ ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🖼️ 이미지 생성 테스트';
  }
}

// ── Tab switching ────────────────────────────────────────────────────────────
function switchTab(name) {
  state.activeTab = name;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
  const btn = $(`tab-${name}`);
  const panel = $(`panel-${name}`);
  if (btn) btn.classList.add('active');
  if (panel) panel.classList.add('active');
}

// ── Empty / validator state ───────────────────────────────────────────────────
function showEmptyState() {
  const main = $('mainContent');
  main.innerHTML = `
    <div class="empty-state" id="emptyState">
      <div class="empty-state-icon">🎨</div>
      <h2>LoRA Maker</h2>
      <p>이미지를 넣으면 자동으로 최적화된 LoRA를 만들어드립니다.<br>왼쪽의 <strong>새 프로젝트</strong> 버튼으로 시작하세요.</p>
      <div style="display:flex;gap:12px;justify-content:center;margin-top:8px;">
        <button class="btn btn-primary" onclick="openWizard()">+ 새 프로젝트 만들기</button>
        <button class="btn btn-ghost" onclick="showValidatorView()">🔬 LoRA 검증하기</button>
      </div>
    </div>
  `;
}

function hideEmptyState() {
  const es = $('emptyState');
  if (es) es.remove();
}

// ── Standalone LoRA Validator View ────────────────────────────────────────────
function showValidatorView() {
  _saveGenForm();   // 생성기 상태 저장
  _saveValForm();   // 검증기 상태도 저장 (이미 열려 있을 경우 대비)
  state.currentProjectId = null;
  document.querySelectorAll('.project-item').forEach(i => i.classList.remove('active'));

  $('mainContent').innerHTML = `
    <div class="validator-view">
      <div class="validator-header">
        <div class="validator-title">🔬 LoRA 검증기</div>
        <div class="validator-sub">학습된 LoRA 파일의 가중치를 분석하고 품질을 확인합니다</div>
      </div>

      <div class="validator-body">
        <!-- Drop zone: file picker only, never modified after creation -->
        <div class="lora-drop-zone" id="loraDropZone"
             onclick="openLoraFileBrowser()"
             ondragover="event.preventDefault();this.classList.add('drag-over')"
             ondragleave="this.classList.remove('drag-over')"
             ondrop="onLoraDropped(event)">
          <div class="lora-drop-icon">💾</div>
          <div class="lora-drop-title">.safetensors 파일을 여기에 드래그</div>
          <div class="lora-drop-hint">또는 클릭해서 파일 선택</div>
        </div>

        <!-- Selected file panel: lives OUTSIDE the drop zone -->
        <div id="loraSelectedPanel" style="display:none;margin-top:12px;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px 16px;">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <span style="font-size:18px;">💾</span>
            <div style="flex:1;min-width:0;">
              <div id="loraSelectedName" style="font-weight:600;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"></div>
              <div id="loraSelectedPath" style="font-size:11px;color:var(--text-muted);margin-top:2px;word-break:break-all;"></div>
            </div>
            <button class="btn btn-ghost btn-sm" onclick="openLoraFileBrowser()">다시 선택</button>
            <button class="btn btn-primary btn-sm" onclick="analyzeStandalonePath()">분석 시작</button>
          </div>
        </div>

        <div id="standaloneValResult" style="margin-top:24px;"></div>

        <!-- Inference test (shown after weight analysis) -->
        <div id="standaloneInferenceSection" style="margin-top:24px;display:none" class="standalone-inference">
          <div class="val-section-title">🖼️ 이미지 생성 테스트 <span style="font-size:11px;font-weight:400;color:var(--text-muted)">(선택사항 · VRAM 필요)</span></div>
          <div class="form-group" style="margin-bottom:8px;margin-top:12px;">
            <label class="form-label">베이스 모델 경로</label>
            <div class="path-input-row">
              <input class="form-input" id="standaloneBaseModel" placeholder="D:/Models/animagine-xl.safetensors">
              <button class="btn-browse" onclick="browsePath('standaloneBaseModel','.safetensors,.ckpt')" title="파일 찾아보기">📁</button>
            </div>
            <div class="form-hint">학습에 사용한 SDXL 베이스 모델 파일 또는 폴더 경로</div>
          </div>
          <div class="form-group" style="margin-bottom:8px;">
            <label class="form-label">트리거 워드 (선택)</label>
            <input class="form-input" id="standaloneTrigger" placeholder="myartstyle">
          </div>
          <div class="form-group" style="margin-bottom:12px;">
            <label class="form-label">테스트 프롬프트</label>
            <input class="form-input" id="standalonePrompt" value="masterpiece, best quality, 1girl, portrait, detailed">
          </div>
          <div class="form-group" style="margin-bottom:8px;">
            <label class="form-label">네거티브 프롬프트 <span style="opacity:.6;font-weight:400">(선택)</span></label>
            <input class="form-input" id="standaloneNeg" placeholder="low quality, bad anatomy, watermark">
          </div>
          <!-- 파라미터 -->
          <div class="gen-params-grid" style="margin-bottom:12px;">
            <div class="form-group">
              <label class="form-label">스케줄러</label>
              <select class="form-input form-select" id="standaloneScheduler">
                <option value="euler">Euler</option>
                <option value="euler_a">Euler a</option>
                <option value="dpm++_2m" selected>DPM++ 2M</option>
                <option value="dpm++_2m_karras">DPM++ 2M Karras</option>
                <option value="ddim">DDIM</option>
                <option value="heun">Heun</option>
                <option value="lms">LMS</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">스텝</label>
              <input class="form-input" id="standaloneSteps" type="number" min="1" max="150" value="20">
            </div>
            <div class="form-group">
              <label class="form-label">CFG</label>
              <input class="form-input" id="standaloneCfg" type="number" min="1" max="20" step="0.5" value="7">
            </div>
            <div class="form-group">
              <label class="form-label">Seed <span style="opacity:.5;font-size:10px">(-1=랜덤)</span></label>
              <input class="form-input" id="standaloneSeed" type="number" value="-1">
            </div>
            <div class="form-group">
              <label class="form-label">LoRA 스트렝스 <span class="param-val-display" id="standaloneLoraScaleVal">1.00</span></label>
              <input class="form-range" id="standaloneLoraScale" type="range" min="0" max="2" step="0.05" value="1.0"
                oninput="$('standaloneLoraScaleVal').textContent = parseFloat(this.value).toFixed(2)">
            </div>
            <div class="form-group">
              <label class="form-label">노이즈 제거량 <span class="param-val-display" id="standaloneDenoisingVal">0.75</span></label>
              <input class="form-range" id="standaloneDenoising" type="range" min="0" max="1" step="0.05" value="0.75"
                oninput="$('standaloneDenoisingVal').textContent = parseFloat(this.value).toFixed(2)">
            </div>
          </div>
          <button class="btn btn-primary btn-sm" id="standaloneRunBtn" onclick="runStandaloneInference()">🖼️ 이미지 생성 테스트</button>
          <div id="standaloneImages" style="margin-top:16px;"></div>
        </div>
      </div>
    </div>
  `;
  _restoreValForm();  // 저장된 상태 복원
}

// current lora file path for standalone validator
state._standaloneLoraPath = null;

function openLoraFileBrowser() {
  // Electron: use IPC dialog
  if (window.electronAPI?.openFile) {
    const filters = [{ name: 'LoRA Files', extensions: ['safetensors', 'ckpt'] }, { name: 'All Files', extensions: ['*'] }];
    window.electronAPI.openFile(filters).then(fullPath => {
      if (fullPath) setLoraPath(fullPath);
    });
    return;
  }
  // Web: use backend tkinter dialog
  api('POST', '/api/browse', { mode: 'file', accept: '.safetensors,.ckpt', title: 'LoRA 파일 선택' })
    .then(res => { if (res.path) setLoraPath(res.path); })
    .catch(() => toast('파일 경로를 직접 입력해주세요', 'info'));
}

function setLoraPath(fullPath) {
  const name = fullPath.split(/[\/]/).pop();
  state._standaloneLoraPath = fullPath;

  // Update the drop zone to show selected state (no buttons inside)
  const zone = $('loraDropZone');
  if (zone) {
    zone.innerHTML = `
      <div class="lora-drop-icon" style="font-size:28px;">✅</div>
      <div class="lora-drop-title" style="font-size:13px;">${escHtml(name)}</div>
      <div class="lora-drop-hint">클릭해서 다시 선택</div>
    `;
  }

  // Show the separate action panel below the drop zone
  const panel = $('loraSelectedPanel');
  if (panel) {
    $('loraSelectedName').textContent = name;
    $('loraSelectedPath').textContent = fullPath;
    panel.style.display = 'block';
  }

  // Clear previous results
  const result = $('standaloneValResult');
  if (result) result.innerHTML = '';
  // 파일 선택하면 바로 이미지 생성 섹션 표시 (분석 없이도 사용 가능)
  const infSec = $('standaloneInferenceSection');
  if (infSec) infSec.style.display = 'block';
}

function onBrowseLora(event) {
  // Legacy: kept for drop event fallback
  const file = event.target.files?.[0];
  if (file) { event.target.value = ''; openLoraFileBrowser(); }
}

function onLoraDropped(event) {
  event.preventDefault();
  const zone = $('loraDropZone');
  if (zone) zone.classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (!file || !file.name.endsWith('.safetensors')) {
    toast('.safetensors 파일만 지원됩니다', 'error');
    return;
  }
  // Drag-drop also can't expose full path — open the browser dialog
  openLoraFileBrowser();
}

async function analyzeStandalonePath() {
  const path = state._standaloneLoraPath;
  if (!path) {
    toast('먼저 파일을 선택하세요', 'error');
    return;
  }
  const result = $('standaloneValResult');
  result.innerHTML = '<div class="val-loading">⏳ 가중치 분석 중...</div>';

  try {
    const r = await api('POST', '/api/validate/file', { file_path: path });
    renderStandaloneResult(r);
  } catch (e) {
    result.innerHTML = `<div class="val-error">❌ 분석 실패: ${e.message}</div>`;
  }
}

function renderStandaloneResult(r) {
  const result = $('standaloneValResult');
  if (r.error) {
    result.innerHTML = `<div class="val-error">❌ ${r.error}</div>`;
    return;
  }
  const gradeColor = { A: '#22c55e', B: '#84cc16', C: '#f59e0b', F: '#ef4444' }[r.grade] || '#888';
  const issueHtml = (r.issues || []).map(i =>
    `<div class="val-issue val-issue-${i.level}">${i.level==='error'?'❌':'⚠️'} ${i.msg}</div>`
  ).join('');
  const statsHtml = Object.entries(r.stats || {}).map(([k,v]) =>
    `<div class="val-stat"><span class="val-stat-key">${k}</span><span class="val-stat-val">${Array.isArray(v)?v.join(', '):v}</span></div>`
  ).join('');

  result.innerHTML = `
    <div class="val-section">
      <div class="val-section-title">📊 가중치 분석 결과</div>
      <div class="val-grade-row">
        <div class="val-grade" style="color:${gradeColor}">Grade ${r.grade}</div>
        <div class="val-filesize">${r.file_size_mb} MB</div>
      </div>
      ${issueHtml || '<div class="val-ok">✅ 이상 없음 — 가중치 정상</div>'}
      <div class="val-stats">${statsHtml}</div>
    </div>
  `;
  // _standaloneLoraPath is already set by setLoraPath() — do not overwrite
  const _inf = $('standaloneInferenceSection'); if(_inf) _inf.style.display='block';
}

async function runStandaloneInference() {
  const btn = $('standaloneRunBtn');
  const baseModel = $('standaloneBaseModel')?.value.trim();
  const prompt = $('standalonePrompt')?.value.trim() || 'masterpiece, best quality, portrait';
  const trigger = $('standaloneTrigger')?.value.trim() || '';
  const seedRaw = parseInt($('standaloneSeed')?.value) ?? -1;
  const seedVal = seedRaw === -1 ? Math.floor(Math.random() * 2147483647) : seedRaw;
  const loraPath = state._standaloneLoraPath;

  if (!baseModel) { toast('베이스 모델 경로를 입력하세요', 'error'); return; }
  if (!loraPath)  { toast('먼저 LoRA 파일을 분석하세요', 'error'); return; }

  $('standaloneImages').innerHTML = '<div class="val-loading">🖼️ 이미지 생성 중... 3가지 해상도 순서대로 진행 (1~3분 소요)</div>';
  btn.disabled = true;
  btn.textContent = '생성 중...';

  try {
    const r = await fetch('/api/validate/inference-file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        file_path: loraPath,
        base_model: baseModel,
        prompt,
        negative_prompt: $('standaloneNeg')?.value.trim() || '',
        trigger_word: trigger,
        steps: parseInt($('standaloneSteps')?.value) || 20,
        cfg_scale: parseFloat($('standaloneCfg')?.value) || 7.0,
        scheduler: $('standaloneScheduler')?.value || 'euler',
        seed: seedVal,
        lora_scale: parseFloat($('standaloneLoraScale')?.value ?? 1.0),
        denoising_strength: parseFloat($('standaloneDenoising')?.value ?? 0.75),
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || '실패');

    // New multi-resolution format: data.results = [{label, size, before, after}, ...]
    const results = data.results || [];
    if (!results.length) throw new Error('결과가 없습니다');

    const cols = results.map((res, i) => `
      <div class="val-resolution-col">
        <div class="val-resolution-header">
          <span class="val-res-badge">${escHtml(res.size)}</span>
          <span class="val-res-label">${escHtml(res.label)}</span>
        </div>
        <div class="val-compare-grid">
          <div class="val-compare-card" onclick="openLightbox('data:image/png;base64,${res.before}', 'LoRA 없음 · ${escHtml(res.size)}')">
            <div class="val-card-tag val-tag-before">Before</div>
            <img src="data:image/png;base64,${res.before}" class="val-img" loading="lazy">
            <div class="val-zoom-hint">🔍 클릭하여 확대</div>
          </div>
          <div class="val-compare-card" onclick="openLightbox('data:image/png;base64,${res.after}', 'LoRA 적용 · ${escHtml(res.size)}')">
            <div class="val-card-tag val-tag-after">After</div>
            <img src="data:image/png;base64,${res.after}" class="val-img" loading="lazy">
            <div class="val-zoom-hint">🔍 클릭하여 확대</div>
          </div>
        </div>
      </div>
    `).join('');

    $('standaloneImages').innerHTML = `
      <div class="val-prompt-bar">
        <span class="val-prompt-icon">💬</span>
        <span class="val-prompt-text">${escHtml(data.prompt_used)}</span>
      </div>
      <div class="val-multi-resolution">${cols}</div>
    `;
  } catch (e) {
    $('standaloneImages').innerHTML = `<div class="val-error">❌ ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🖼️ 이미지 생성 테스트';
  }
}



// ── Standalone Image Generator View ──────────────────────────────────────────
// ── Standalone Image Generator View ──────────────────────────────────────────

function _saveGenForm() {
  const f = window._genForm;
  const safe = (id, def) => $(id)?.value ?? def;
  f.baseModel = safe('genBaseModel', f.baseModel);
  f.loraFile  = safe('genLoraFile',  f.loraFile);
  f.prompt    = safe('genPrompt',    f.prompt);
  f.neg       = safe('genNeg',       f.neg);
  f.trigger   = safe('genTrigger',   f.trigger);
  f.scheduler = safe('genScheduler', f.scheduler);
  f.steps     = parseInt(safe('genSteps', f.steps));
  f.cfg       = parseFloat(safe('genCfg', f.cfg));
  f.seed      = parseInt(safe('genSeed', f.seed));
  f.count     = parseInt(safe('genCount', f.count));
  f.loraScale = parseFloat(safe('genLoraScale', f.loraScale));
  f.denoising = parseFloat(safe('genDenoising', f.denoising));
  f.customW   = parseInt(safe('genCustomW', f.customW));
  f.customH   = parseInt(safe('genCustomH', f.customH));
  document.querySelectorAll('.gen-res-check').forEach(cb => {
    f['res_' + cb.value.replace(/[|×]/g,'_')] = cb.checked;
  });
  f.inputImagePath = $('genInputImage')?.value.trim() || '';
  const gp = $('genGalleryPanel');
  if (gp && !gp.querySelector('.gen-gallery-empty')) f.galleryHTML = gp.innerHTML;
  const qp = $('genQueuePanel');
  if (qp && qp.style.display !== 'none') f.queueHTML = qp.innerHTML;
}

function _restoreGenForm() {
  const f = window._genForm;
  const set = (id, val) => { const el = $(id); if (el && val !== undefined && val !== '') el.value = val; };
  set('genBaseModel', f.baseModel);
  set('genLoraFile',  f.loraFile);
  set('genPrompt',    f.prompt);
  set('genNeg',       f.neg);
  set('genTrigger',   f.trigger);
  set('genScheduler', f.scheduler);
  set('genSteps',     f.steps);
  set('genCfg',       f.cfg);
  set('genSeed',      f.seed);
  set('genCount',     f.count);
  set('genLoraScale', f.loraScale);
  set('genDenoising', f.denoising);
  set('genCustomW',   f.customW);
  set('genCustomH',   f.customH);
  // update slider displays
  if ($('genLoraScaleVal'))  $('genLoraScaleVal').textContent  = parseFloat(f.loraScale).toFixed(2);
  if ($('genDenoisingVal'))  $('genDenoisingVal').textContent  = parseFloat(f.denoising).toFixed(2);
  // restore i2i image
  if (f.inputImagePath) setInputImage(f.inputImagePath);
  // restore gallery — galleryHTML is kept up-to-date by _autoSaveGallery on every result
  const gp = $('genGalleryPanel');
  if (gp && f.galleryHTML) {
    gp.innerHTML = f.galleryHTML;
    _updateGalleryCount();
  }
    // restore queue
  const qp = $('genQueuePanel');
  if (qp) _renderGenQueue();
  updateGenHint();
  // Restore running state UI if generation is still active
  if (_gen.running) {
    const rb = $('genRunBtn');
    if (rb) rb.textContent = '🔄 생성 중 (클릭: 대기열 추가)';
    const cb = $('genCancelBtn');
    if (cb) cb.style.display = 'inline-flex';
    const pw = $('genProgressWrap');
    if (pw) pw.style.display = 'block';
  }
}

function _saveValForm() {
  const f = window._valForm;
  const safe = (id, def) => $(id)?.value ?? def;
  f.loraPath  = window.state?._standaloneLoraPath || f.loraPath;
  f.baseModel = safe('standaloneBaseModel', f.baseModel);
  f.prompt    = safe('standalonePrompt',    f.prompt);
  f.neg       = safe('standaloneNeg',       f.neg);
  f.trigger   = safe('standaloneTrigger',   f.trigger);
  f.scheduler = safe('standaloneScheduler', f.scheduler);
  f.steps     = parseInt(safe('standaloneSteps', f.steps));
  f.cfg       = parseFloat(safe('standaloneCfg', f.cfg));
  f.seed      = parseInt(safe('standaloneSeed', f.seed));
  f.loraScale = parseFloat(safe('standaloneLoraScale', f.loraScale));
  f.denoising = parseFloat(safe('standaloneDenoising', f.denoising));
  const ri = $('standaloneImages');
  if (ri && ri.innerHTML && !ri.querySelector('.val-loading')) f.resultsHTML = ri.innerHTML;
}

function _restoreValForm() {
  const f = window._valForm;
  const set = (id, val) => { const el = $(id); if (el && val !== undefined && val !== '') el.value = val; };
  if (f.loraPath) {
    state._standaloneLoraPath = f.loraPath;
    // update path display
    const lp = $('standaloneLoraPath');
    if (lp) lp.textContent = f.loraPath.split(/[\\/]/).pop();
    const inf = $('standaloneInferenceSection');
    if (inf) inf.style.display = 'block';
  }
  set('standaloneBaseModel', f.baseModel);
  set('standalonePrompt',    f.prompt);
  set('standaloneNeg',       f.neg);
  set('standaloneTrigger',   f.trigger);
  set('standaloneScheduler', f.scheduler);
  set('standaloneSteps',     f.steps);
  set('standaloneCfg',       f.cfg);
  set('standaloneSeed',      f.seed);
  set('standaloneLoraScale', f.loraScale);
  set('standaloneDenoising', f.denoising);
  if ($('standaloneLoraScaleVal')) $('standaloneLoraScaleVal').textContent = parseFloat(f.loraScale).toFixed(2);
  if ($('standaloneDenoisingVal')) $('standaloneDenoisingVal').textContent = parseFloat(f.denoising).toFixed(2);
  const ri = $('standaloneImages');
  if (ri && f.resultsHTML) ri.innerHTML = f.resultsHTML;
}
const _gen = window._genState;

function showImageGenView() {
  _saveValForm();   // 검증기 상태 저장
  _saveGenForm();   // 생성기 상태도 저장 (이미 열려 있을 경우 대비)
  state.currentProjectId = null;
  document.querySelectorAll('.project-item').forEach(i => i.classList.remove('active'));
  $('mainContent').innerHTML = `
    <div class="gen-layout">
      <!-- ── Left: config ─────────────────────── -->
      <div class="gen-left">
        <div class="gen-config-card">
          <div class="gen-view-title">🎨 이미지 생성기</div>

          <div class="gen-section-title" style="margin-top:14px;">📁 모델</div>
          <div class="form-group">
            <label class="form-label">베이스 모델</label>
            <div class="path-input-row">
              <input class="form-input" id="genBaseModel" placeholder="D:/Models/animagine-xl.safetensors">
              <button class="btn-browse" onclick="browsePath('genBaseModel','.safetensors,.ckpt')">📁</button>
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">LoRA 파일</label>
            <div class="path-input-row">
              <input class="form-input" id="genLoraFile" placeholder="D:/LoRA/my_lora.safetensors">
              <button class="btn-browse" onclick="browsePath('genLoraFile','.safetensors,.ckpt')">📁</button>
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">트리거 워드 <span style="opacity:.6;font-weight:400">(선택)</span></label>
            <input class="form-input" id="genTrigger" placeholder="myartstyle">
          </div>

          <div class="gen-divider"></div>

          <div class="gen-section-title">✏️ 프롬프트</div>
          <div class="form-group">
            <label class="form-label">포지티브</label>
            <textarea class="form-input gen-textarea" id="genPrompt" rows="3">masterpiece, best quality, 1girl, portrait, detailed</textarea>
          </div>
          <div class="form-group">
            <label class="form-label">네거티브</label>
            <textarea class="form-input gen-textarea" id="genNeg" rows="2" placeholder="low quality, bad anatomy, watermark, blurry"></textarea>
          </div>

          <div class="gen-divider"></div>

          <div class="gen-section-title">📐 해상도</div>
          <div class="gen-res-list">
            <label class="gen-res-item"><input type="checkbox" class="gen-res-check" value="1024|1024|1:1" checked><span class="gen-res-size">1024×1024</span><span class="gen-res-ratio">1:1</span></label>
            <label class="gen-res-item"><input type="checkbox" class="gen-res-check" value="832|1216|2:3 세로" checked><span class="gen-res-size">832×1216</span><span class="gen-res-ratio">2:3 세로</span></label>
            <label class="gen-res-item"><input type="checkbox" class="gen-res-check" value="1216|832|3:2 가로" checked><span class="gen-res-size">1216×832</span><span class="gen-res-ratio">3:2 가로</span></label>
            <label class="gen-res-item gen-res-custom-row">
              <input type="checkbox" class="gen-res-check" id="genCustomResCheck" value="custom">
              <span class="gen-res-size">Custom</span>
              <span class="gen-res-ratio gen-custom-inputs">
                <input class="form-input gen-custom-dim" id="genCustomW" type="number" min="64" max="2048" step="64" value="512" onclick="event.stopPropagation()">
                <span style="color:var(--text-muted)">×</span>
                <input class="form-input gen-custom-dim" id="genCustomH" type="number" min="64" max="2048" step="64" value="512" onclick="event.stopPropagation()">
              </span>
            </label>
          </div>

          <div class="gen-divider"></div>

          <!-- i2i 이미지 입력 -->
          <div class="gen-section-title">🖼️ 입력 이미지 <span style="font-size:10px;font-weight:400;color:var(--text-muted)">(없으면 t2i, 있으면 i2i)</span></div>
          <div class="form-group" style="margin-bottom:4px;">
            <div class="path-input-row">
              <input class="form-input" id="genInputImage" placeholder="비워두면 t2i 모드 (텍스트→이미지)" readonly
                style="cursor:pointer;font-size:11px;color:var(--text-muted);"
                onclick="browseInputImage()">
              <button class="btn-browse" onclick="browseInputImage()" title="이미지 선택">📁</button>
              <button class="btn-browse" id="genInputImageClear" onclick="clearInputImage()" title="지우기" style="display:none;color:var(--red);">✕</button>
            </div>
          </div>
          <!-- i2i 미리보기 + 노이즈 제거량 (이미지 있을 때만 표시) -->
          <div id="genI2IPanel" style="display:none;margin-bottom:8px;">
            <div class="i2i-preview-box">
              <img id="genInputImageThumb" src="" alt="입력 이미지"
                style="width:100%;max-height:220px;object-fit:contain;border-radius:8px;background:var(--bg-deep);display:block;">
            </div>
            <div class="form-group" style="margin-top:10px;margin-bottom:0;">
              <label class="form-label">노이즈 제거량 <span class="param-val-display" id="genDenoisingVal">0.75</span></label>
              <input class="form-range" id="genDenoising" type="range" min="0.1" max="1" step="0.05" value="0.75"
                oninput="$('genDenoisingVal').textContent = parseFloat(this.value).toFixed(2)">
              <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--text-muted);margin-top:3px;">
                <span>0.1 미세변형</span><span>0.75 권장</span><span>1.0 완전변환</span>
              </div>
            </div>
          </div>

          <div class="gen-divider"></div>

          <div class="gen-section-title">⚙️ 파라미터</div>
          <div class="gen-params-grid">
            <div class="form-group" style="grid-column:1/-1;">
              <label class="form-label">스케줄러</label>
              <select class="form-input form-select" id="genScheduler">
                <option value="euler">Euler</option>
                <option value="euler_a">Euler a</option>
                <option value="dpm++_2m" selected>DPM++ 2M</option>
                <option value="dpm++_2m_karras">DPM++ 2M Karras</option>
                <option value="ddim">DDIM</option>
                <option value="heun">Heun</option>
                <option value="lms">LMS</option>
              </select>
            </div>
            <div class="form-group">
              <label class="form-label">스텝</label>
              <input class="form-input" id="genSteps" type="number" min="1" max="150" value="20">
            </div>
            <div class="form-group">
              <label class="form-label">CFG</label>
              <input class="form-input" id="genCfg" type="number" min="1" max="20" step="0.5" value="7">
            </div>
          </div>
          <div class="gen-params-grid" style="margin-top:0;">
            <div class="form-group">
              <label class="form-label">Seed <span style="opacity:.5;font-size:10px">(-1=랜덤)</span></label>
              <input class="form-input" id="genSeed" type="number" value="-1">
            </div>
            <div class="form-group">
              <label class="form-label">생성 횟수</label>
              <input class="form-input" id="genCount" type="number" min="1" max="20" value="1" oninput="updateGenHint()">
            </div>
            <div class="form-group">
              <label class="form-label">LoRA 스트렝스 <span class="param-val-display" id="genLoraScaleVal">1.00</span></label>
              <input class="form-range" id="genLoraScale" type="range" min="0" max="2" step="0.05" value="1.0"
                oninput="$('genLoraScaleVal').textContent = parseFloat(this.value).toFixed(2)">
            </div>

          </div>
          <div class="gen-count-hint" id="genCountHint">해상도 3개 × 1회 = 총 6장 생성</div>

          <div style="display:flex;gap:8px;margin-top:12px;">
            <button class="btn btn-primary" id="genRunBtn" onclick="runImageGen()" style="flex:1;">
              🖼️ 생성 시작
            </button>
            <button class="btn btn-danger" id="genCancelBtn" onclick="cancelImageGen()" style="display:none;">
              ✕ 취소
            </button>
          </div>

          <!-- 진행 상황 -->
          <div id="genProgressWrap" style="display:none;margin-top:12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
              <div class="gen-progress-label" id="genProgressLabel">준비 중...</div>
              <div class="gen-progress-pct" id="genProgressPct">0%</div>
            </div>
            <div class="gen-progress-track">
              <div class="gen-progress-fill" id="genProgressFill" style="width:0%"></div>
            </div>
          </div>

          <!-- 대기열 -->
          <div id="genQueuePanel" style="display:none;margin-top:10px;"></div>
        </div>
      </div>

      <!-- ── Right: gallery ────────────────────── -->
      <div class="gen-right">
        <div class="gen-gallery-header">
          <span id="genGalleryCount" class="gen-gallery-stat">생성된 이미지 없음</span>
          <button class="btn btn-ghost btn-sm" onclick="clearGenGallery()">🗑 초기화</button>
        </div>
        <div id="genGalleryPanel" class="gen-gallery-panel">
          <div class="gen-gallery-empty">
            <div class="gen-gallery-empty-icon">🖼️</div>
            <div>생성된 이미지가 여기에 표시됩니다</div>
          </div>
        </div>
      </div>
    </div>
  `;

  // 체크박스 변경 시 힌트 업데이트
  document.querySelectorAll('.gen-res-check').forEach(cb => cb.addEventListener('change', updateGenHint));
  _restoreGenForm();   // 저장된 상태 복원
}

function updateGenHint() {
  const checked = document.querySelectorAll('.gen-res-check:checked').length;
  const count = parseInt($('genCount')?.value) || 1;
  const hint = $('genCountHint');
  if (hint) hint.textContent = `해상도 ${checked}개 × ${count}회 = 총 ${checked * count * 2}장 생성 (Before/After 포함)`;
}

function getSelectedResolutions() {
  const list = [];
  document.querySelectorAll('.gen-res-check:checked').forEach(cb => {
    if (cb.value === 'custom') {
      const w = parseInt($('genCustomW')?.value) || 512;
      const h = parseInt($('genCustomH')?.value) || 512;
      list.push({ width: w, height: h, label: 'Custom' });
    } else {
      const [w, h, label] = cb.value.split('|');
      list.push({ width: parseInt(w), height: parseInt(h), label });
    }
  });
  return list;
}

function browseInputImage() {
  if (window.electronAPI?.openFile) {
    const filters = [{ name: 'Images', extensions: ['png','jpg','jpeg','webp','bmp'] }];
    window.electronAPI.openFile(filters).then(p => { if (p) setInputImage(p); });
    return;
  }
  api('POST', '/api/browse', { mode: 'file', accept: '.png,.jpg,.jpeg,.webp,.bmp', title: '입력 이미지 선택' })
    .then(res => { if (res.path) setInputImage(res.path); })
    .catch(() => toast('파일 경로를 직접 붙여넣기 하세요', 'info'));
}

function setInputImage(imgPath) {
  const inp = $('genInputImage');
  if (inp) { inp.value = imgPath; inp.style.color = ''; }
  const thumb = $('genInputImageThumb');
  if (thumb) thumb.src = '/api/image?path=' + encodeURIComponent(imgPath);
  const panel = $('genI2IPanel');
  if (panel) panel.style.display = 'block';
  const clrBtn = $('genInputImageClear');
  if (clrBtn) clrBtn.style.display = '';
  window._genForm.inputImagePath = imgPath;
}

function clearInputImage() {
  const inp = $('genInputImage');
  if (inp) { inp.value = ''; inp.style.color = 'var(--text-muted)'; }
  const panel = $('genI2IPanel');
  if (panel) panel.style.display = 'none';
  const clrBtn = $('genInputImageClear');
  if (clrBtn) clrBtn.style.display = 'none';
  window._genForm.inputImagePath = '';
}

function _readGenParams() {
  return {
    baseModel:   $('genBaseModel')?.value.trim() || '',
    loraFile:    $('genLoraFile')?.value.trim() || '',
    prompt:      $('genPrompt')?.value.trim() || 'masterpiece, best quality, portrait',
    neg:         $('genNeg')?.value.trim() || '',
    trigger:     $('genTrigger')?.value.trim() || '',
    steps:       parseInt($('genSteps')?.value) || 20,
    cfg:         parseFloat($('genCfg')?.value) || 7.0,
    scheduler:   $('genScheduler')?.value || 'euler',
    baseSeed:    parseInt($('genSeed')?.value) ?? -1,
    count:       Math.min(parseInt($('genCount')?.value) || 1, 20),
    resolutions: getSelectedResolutions(),
    loraScale:      parseFloat($('genLoraScale')?.value ?? 1.0),
    denoising:      parseFloat($('genDenoising')?.value ?? 0.75),
    inputImagePath: $('genInputImage')?.value.trim() || '',
  };
}

function runImageGen() {
  const p = _readGenParams();
  if (!p.baseModel)          { toast('베이스 모델 경로를 입력하세요', 'error'); return; }
  if (!p.loraFile)           { toast('LoRA 파일 경로를 입력하세요', 'error'); return; }
  if (!p.resolutions.length) { toast('해상도를 하나 이상 선택하세요', 'error'); return; }

  const jobId = ++_gen.queueCounter;
  const job = { id: jobId, params: p, label: `#${jobId} · Seed ${p.baseSeed === -1 ? '랜덤' : p.baseSeed} · ${p.resolutions.length}해상도 × ${p.count}회` };

  if (_gen.running) {
    _gen.queue.push(job);
    _renderGenQueue();
    toast(`대기열에 추가됨 (대기 ${_gen.queue.length}개)`, 'info');
    return;
  }
  _startGenJob(job);
}

function _renderGenQueue() {
  const el = $('genQueuePanel');
  if (!el) return;
  if (_gen.queue.length === 0) { el.style.display = 'none'; el.innerHTML = ''; return; }
  el.style.display = 'block';
  el.innerHTML = `
    <div class="gen-queue-header">⏳ 대기열 <span class="gen-queue-badge">${_gen.queue.length}</span></div>
    ${_gen.queue.map(j => `
      <div class="gen-queue-item" id="qitem-${j.id}">
        <span class="gen-queue-label">${escHtml(j.label)}</span>
        <button class="btn-queue-cancel" onclick="cancelQueueItem(${j.id})" title="대기열에서 제거">✕</button>
      </div>
    `).join('')}
    <button class="btn btn-ghost btn-sm" style="width:100%;margin-top:4px;" onclick="clearGenQueue()">🗑 대기열 전체 취소</button>
  `;
}

function cancelQueueItem(id) {
  _gen.queue = _gen.queue.filter(j => j.id !== id);
  _renderGenQueue();
  toast('대기열 항목 제거됨', 'info');
}

function clearGenQueue() {
  _gen.queue = [];
  _renderGenQueue();
  toast('대기열이 비워졌습니다', 'info');
}

async function _startGenJob(job) {
  const { params: p, id: jobId } = job;
  const btn       = $('genRunBtn');
  const cancelBtn = $('genCancelBtn');
  const pw  = $('genProgressWrap');
  const pl  = $('genProgressLabel');
  const pf  = $('genProgressFill');
  const pct = $('genProgressPct');

  _gen.running   = true;
  _gen.cancelled = false;
  if (btn) { btn.textContent = '🔄 생성 중 (클릭: 대기열 추가)'; }
  if (cancelBtn) cancelBtn.style.display = 'inline-flex';
  if (pw) pw.style.display = 'block';

  let pollTimer = null;
  function startPoll(genId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const r = await fetch('/api/generate/progress/' + genId);
        const pp = await r.json();
        if (pl)  pl.textContent  = pp.label  || '생성 중...';
        if (pf)  pf.style.width  = (pp.percent || 0) + '%';
        if (pct) pct.textContent = (pp.percent || 0) + '%';
      } catch (_) {}
    }, 350);
  }
  function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

  if (pl)  pl.textContent  = `[#${jobId}] 준비 중...`;
  if (pf)  pf.style.width  = '0%';
  if (pct) pct.textContent = '0%';

  try {
    for (let i = 0; i < p.count; i++) {
      if (_gen.cancelled) break;
      const seed  = p.baseSeed === -1 ? Math.floor(Math.random() * 2147483647) : p.baseSeed + i;
      const genId = 'gen-' + Date.now() + '-' + i;
      _gen.currentGenId = genId;
      const loadId = 'gen-load-' + genId;
      _genPrependLoading(loadId, i + 1, p.count, seed);
      startPoll(genId);
      try {
        const r = await fetch('/api/validate/inference-file', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_path: p.loraFile, base_model: p.baseModel,
            prompt: p.prompt, negative_prompt: p.neg, trigger_word: p.trigger,
            steps: p.steps, cfg_scale: p.cfg, scheduler: p.scheduler,
            seed, resolutions: p.resolutions, gen_id: genId,
            lora_scale: p.loraScale, denoising_strength: p.denoising,
            input_image_path: p.inputImagePath || '',
          }),
        });
        stopPoll();
        const data = await r.json();
        if (data.cancelled) { _genReplaceError(loadId, '취소됨'); break; }
        if (!r.ok) throw new Error(data.detail || '실패');
        const actualSeed = data.seed_used ?? seed;  // 백엔드 실제 seed 우선
        _genReplaceResult(loadId, {
          seed: actualSeed, scheduler: p.scheduler, steps: p.steps, cfg: p.cfg,
          prompt: data.prompt_used, results: data.results || [],
          time: new Date().toLocaleTimeString(),
        });
      } catch (e) {
        stopPoll();
        _genReplaceError(loadId, _gen.cancelled ? '취소됨' : e.message);
        if (_gen.cancelled) break;
      }
    }
  } finally {
    stopPoll();
    _gen.currentGenId = null;
    _gen.running = false;
    if (btn) btn.textContent = '🖼️ 생성 시작';
    if (cancelBtn) cancelBtn.style.display = 'none';
    if (pl)  pl.textContent  = _gen.cancelled ? '⛔ 취소됨' : '✅ 완료!';
    if (pf)  pf.style.width  = '100%';
    if (pct) pct.textContent = '100%';
    setTimeout(() => { if (pw) pw.style.display = 'none'; }, 2000);
    // process next in queue
    if (_gen.queue.length > 0 && !_gen.cancelled) {
      const next = _gen.queue.shift();
      _renderGenQueue();
      _startGenJob(next);
    } else {
      _gen.queue = [];
      _renderGenQueue();
    }
  }
}

function cancelImageGen() {
  _gen.cancelled = true;
  if (_gen.currentGenId) {
    fetch('/api/generate/cancel/' + _gen.currentGenId, { method: 'POST' }).catch(() => {});
  }
  const pl = $('genProgressLabel');
  if (pl) pl.textContent = '⛔ 취소 요청 중... (현재 해상도 완료 후 중단)';
  // also clear queue
  _gen.queue = [];
  _renderGenQueue();
}

// ── Gallery helpers ───────────────────────────────────────────────────────────
function _genPrependLoading(id, cur, total, seed) {
  // Store pending slot in buffer regardless of whether view is active
  if (!_gen.pendingSlots) _gen.pendingSlots = {};
  _gen.pendingSlots[id] = { cur, total, seed, status: 'loading' };

  const panel = $('genGalleryPanel');
  if (!panel) return;
  panel.querySelector('.gen-gallery-empty')?.remove();
  const el = document.createElement('div');
  el.id = id;
  el.className = 'gen-result-card gen-result-loading';
  el.innerHTML = `
    <div class="gen-result-header">
      <span class="gen-result-run">#${_gen.gallery.length + cur}</span>
      <span class="gen-result-meta">Seed ${seed} · 생성 중... (${cur}/${total})</span>
    </div>
    <div class="gen-loading-pulse">⏳ 이미지 생성 중...</div>
  `;
  panel.insertBefore(el, panel.firstChild);
  _updateGalleryCount();
}

function _genReplaceResult(id, data) {
  // Always record the result — even if the view is not currently visible
  _gen.gallery.unshift(data);
  if (!_gen.resultBuffer) _gen.resultBuffer = [];
  _gen.resultBuffer.unshift({ id, data });
  if (_gen.pendingSlots) delete _gen.pendingSlots[id];

  // Build card HTML (used both for DOM insertion and for off-screen save)
  const _cardAfterCols = data.results.map(res => `
    <div class="gen-after-col">
      <div class="val-compare-card" onclick="openLightbox('data:image/png;base64,${res.after}','LoRA 적용 · ${escHtml(res.size)} · Seed:${data.seed}')">
        <div class="val-card-tag val-tag-after">After</div>
        <img src="data:image/png;base64,${res.after}" class="val-img gen-after-img" loading="lazy">
        <div class="val-zoom-hint">🔍 확대</div>
      </div>
      <div class="gen-before-strip" onclick="openLightbox('data:image/png;base64,${res.before}','Before · ${escHtml(res.size)}')">
        <img src="data:image/png;base64,${res.before}" loading="lazy">
        <span class="gen-before-label">${escHtml(res.size)}</span>
      </div>
    </div>
  `).join('');
  const _cardHTML = `<div class="gen-result-header">
      <span class="gen-result-run">#${_gen.gallery.length}</span>
      <span class="gen-result-meta">Seed ${data.seed} · ${escHtml(data.scheduler)} · ${data.steps}step · CFG ${data.cfg}</span>
      <span class="gen-result-time">${data.time}</span>
    </div>
    <div class="gen-prompt-line">💬 ${escHtml(data.prompt)}</div>
    <div class="gen-after-row">${_cardAfterCols}</div>`;

  const el = document.getElementById(id);
  if (!el) {
    // View not active — directly write the card HTML into galleryHTML so it's ready on return
    const existingGallery = window._genForm.galleryHTML || '';
    window._genForm.galleryHTML =
      '<div class="gen-result-card">' + _cardHTML + '</div>' + existingGallery;
    return;
  }

  el.className = 'gen-result-card';
  el.innerHTML = _cardHTML;
  _updateGalleryCount();
  _autoSaveGallery();  // persist so view-switch doesn't lose results
}

function _genReplaceError(id, msg) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'gen-result-card';
  el.innerHTML = `<div class="val-error">❌ ${escHtml(msg)}</div>`;
}

function _autoSaveGallery() {
  const panel = $('genGalleryPanel');
  if (panel && !panel.querySelector('.gen-gallery-empty')) {
    window._genForm.galleryHTML = panel.innerHTML;
  }
}

function clearGenGallery() {
  _gen.gallery = [];
  const panel = $('genGalleryPanel');
  if (panel) panel.innerHTML = `<div class="gen-gallery-empty"><div class="gen-gallery-empty-icon">🖼️</div><div>생성된 이미지가 여기에 표시됩니다</div></div>`;
  _updateGalleryCount();
}

function _updateGalleryCount() {
  const el = $('genGalleryCount');
  if (!el) return;
  const n = _gen.gallery.length;
  el.textContent = n ? `총 ${n}회 생성됨` : '생성된 이미지 없음';
}

// ── Lightbox ─────────────────────────────────────────────────────────────────
function openLightbox(src, caption) {
  let lb = document.getElementById('imgLightbox');
  if (!lb) {
    lb = document.createElement('div');
    lb.id = 'imgLightbox';
    lb.className = 'lightbox-overlay';
    lb.addEventListener('click', function(e) {
      if (e.target === lb || e.target.classList.contains('lightbox-backdrop')) closeLightbox();
    });
    lb.innerHTML = `
      <div class="lightbox-backdrop"></div>
      <div class="lightbox-box" onclick="event.stopPropagation()">
        <div class="lightbox-caption" id="lbCaption"></div>
        <img id="lbImg" class="lightbox-img">
        <button class="lightbox-close" onclick="closeLightbox()">✕</button>
      </div>
    `;
    document.body.appendChild(lb);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });
  }
  document.getElementById('lbImg').src = src;
  document.getElementById('lbCaption').textContent = caption || '';
  lb.classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  const lb = document.getElementById('imgLightbox');
  if (lb) lb.classList.remove('open');
  document.body.style.overflow = '';
}

// ── File path browser helper ──────────────────────────────────────────────────
let _browseTargetId = null;

function browsePath(targetInputId, accept = '') {
  _browseTargetId = targetInputId;

  // Electron: use native file dialog to get the full path
  if (window.electronAPI?.openFile) {
    const extList = accept.split(',').map(e => e.trim().replace(/^\./, ''));
    const filters = extList.length
      ? [{ name: 'Model Files', extensions: extList }, { name: 'All Files', extensions: ['*'] }]
      : [{ name: 'All Files', extensions: ['*'] }];

    window.electronAPI.openFile(filters).then(fullPath => {
      if (!fullPath) return;
      const target = $(_browseTargetId);
      if (target) target.value = fullPath;
    });
    return;
  }

  // Web mode: call backend /api/browse to open a native tkinter dialog
  api('POST', '/api/browse', { mode: 'file', accept, title: '파일 선택' })
    .then(res => {
      if (res.path) {
        const target = $(_browseTargetId);
        if (target) target.value = res.path;
      }
    })
    .catch(() => {
      toast('파일 경로를 직접 입력해주세요', 'info');
    });
}

function onBrowseFile(event) {
  // Legacy handler kept for <input type="file"> — not used in Electron mode
  event.target.value = '';
}

// ── Settings Modal (global) ──────────────────────────────────────────────────
function openGlobalSettings() {
  $('settingsModal').classList.add('open');
  loadVastAIStatus();
}
function closeGlobalSettings() { $('settingsModal').classList.remove('open'); }

async function loadVastAIStatus() {
  try {
    const s = await api('GET', '/api/vastai/settings');
    $('vastaiCurrentKey').textContent = s.has_api_key ? `설정됨 (${s.api_key_preview})` : '미설정';
    $('vastaiCurrentSsh').textContent = s.ssh_key_path || '미설정';
  } catch (e) {}
}

async function saveVastAISettings() {
  const key = $('globalVastaiKey').value.trim();
  const ssh = $('globalVastaiSsh').value.trim();
  if (!key) { toast('API 키를 입력하세요', 'error'); return; }
  try {
    await api('POST', '/api/vastai/settings', { api_key: key, ssh_key_path: ssh });
    toast('Vast.ai 설정 저장됨', 'success');
    loadVastAIStatus();
    $('globalVastaiKey').value = '';
  } catch (e) { toast(e.message, 'error'); }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function bindEvents() {
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      $('wizardModal').classList.remove('open');
      $('settingsModal').classList.remove('open');
    }
  });
  $('wizardModal').addEventListener('click', e => { if (e.target === $('wizardModal')) closeWizard(); });
  $('settingsModal').addEventListener('click', e => { if (e.target === $('settingsModal')) closeGlobalSettings(); });
}

function toast(msg, type = 'info') {
  const container = $('toastContainer');
  const t = el('div', `toast ${type}`, msg);
  container.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function typeEmoji(type) {
  return { style: '🎨', character: '👤', face: '😊', object: '📦' }[type] || '🔧';
}
function typeLabel(type) {
  return { style: '그림체', character: '캐릭터', face: '얼굴', object: '사물' }[type] || type;
}
function typeColor(type) {
  return { style: '#7c3aed', character: '#1f6feb', face: '#2ea043', object: '#9e6a03' }[type] || '#7c3aed';
}
function statusLabel(s) {
  return { pending: '대기 중', running: '실행 중', preprocessing: '전처리 중', captioning: '캡션 생성 중', training: '학습 중', completed: '완료', failed: '실패', cancelled: '취소됨' }[s] || s;
}
function formatEta(s) {
  if (!s || s < 0) return '—';
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}시간 ${m}분`;
  if (m) return `${m}분 ${sec}초`;
  return `${sec}초`;
}
function escHtml(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
