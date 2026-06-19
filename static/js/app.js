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
          <div class="setting-item"><div class="setting-label">Text Encoder LR</div><div class="setting-value">${Array.isArray(t.text_encoder_lr) ? t.text_encoder_lr[0] : (t.text_encoder_lr || '—')}</div></div>
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
          <div class="setting-item"><div class="setting-label">반복 횟수</div><div class="setting-value">${t.num_repeats || '—'} <span style="font-size:11px;color:var(--text-dim)">(자동 계산)</span></div></div>
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
    { key: 'style', emoji: '🎨', title: '그림체', desc: '특정 작가나 스타일의 화풍 전체를 복사', specs: ['Rank 64', 'Dual TE LR', '15 에폭', '최소 50장+'] },
    { key: 'character', emoji: '👤', title: '캐릭터', desc: '특정 캐릭터의 외형과 의상을 학습', specs: ['Rank 32', 'TE 포함', '10 에폭', '최소 20장+'] },
    { key: 'face', emoji: '😊', title: '얼굴', desc: '특정 인물의 얼굴 특징을 세밀하게 학습', specs: ['Rank 16', '얼굴 크롭', '10 에폭', '최소 15장+'] },
    { key: 'object', emoji: '📦', title: '사물/개념', desc: '특정 오브젝트, 아이템, 개념을 학습', specs: ['Rank 32', '전체 이미지', '10 에폭', '최소 10장+'] },
  ];

  return `
    <div style="margin-bottom:16px;">
      <div style="font-size:15px;font-weight:600;margin-bottom:6px;">LoRA 목적을 선택하세요</div>
      <div style="font-size:12px;color:var(--text-muted);">목적에 따라 최적의 학습 설정이 자동으로 적용됩니다</div>
    </div>
    <div class="purpose-grid">
      ${types.map(t => `
        <div class="purpose-card${state.wizard.lora_type === t.key ? ' selected' : ''}"
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
  state.wizard.lora_type = key;
  document.querySelectorAll('.purpose-card').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
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
    project = await api('POST', '/api/projects', {
      name: state.wizard.name,
      lora_type: state.wizard.lora_type,
      trigger_word: state.wizard.trigger_word,
      base_model: state.wizard.base_model,
      gpu_mode: state.wizard.gpu_mode,
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
  state.currentProjectId = null;
  document.querySelectorAll('.project-item').forEach(i => i.classList.remove('active'));

  $('mainContent').innerHTML = `
    <div class="validator-view">
      <div class="validator-header">
        <div class="validator-title">🔬 LoRA 검증기</div>
        <div class="validator-sub">학습된 LoRA 파일의 가중치를 분석하고 품질을 확인합니다</div>
      </div>

      <div class="validator-body">
        <!-- Drop zone -->
        <div class="lora-drop-zone" id="loraDropZone"
             onclick="$('browseLoraInput').click()"
             ondragover="event.preventDefault();this.classList.add('drag-over')"
             ondragleave="this.classList.remove('drag-over')"
             ondrop="onLoraDropped(event)">
          <div class="lora-drop-icon">💾</div>
          <div class="lora-drop-title">.safetensors 파일을 여기에 드래그</div>
          <div class="lora-drop-hint">또는 클릭해서 파일 선택</div>
        </div>

        <div id="standaloneValResult" style="margin-top:24px;"></div>

        <!-- Inference test (shown after weight analysis) -->
        <div id="standaloneInferenceSection" class="hidden" style="margin-top:24px;">
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
          <button class="btn btn-primary btn-sm" id="standaloneRunBtn" onclick="runStandaloneInference()">🖼️ 이미지 생성 테스트</button>
          <div id="standaloneImages" style="margin-top:16px;"></div>
        </div>
      </div>
    </div>
  `;
}

// current lora file path for standalone validator
state._standaloneLoraPath = null;

function onBrowseLora(event) {
  const file = event.target.files[0];
  if (!file) return;
  // Browser security: we only get the filename, not full path.
  // Show name and let user confirm/edit full path.
  const zone = $('loraDropZone');
  if (zone) {
    zone.innerHTML = `
      <div class="lora-drop-icon">💾</div>
      <div class="lora-drop-title">${escHtml(file.name)}</div>
      <div class="lora-drop-hint" style="color:var(--text-muted)">
        전체 경로를 아래에 입력하세요
      </div>
      <input class="form-input" id="loraPathInput" style="margin-top:10px;max-width:420px;"
             placeholder="D:/lora-maker/lora-maker/data/jobs/.../output/xxx.safetensors"
             value="">
      <button class="btn btn-primary btn-sm" style="margin-top:8px;"
              onclick="analyzeStandalonePath()">분석 시작</button>
    `;
  }
  event.target.value = '';
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
  if (zone) {
    zone.innerHTML = `
      <div class="lora-drop-icon">💾</div>
      <div class="lora-drop-title">${escHtml(file.name)}</div>
      <div class="lora-drop-hint">전체 경로를 입력하세요</div>
      <input class="form-input" id="loraPathInput" style="margin-top:10px;max-width:420px;"
             placeholder="C:/lora-maker/lora-maker/data/jobs/.../output/xxx.safetensors">
      <button class="btn btn-primary btn-sm" style="margin-top:8px;"
              onclick="analyzeStandalonePath()">분석 시작</button>
    `;
  }
}

async function analyzeStandalonePath() {
  const input = $('loraPathInput');
  if (!input || !input.value.trim()) {
    toast('경로를 입력하세요', 'error');
    return;
  }
  const path = input.value.trim();
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
  state._standaloneLoraPath = $('loraPathInput')?.value?.trim();
  $('standaloneInferenceSection')?.classList.remove('hidden');
}

async function runStandaloneInference() {
  const btn = $('standaloneRunBtn');
  const baseModel = $('standaloneBaseModel')?.value.trim();
  const prompt = $('standalonePrompt')?.value.trim() || 'masterpiece, best quality, portrait';
  const trigger = $('standaloneTrigger')?.value.trim() || '';
  const loraPath = state._standaloneLoraPath;

  if (!baseModel) { toast('베이스 모델 경로를 입력하세요', 'error'); return; }
  if (!loraPath)  { toast('먼저 LoRA 파일을 분석하세요', 'error'); return; }

  $('standaloneImages').innerHTML = '<div class="val-loading">🖼️ 이미지 생성 중... (30~120초)</div>';
  btn.disabled = true;
  btn.textContent = '생성 중...';

  try {
    const r = await fetch('/api/validate/inference-file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_path: loraPath, base_model: baseModel, prompt, trigger_word: trigger }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || '실패');
    $('standaloneImages').innerHTML = `
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
      <div style="font-size:11px;color:var(--text-muted);margin-top:8px;">프롬프트: ${escHtml(data.prompt_used)}</div>
    `;
  } catch (e) {
    $('standaloneImages').innerHTML = `<div class="val-error">❌ ${e.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🖼️ 이미지 생성 테스트';
  }
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
