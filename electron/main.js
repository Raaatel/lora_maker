const { app, BrowserWindow, shell, Menu, Tray, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');

const PORT = 7860;
let mainWindow = null;
let serverProcess = null;
let tray = null;
let serverReady = false;

// ── Python server paths ──────────────────────────────────────────────────────
function getAppDir() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'app');
  }
  return path.join(__dirname, '..');
}

function getPythonExe() {
  const appDir = getAppDir();
  const candidates = [
    path.join(appDir, 'venv', 'Scripts', 'python.exe'),
    path.join(appDir, 'venv', 'bin', 'python'),
    // Python 3.10 default location
    path.join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python310', 'python.exe'),
    'python',
    'python3',
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return 'python';
}

// ── Start Python FastAPI server ──────────────────────────────────────────────
function startServer() {
  const appDir = getAppDir();
  const python = getPythonExe();
  const script = path.join(appDir, 'app.py');

  console.log(`Starting server: ${python} ${script}`);

  serverProcess = spawn(python, [script, '--no-browser', '--port', String(PORT)], {
    cwd: appDir,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  serverProcess.stdout.on('data', d => {
    const msg = d.toString().trim();
    if (msg) console.log('[server]', msg);
  });

  serverProcess.stderr.on('data', d => {
    const msg = d.toString().trim();
    if (msg) console.error('[server]', msg);
  });

  serverProcess.on('close', code => {
    console.log(`Server exited with code ${code}`);
    serverReady = false;
    if (mainWindow && !app.isQuitting) {
      dialog.showErrorBox('서버 오류', `Python 서버가 종료되었습니다 (코드: ${code}).\n앱을 다시 시작해주세요.`);
    }
  });
}

// ── Wait for server to be ready ──────────────────────────────────────────────
function waitForServer(retries = 40, interval = 500) {
  return new Promise((resolve, reject) => {
    let attempts = 0;
    const check = () => {
      http.get(`http://localhost:${PORT}/`, res => {
        serverReady = true;
        resolve();
      }).on('error', () => {
        attempts++;
        if (attempts >= retries) {
          reject(new Error('서버 시작 실패'));
          return;
        }
        setTimeout(check, interval);
      });
    };
    check();
  });
}

// ── Create window ────────────────────────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 900,
    minHeight: 600,
    title: 'LoRA Maker',
    backgroundColor: '#0d1117',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    // Frameless option (uncomment for custom titlebar):
    // frame: false,
    autoHideMenuBar: true,
  });

  // Remove default menu
  Menu.setApplicationMenu(buildMenu());

  // Load splash while server starts
  mainWindow.loadURL('data:text/html,<html style="background:#0d1117;display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;color:#9f67fa;flex-direction:column;gap:16px"><div style="font-size:28px;font-weight:700">LoRA Maker</div><div style="font-size:14px;color:#8b949e">서버 시작 중...</div></html>');
  mainWindow.show();

  waitForServer()
    .then(() => {
      mainWindow.loadURL(`http://localhost:${PORT}`);
    })
    .catch(err => {
      dialog.showErrorBox('시작 오류', `서버에 연결할 수 없습니다.\n${err.message}\n\nPython과 필요 패키지가 설치되었는지 확인하세요.`);
    });

  // Open external links in default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => { mainWindow = null; });
  mainWindow.on('close', e => {
    if (!app.isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });
}

function buildMenu() {
  const template = [
    {
      label: 'LoRA Maker',
      submenu: [
        { label: '새 프로젝트', accelerator: 'CmdOrCtrl+N', click: () => mainWindow?.webContents.executeJavaScript('openWizard()') },
        { type: 'separator' },
        { label: '개발자 도구', accelerator: 'F12', click: () => mainWindow?.webContents.toggleDevTools() },
        { label: '새로고침', accelerator: 'CmdOrCtrl+R', click: () => mainWindow?.reload() },
        { type: 'separator' },
        { label: '종료', accelerator: 'CmdOrCtrl+Q', click: () => { app.isQuitting = true; app.quit(); } },
      ],
    },
  ];
  return Menu.buildFromTemplate(template);
}

// ── App lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(() => {
  startServer();
  createWindow();
});

app.on('before-quit', () => {
  app.isQuitting = true;
  if (serverProcess) {
    serverProcess.kill();
    serverProcess = null;
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (serverProcess) serverProcess.kill();
    app.quit();
  }
});

app.on('activate', () => {
  if (!mainWindow) createWindow();
  else mainWindow.show();
});
