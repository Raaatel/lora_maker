const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  platform: process.platform,
  version: process.env.npm_package_version || '1.0.0',

  // Native file open dialog — returns full path string or null
  openFile: (filters) => ipcRenderer.invoke('dialog:openFile', { filters }),
});
