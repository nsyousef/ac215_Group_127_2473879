// Small environment helpers

export function isElectron() {
  // Renderer process in Electron often exposes process.type === 'renderer'
  try {
    if (typeof window !== 'undefined' && typeof navigator !== 'undefined') {
      const ua = navigator.userAgent || '';
      if (ua.toLowerCase().includes('electron')) return true;
    }
  } catch (e) {
    // ignore
  }

  // Also support an environment variable flag (for Node contexts)
  try {
    if (typeof process !== 'undefined' && process.env && process.env.ELECTRON) return true;
  } catch (e) {}

  return false;
}
