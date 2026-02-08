/**
 * Minimal type declarations for the Pyodide API surface we use.
 * Avoids adding @pyodide/pyodide as a dependency.
 */
interface PyodideInterface {
  loadPackage(pkg: string | string[]): Promise<void>;
  runPythonAsync(code: string): Promise<PyProxy>;
}

export interface PyProxy {
  toJs(): Map<string, unknown>;
  destroy(): void;
}

let cachedPyodide: PyodideInterface | null = null;
let loadingPromise: Promise<PyodideInterface> | null = null;

function injectScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.crossOrigin = 'anonymous'; // Required for COEP: require-corp
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

export async function loadPyodideRuntime(): Promise<PyodideInterface> {
  if (cachedPyodide) return cachedPyodide;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    const CDN_URL = 'https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js';
    await injectScript(CDN_URL);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const loadPyodide = (window as any).loadPyodide as
      | (() => Promise<PyodideInterface>)
      | undefined;

    if (!loadPyodide) {
      throw new Error('loadPyodide not found on window after script injection');
    }

    const pyodide = await loadPyodide();
    await pyodide.loadPackage('numpy');

    cachedPyodide = pyodide;
    loadingPromise = null;
    return pyodide;
  })();

  return loadingPromise;
}
