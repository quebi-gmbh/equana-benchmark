export interface MtModule {
  HEAPF64: Float64Array;
  _set_num_threads: (n: number) => void;
  _get_num_threads: () => number;
  _malloc_f64: (count: number) => number;
  _free_f64: (ptr: number) => void;
  _cleanup: () => void;
  [key: string]: unknown;
}

const moduleCache = new Map<string, MtModule>();

function assertSharedArrayBuffer(): void {
  if (typeof SharedArrayBuffer === 'undefined') {
    throw new Error(
      'SharedArrayBuffer is not available. Multi-threaded WASM requires cross-origin isolation (COOP/COEP headers).',
    );
  }
}

function injectScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

async function loadNonModularized(jsFile: string): Promise<MtModule> {
  const baseUrl = `${import.meta.env.BASE_URL}wasm/`;
  const src = `${baseUrl}${jsFile}`;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const win = window as any;

  const module = await new Promise<MtModule>((resolve, reject) => {
    win.Module = {
      locateFile: (path: string) => `${baseUrl}${path}`,
      onRuntimeInitialized: () => {
        resolve(win.Module as MtModule);
      },
      onAbort: () => reject(new Error(`WASM module ${jsFile} aborted`)),
    };
    injectScript(src).catch(reject);
  });

  return module;
}

async function loadModularized(
  jsFile: string,
  factoryName: string,
  subDir: string,
): Promise<MtModule> {
  const baseUrl = `${import.meta.env.BASE_URL}wasm/${subDir}`;
  const src = `${baseUrl}${jsFile}`;

  await injectScript(src);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const factory = (window as any)[factoryName] as
    | ((opts?: Record<string, unknown>) => Promise<MtModule>)
    | undefined;

  if (!factory) {
    throw new Error(`Factory function "${factoryName}" not found after loading ${jsFile}`);
  }

  const module = await factory({
    locateFile: (path: string) => `${baseUrl}${path}`,
  });

  return module;
}

export async function loadMtModule(moduleId: string): Promise<MtModule> {
  const cached = moduleCache.get(moduleId);
  if (cached) return cached;

  assertSharedArrayBuffer();

  let module: MtModule;

  switch (moduleId) {
    case 'matmul_mt':
      module = await loadNonModularized('matmul_mt.js');
      break;
    case 'matmul_sse_mt':
      module = await loadModularized('matmul_sse_mt.js', 'createSSEModule', '');
      break;
    default:
      throw new Error(`Unknown MT module: ${moduleId}`);
  }

  moduleCache.set(moduleId, module);
  return module;
}
