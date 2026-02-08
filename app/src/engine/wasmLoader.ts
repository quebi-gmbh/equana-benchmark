export interface WasmExports {
  memory: WebAssembly.Memory;
  malloc_f64: (count: number) => number;
  free_f64: (ptr: number) => void;
  [funcName: string]: unknown;
}

const wasmCache = new Map<string, WasmExports>();

export async function loadStandaloneWasm(filename: string): Promise<WasmExports> {
  const cached = wasmCache.get(filename);
  if (cached) return cached;

  const url = `${import.meta.env.BASE_URL}wasm/${filename}`;
  const response = await fetch(url);
  const bytes = await response.arrayBuffer();

  const { instance } = await WebAssembly.instantiate(bytes, {
    env: {
      emscripten_notify_memory_growth: () => {},
    },
  });

  const exports = instance.exports as unknown as WasmExports;
  wasmCache.set(filename, exports);
  return exports;
}
