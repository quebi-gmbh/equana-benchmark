export type Category = 'javascript' | 'wasm' | 'wasm-mt' | 'pyodide';
export type RunnerType = 'js' | 'wasm' | 'mt' | 'pyodide';

export interface JsConfig {
  fn: 'naiveArray' | 'naiveFloat64' | 'cacheArray' | 'packedArray';
  inputType: 'array' | 'float64';
}

export interface WasmConfig {
  wasmFile: string;
  funcName: string;
}

export interface MtConfig {
  moduleId: 'matmul_mt' | 'matmul_sse_mt';
  funcName: string;
}

export interface PyodideConfig {
  _tag: 'pyodide';
}

export interface BenchmarkVariant {
  id: string;
  name: string;
  category: Category;
  description: string;
  runner: RunnerType;
  config: JsConfig | WasmConfig | MtConfig | PyodideConfig;
  /** Maximum matrix size this variant supports. Omit for no limit. */
  maxSize?: number;
}

export interface BenchmarkResult {
  avg: number;
  times: number[];
  gflops: number;
}

export interface MatrixData {
  A_f64: Float64Array;
  B_f64: Float64Array;
  A_arr: number[];
  B_arr: number[];
}

export type BenchmarkAction =
  | { type: 'SET_SIZE'; payload: number }
  | { type: 'SET_ROUNDS'; payload: number }
  | { type: 'SET_THREADS'; payload: number }
  | { type: 'SET_RUNNING'; payload: string }
  | { type: 'SET_RESULT'; payload: { id: string; result: BenchmarkResult } }
  | { type: 'SET_ERROR'; payload: { id: string; error: string } }
  | { type: 'SET_IDLE' };

export interface BenchmarkState {
  matrixSize: number;
  rounds: number;
  threadCount: number;
  results: Record<string, BenchmarkResult>;
  errors: Record<string, string>;
  runningId: string | null;
  globalRunning: boolean;
}
