import type { BenchmarkVariant, BenchmarkResult, BenchmarkAction, JsConfig, WasmConfig, MtConfig, PyodideConfig } from './types';
import { generateMatrices } from './matrixUtils';
import { runJsBenchmark } from './runners/jsRunner';
import { runWasmBenchmark } from './runners/wasmRunner';
import { runMtBenchmark } from './runners/mtRunner';
import { runPyodideBenchmark } from './runners/pyodideRunner';

function yieldToUI(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      setTimeout(resolve, 0);
    });
  });
}

async function executeSingle(
  variant: BenchmarkVariant,
  N: number,
  A_f64: Float64Array,
  B_f64: Float64Array,
  A_arr: number[],
  B_arr: number[],
  rounds: number,
  threadCount: number,
): Promise<BenchmarkResult> {
  switch (variant.runner) {
    case 'js':
      return runJsBenchmark(variant.config as JsConfig, N, A_f64, B_f64, A_arr, B_arr, rounds);
    case 'wasm':
      return runWasmBenchmark(variant.config as WasmConfig, N, A_f64, B_f64, rounds);
    case 'mt':
      return runMtBenchmark(variant.config as MtConfig, N, A_f64, B_f64, rounds, threadCount);
    case 'pyodide':
      return runPyodideBenchmark(variant.config as PyodideConfig, N, A_f64, B_f64, rounds);
  }
}

export async function runSingleBenchmark(
  variant: BenchmarkVariant,
  N: number,
  rounds: number,
  threadCount: number,
  dispatch: React.Dispatch<BenchmarkAction>,
): Promise<void> {
  const { A_f64, B_f64, A_arr, B_arr } = generateMatrices(N);

  dispatch({ type: 'SET_RUNNING', payload: variant.id });
  await yieldToUI();

  try {
    const result = await executeSingle(variant, N, A_f64, B_f64, A_arr, B_arr, rounds, threadCount);
    dispatch({ type: 'SET_RESULT', payload: { id: variant.id, result } });
  } catch (error) {
    dispatch({ type: 'SET_ERROR', payload: { id: variant.id, error: (error as Error).message } });
  }

  dispatch({ type: 'SET_IDLE' });
}

export async function runAllBenchmarks(
  benchmarks: BenchmarkVariant[],
  N: number,
  rounds: number,
  threadCount: number,
  dispatch: React.Dispatch<BenchmarkAction>,
): Promise<void> {
  const { A_f64, B_f64, A_arr, B_arr } = generateMatrices(N);

  for (const variant of benchmarks) {
    dispatch({ type: 'SET_RUNNING', payload: variant.id });
    await yieldToUI();

    try {
      const result = await executeSingle(variant, N, A_f64, B_f64, A_arr, B_arr, rounds, threadCount);
      dispatch({ type: 'SET_RESULT', payload: { id: variant.id, result } });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: { id: variant.id, error: (error as Error).message } });
    }

    await yieldToUI();
  }

  dispatch({ type: 'SET_IDLE' });
}
