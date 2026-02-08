import { useCallback } from 'react';
import type { BenchmarkVariant } from '../engine/types';
import { BENCHMARKS } from '../engine/benchmarkRegistry';
import { runSingleBenchmark, runAllBenchmarks } from '../engine/scheduler';
import { useBenchmarkStore } from '../hooks/useBenchmarkStore';
import { useSEO } from '../hooks/useSEO';
import { MatrixSizePicker } from '../components/MatrixSizePicker';
import { RoundsPicker } from '../components/RoundsPicker';
import { ThreadCountPicker } from '../components/ThreadCountPicker';
import { RunAllButton } from '../components/RunAllButton';
import { BenchmarkTable } from '../components/BenchmarkTable';

export function BenchmarkPage() {
  useSEO({
    title: 'Matrix Multiplication Performance',
    description:
      'Compare JavaScript, WebAssembly SIMD, and multi-threaded WASM matrix multiplication performance directly in your browser using double-precision (f64) arithmetic.',
    path: '/',
  });

  const [state, dispatch] = useBenchmarkStore();

  const handleRunAll = useCallback(() => {
    void runAllBenchmarks(BENCHMARKS, state.matrixSize, state.rounds, state.threadCount, dispatch);
  }, [state.matrixSize, state.rounds, state.threadCount, dispatch]);

  const handleRunSingle = useCallback(
    (variant: BenchmarkVariant) => {
      void runSingleBenchmark(variant, state.matrixSize, state.rounds, state.threadCount, dispatch);
    },
    [state.matrixSize, state.rounds, state.threadCount, dispatch],
  );

  const completedCount = Object.keys(state.results).length + Object.keys(state.errors).length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-gray-100">
          Matrix Multiplication Benchmark
        </h1>
        <p className="mt-1 text-sm text-gray-400">
          Compare JavaScript, WebAssembly SIMD, and multi-threaded WASM implementations.
          All benchmarks run in your browser using double-precision (f64) arithmetic.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-6 rounded-lg border border-gray-800 bg-gray-900/50 p-4">
        <MatrixSizePicker
          value={state.matrixSize}
          onChange={(size) => dispatch({ type: 'SET_SIZE', payload: size })}
          isDisabled={state.globalRunning}
        />
        <RoundsPicker
          value={state.rounds}
          onChange={(rounds) => dispatch({ type: 'SET_ROUNDS', payload: rounds })}
          isDisabled={state.globalRunning}
        />
        <ThreadCountPicker
          value={state.threadCount}
          onChange={(threads) => dispatch({ type: 'SET_THREADS', payload: threads })}
          isDisabled={state.globalRunning}
        />
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium tracking-wide text-gray-400 uppercase opacity-0">Action</span>
          <RunAllButton
            onPress={handleRunAll}
            isRunning={state.globalRunning}
            progress={state.globalRunning ? { current: completedCount, total: BENCHMARKS.length } : undefined}
          />
        </div>
      </div>

      <BenchmarkTable
        benchmarks={BENCHMARKS}
        results={state.results}
        errors={state.errors}
        runningId={state.runningId}
        isAnyRunning={state.globalRunning}
        onRunSingle={handleRunSingle}
      />
    </div>
  );
}
