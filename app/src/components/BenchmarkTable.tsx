import type { BenchmarkVariant, BenchmarkResult } from '../engine/types';
import { BenchmarkRow } from './BenchmarkRow';

interface BenchmarkTableProps {
  benchmarks: BenchmarkVariant[];
  results: Record<string, BenchmarkResult>;
  errors: Record<string, string>;
  runningId: string | null;
  isAnyRunning: boolean;
  onRunSingle: (variant: BenchmarkVariant) => void;
}

export function BenchmarkTable({
  benchmarks,
  results,
  errors,
  runningId,
  isAnyRunning,
  onRunSingle,
}: BenchmarkTableProps) {
  const baselineResult = results[benchmarks[0]?.id ?? ''];
  const baselineAvg = baselineResult?.avg;

  const stBenchmarks = benchmarks.filter((b) => b.category !== 'wasm-mt' && b.category !== 'pyodide');
  const mtBenchmarks = benchmarks.filter((b) => b.category === 'wasm-mt');
  const pyodideBenchmarks = benchmarks.filter((b) => b.category === 'pyodide');

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-700">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-gray-600/50 bg-gray-800/80">
            <th className="sticky top-0 bg-gray-900/80 px-3 py-2.5 text-left text-xs font-medium tracking-wide text-gray-400 uppercase">#</th>
            <th className="sticky top-0 bg-gray-900/80 px-3 py-2.5 text-left text-xs font-medium tracking-wide text-gray-400 uppercase">Name</th>
            <th className="sticky top-0 bg-gray-900/80 px-3 py-2.5 text-left text-xs font-medium tracking-wide text-gray-400 uppercase">Type</th>
            <th className="sticky top-0 bg-gray-900/80 px-3 py-2.5 text-left text-xs font-medium tracking-wide text-gray-400 uppercase">Description</th>
            <th className="sticky top-0 bg-gray-900/80 px-3 py-2.5 text-right text-xs font-medium tracking-wide text-gray-400 uppercase">Time (ms)</th>
            <th className="sticky top-0 bg-gray-900/80 px-3 py-2.5 text-right text-xs font-medium tracking-wide text-gray-400 uppercase">GFLOPS</th>
            <th className="sticky top-0 bg-gray-900/80 px-3 py-2.5 text-right text-xs font-medium tracking-wide text-gray-400 uppercase">vs Baseline</th>
            <th className="sticky top-0 bg-gray-900/80 px-3 py-2.5 text-left text-xs font-medium tracking-wide text-gray-400 uppercase">Action</th>
          </tr>
        </thead>
        <tbody>
          {pyodideBenchmarks.length > 0 && (
            <tr className="border-b border-gray-700 bg-gray-800/40">
              <td colSpan={8} className="px-3 py-2 text-xs font-semibold tracking-wide text-purple-400 uppercase">
                Python / Pyodide (NumPy via WebAssembly)
              </td>
            </tr>
          )}

          {pyodideBenchmarks.map((variant, i) => (
            <BenchmarkRow
              key={variant.id}
              index={i}
              variant={variant}
              result={results[variant.id]}
              error={errors[variant.id]}
              isRunning={runningId === variant.id}
              isAnyRunning={isAnyRunning}
              baselineAvg={baselineAvg}
              onRun={() => onRunSingle(variant)}
            />
          ))}

          {stBenchmarks.map((variant, i) => (
            <BenchmarkRow
              key={variant.id}
              index={pyodideBenchmarks.length + i}
              variant={variant}
              result={results[variant.id]}
              error={errors[variant.id]}
              isRunning={runningId === variant.id}
              isAnyRunning={isAnyRunning}
              baselineAvg={baselineAvg}
              onRun={() => onRunSingle(variant)}
            />
          ))}

          {mtBenchmarks.length > 0 && (
            <tr className="border-b border-gray-700 bg-gray-800/40">
              <td colSpan={8} className="px-3 py-2 text-xs font-semibold tracking-wide text-emerald-400 uppercase">
                Multi-Threaded (pthreads / SharedArrayBuffer)
              </td>
            </tr>
          )}

          {mtBenchmarks.map((variant, i) => (
            <BenchmarkRow
              key={variant.id}
              index={pyodideBenchmarks.length + stBenchmarks.length + i}
              variant={variant}
              result={results[variant.id]}
              error={errors[variant.id]}
              isRunning={runningId === variant.id}
              isAnyRunning={isAnyRunning}
              baselineAvg={baselineAvg}
              onRun={() => onRunSingle(variant)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}
