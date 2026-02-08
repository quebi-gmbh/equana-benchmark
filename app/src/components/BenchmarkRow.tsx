import { Button } from 'react-aria-components';
import type { BenchmarkVariant, BenchmarkResult } from '../engine/types';
import { StatusBadge } from './StatusBadge';

interface BenchmarkRowProps {
  index: number;
  variant: BenchmarkVariant;
  result: BenchmarkResult | undefined;
  error: string | undefined;
  isRunning: boolean;
  isAnyRunning: boolean;
  baselineAvg: number | undefined;
  onRun: () => void;
}

const categoryStyles: Record<string, string> = {
  javascript: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  wasm: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  'wasm-mt': 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
};

const categoryLabels: Record<string, string> = {
  javascript: 'JS',
  wasm: 'WASM',
  'wasm-mt': 'MT',
};

export function BenchmarkRow({
  index,
  variant,
  result,
  error,
  isRunning,
  isAnyRunning,
  baselineAvg,
  onRun,
}: BenchmarkRowProps) {
  const status = isRunning ? 'running' : error ? 'error' : result ? 'done' : 'idle';
  const speedup = result && baselineAvg ? baselineAvg / result.avg : undefined;

  return (
    <tr className={`border-b border-gray-800/50 transition-colors ${isRunning ? 'bg-blue-500/5' : 'hover:bg-gray-800/30'}`}>
      <td className="px-3 py-2.5 text-sm text-gray-500 font-mono">{index + 1}</td>
      <td className="px-3 py-2.5">
        <div className="flex items-center gap-2">
          <StatusBadge status={status} />
          <span className="text-sm font-medium text-gray-200">{variant.name}</span>
        </div>
      </td>
      <td className="px-3 py-2.5">
        <span className={`inline-block rounded border px-1.5 py-0.5 text-xs font-medium ${categoryStyles[variant.category]}`}>
          {categoryLabels[variant.category]}
        </span>
      </td>
      <td className="px-3 py-2.5 text-sm text-gray-400 max-w-xs">
        {variant.description}
      </td>
      <td className={`px-3 py-2.5 text-sm font-mono tabular-nums text-right ${isRunning ? 'animate-pulse text-blue-400' : 'text-gray-200'}`}>
        {isRunning ? '...' : result ? result.avg.toFixed(2) : error ? '—' : ''}
      </td>
      <td className="px-3 py-2.5 text-sm font-mono tabular-nums text-right text-gray-200">
        {result ? result.gflops.toFixed(2) : ''}
      </td>
      <td className="px-3 py-2.5 text-sm font-mono tabular-nums text-right">
        {speedup !== undefined ? (
          <span className={speedup >= 10 ? 'text-emerald-400 font-semibold' : speedup >= 3 ? 'text-blue-400' : 'text-gray-300'}>
            {speedup.toFixed(1)}x
          </span>
        ) : ''}
      </td>
      <td className="px-3 py-2.5">
        {error ? (
          <span className="text-xs text-red-400 truncate max-w-[120px] inline-block" title={error}>
            {error}
          </span>
        ) : (
          <Button
            onPress={onRun}
            isDisabled={isAnyRunning}
            className="rounded-md bg-gray-800 px-2.5 py-1 text-xs font-medium text-gray-300 transition-colors
              hover:bg-gray-700 hover:text-white
              data-[disabled]:opacity-30 data-[disabled]:cursor-not-allowed
              data-[focus-visible]:ring-2 data-[focus-visible]:ring-blue-400"
          >
            Run
          </Button>
        )}
      </td>
    </tr>
  );
}
