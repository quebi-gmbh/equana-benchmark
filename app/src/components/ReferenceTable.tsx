import type { ReferenceDataPoint } from '../data/referenceResults';

interface ReferenceTableProps {
  data: ReferenceDataPoint[];
}

export function ReferenceTable({ data }: ReferenceTableProps) {
  const allValues = data.flatMap((d) => [
    d.scalar.numpy, d.scalar.openblas,
    d.sse.numpy, d.sse.openblas,
    d.avx2.numpy, d.avx2.openblas,
    d.avx512.numpy, d.avx512.openblas, d.avx512.matlab,
  ]);
  const peak = Math.max(...allValues);

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-800">
      <table className="w-full border-collapse">
        <thead>
          <tr className="border-b border-gray-700 bg-gray-900/80">
            <th className="px-3 py-2.5 text-left text-xs font-medium tracking-wide text-gray-400 uppercase" rowSpan={2}>Size</th>
            <th className="px-3 py-2.5 text-center text-xs font-medium tracking-wide text-gray-400 uppercase" colSpan={2}>Scalar (GFLOPS)</th>
            <th className="px-3 py-2.5 text-center text-xs font-medium tracking-wide text-gray-400 uppercase" colSpan={2}>SSE (GFLOPS)</th>
            <th className="px-3 py-2.5 text-center text-xs font-medium tracking-wide text-gray-400 uppercase" colSpan={2}>AVX2 (GFLOPS)</th>
            <th className="px-3 py-2.5 text-center text-xs font-medium tracking-wide text-gray-400 uppercase" colSpan={3}>AVX-512 (GFLOPS)</th>
          </tr>
          <tr className="border-b border-gray-700 bg-gray-900/80">
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-purple-400">NumPy</th>
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-blue-400">C</th>
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-purple-400">NumPy</th>
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-blue-400">C</th>
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-purple-400">NumPy</th>
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-blue-400">C</th>
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-purple-400">NumPy</th>
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-blue-400">C</th>
            <th className="px-2 py-1.5 text-right text-[10px] font-medium text-orange-400">MATLAB</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.size} className="border-b border-gray-800/50 hover:bg-gray-800/30">
              <td className="px-3 py-2.5 text-sm font-mono text-gray-300">{row.size}</td>
              <GflopsCell value={row.scalar.numpy} peak={peak} variant="numpy" />
              <GflopsCell value={row.scalar.openblas} peak={peak} variant="openblas" />
              <GflopsCell value={row.sse.numpy} peak={peak} variant="numpy" />
              <GflopsCell value={row.sse.openblas} peak={peak} variant="openblas" />
              <GflopsCell value={row.avx2.numpy} peak={peak} variant="numpy" />
              <GflopsCell value={row.avx2.openblas} peak={peak} variant="openblas" />
              <GflopsCell value={row.avx512.numpy} peak={peak} variant="numpy" />
              <GflopsCell value={row.avx512.openblas} peak={peak} variant="openblas" />
              <GflopsCell value={row.avx512.matlab} peak={peak} variant="matlab" />
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GflopsCell({
  value,
  peak,
  variant,
}: {
  value: number;
  peak: number;
  variant: 'numpy' | 'openblas' | 'matlab';
}) {
  const isPeak = value === peak && peak > 0;
  const borderClass = variant === 'openblas' && 'border-r border-gray-800/30';
  return (
    <td
      className={`px-2 py-2.5 text-right text-sm font-mono tabular-nums ${borderClass || ''} ${
        isPeak ? 'font-semibold text-emerald-400' : 'text-gray-200'
      }`}
    >
      {value.toFixed(1)}
    </td>
  );
}
