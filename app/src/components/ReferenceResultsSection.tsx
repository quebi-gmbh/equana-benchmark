import { ReferenceTable } from './ReferenceTable';
import { RYZEN_9950X_RESULTS } from '../data/referenceResults';

export function ReferenceResultsSection() {
  const ref = RYZEN_9950X_RESULTS;

  return (
    <section className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-blue-400">
          Reference Results: NumPy vs Native OpenBLAS vs MATLAB/MKL DGEMM
        </h2>
        <p className="mt-1 text-sm text-gray-400">
          NumPy (OpenBLAS-backed), native C/OpenBLAS, and MATLAB/MKL DGEMM performance across SIMD targets,
          measured on an {ref.hardware}. MATLAB/MKL results shown in the AVX-512 column only (MKL auto-selects SIMD target).
        </p>
      </div>

      <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-3">
        <p className="text-sm text-amber-400">
          <span className="font-semibold">Note:</span> These results were measured on a {ref.hardware}.
          Native performance varies significantly by CPU architecture and generation.
          Clone the repo and run the benchmark on your own hardware for an accurate comparison.
        </p>
      </div>

      <div className="space-y-8">
        {ref.results.map((r) => (
          <div key={r.threadCount} className="space-y-2">
            <h3 className="text-sm font-semibold text-gray-300">
              {r.label}{' '}
              <span className="font-normal text-gray-500">
                (OPENBLAS_NUM_THREADS={r.threadCount})
              </span>
            </h3>
            <ReferenceTable data={r.data} />
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-gray-800 bg-gray-900/30 p-4 text-sm text-gray-400">
        <span className="font-medium text-gray-300">Architecture targets:</span>
        <ul className="mt-2 space-y-1">
          {ref.notes.map((note) => (
            <li key={note.arch}>
              <span className="font-mono text-gray-300">{note.arch}</span> &mdash; {note.description}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
