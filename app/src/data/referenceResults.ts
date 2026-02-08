export interface ReferenceDataPoint {
  size: number;
  scalar: { numpy: number; openblas: number };
  sse: { numpy: number; openblas: number };
  avx2: { numpy: number; openblas: number; matlab: number; mkl: number };
  avx512: { numpy: number; openblas: number; mex: number };
}

export interface ThreadResult {
  threadCount: number;
  label: string;
  data: ReferenceDataPoint[];
}

export interface ArchitectureNote {
  arch: string;
  description: string;
}

export interface ReferenceResultSet {
  hardware: string;
  notes: ArchitectureNote[];
  results: ThreadResult[];
}

/** NumPy (n) and native OpenBLAS (o) GFLOPS side by side */
function d(n: number, o: number) {
  return { numpy: n, openblas: o };
}

/** NumPy (n), native OpenBLAS (o), MATLAB/MKL (m), and direct MKL (k) GFLOPS */
function dm(n: number, o: number, m: number, k: number) {
  return { numpy: n, openblas: o, matlab: m, mkl: k };
}

/** NumPy (n), native OpenBLAS (o), and custom AVX-512 MEX (x) GFLOPS */
function dx(n: number, o: number, x: number) {
  return { numpy: n, openblas: o, mex: x };
}

export const RYZEN_9950X_RESULTS: ReferenceResultSet = {
  hardware: 'AMD Ryzen 9 9950X',
  notes: [
    { arch: 'Scalar', description: 'PRESCOTT target with -fno-tree-vectorize' },
    { arch: 'SSE', description: 'NEHALEM target (128-bit XMM registers)' },
    { arch: 'AVX2', description: 'HASWELL target (256-bit YMM registers + FMA)' },
    { arch: 'AVX-512', description: 'COOPERLAKE target (512-bit ZMM registers)' },
  ],
  results: [
    {
      threadCount: 1,
      label: 'Single-threaded',
      data: [
        { size: 64,   scalar: d(23.3, 10.8), sse: d(25.0, 20.3), avx2: dm(62.2, 37.4, 5.2, 38.9), avx512: dx(106.6, 122.3, 11.9) },
        { size: 128,  scalar: d(31.9, 23.2), sse: d(31.1, 23.2), avx2: dm(72.3, 55.6, 25.3, 52.9), avx512: dx(97.2, 92.7, 47.7) },
        { size: 256,  scalar: d(29.4, 24.0), sse: d(28.9, 23.9), avx2: dm(76.1, 59.7, 48.3, 60.0), avx512: dx(103.7, 99.6, 103.4) },
        { size: 512,  scalar: d(30.7, 29.2), sse: d(31.3, 31.8), avx2: dm(78.4, 79.7, 60.3, 64.8), avx512: dx(119.3, 123.3, 115.7) },
        { size: 1024, scalar: d(30.9, 31.3), sse: d(31.9, 31.5), avx2: dm(79.4, 77.9, 72.8, 76.4), avx512: dx(131.3, 131.0, 145.5) },
        { size: 2048, scalar: d(31.1, 30.1), sse: d(32.0, 32.3), avx2: dm(77.7, 81.3, 74.7, 78.9), avx512: dx(125.4, 135.9, 133.4) },
        { size: 4096, scalar: d(31.1, 31.2), sse: d(32.3, 32.8), avx2: dm(80.7, 81.6, 79.0, 79.4), avx512: dx(132.7, 138.7, 128.7) },
      ],
    },
    {
      threadCount: 2,
      label: 'Multi-threaded (2 threads)',
      data: [
        { size: 64,   scalar: d(26.4, 23.4), sse: d(27.0, 25.8), avx2: dm(62.5, 49.9, 4.0, 37.5), avx512: dx(104.4, 165.6, 5.8) },
        { size: 128,  scalar: d(51.9, 45.4), sse: d(59.7, 52.9), avx2: dm(130.2, 107.3, 29.2, 73.1), avx512: dx(182.2, 167.4, 42.9) },
        { size: 256,  scalar: d(56.3, 60.7), sse: d(55.4, 58.6), avx2: dm(123.6, 132.5, 56.6, 109.2), avx512: dx(163.5, 113.9, 146.4) },
        { size: 512,  scalar: d(55.3, 60.5), sse: d(56.8, 62.8), avx2: dm(154.2, 134.7, 123.2, 125.2), avx512: dx(220.7, 221.7, 212.0) },
        { size: 1024, scalar: d(60.3, 60.8), sse: d(63.2, 63.8), avx2: dm(155.5, 158.3, 138.1, 148.8), avx512: dx(248.5, 263.7, 268.1) },
        { size: 2048, scalar: d(60.1, 59.7), sse: d(63.1, 64.2), avx2: dm(154.9, 159.5, 148.3, 150.9), avx512: dx(247.1, 265.8, 248.7) },
        { size: 4096, scalar: d(59.6, 60.7), sse: d(64.2, 64.6), avx2: dm(160.0, 161.3, 154.8, 151.5), avx512: dx(261.1, 270.0, 256.3) },
      ],
    },
    {
      threadCount: 4,
      label: 'Multi-threaded (4 threads)',
      data: [
        { size: 64,   scalar: d(28.3, 25.5), sse: d(28.1, 25.3), avx2: dm(60.7, 51.8, 4.5, 46.7), avx512: dx(112.2, 152.4, 8.2) },
        { size: 128,  scalar: d(112.6, 77.3), sse: d(100.0, 78.7), avx2: dm(176.3, 170.3, 30.8, 105.7), avx512: dx(303.8, 257.3, 53.1) },
        { size: 256,  scalar: d(108.5, 112.5), sse: d(96.1, 92.7), avx2: dm(202.2, 230.5, 145.6, 201.2), avx512: dx(280.2, 293.8, 202.9) },
        { size: 512,  scalar: d(115.0, 114.9), sse: d(117.4, 118.8), avx2: dm(250.2, 261.4, 216.4, 199.0), avx512: dx(413.6, 438.4, 401.5) },
        { size: 1024, scalar: d(117.9, 120.6), sse: d(122.9, 124.7), avx2: dm(295.0, 310.8, 244.9, 290.5), avx512: dx(489.7, 499.8, 480.0) },
        { size: 2048, scalar: d(113.3, 118.8), sse: d(97.5, 125.6), avx2: dm(293.1, 311.2, 273.8, 271.0), avx512: dx(460.9, 508.6, 404.8) },
        { size: 4096, scalar: d(121.0, 121.1), sse: d(126.9, 126.9), avx2: dm(309.5, 309.5, 300.7, 285.2), avx512: dx(508.8, 515.8, 466.8) },
      ],
    },
    {
      threadCount: 8,
      label: 'Multi-threaded (8 threads)',
      data: [
        { size: 64,   scalar: d(27.8, 26.9), sse: d(28.0, 25.3), avx2: dm(61.7, 51.5, 4.9, 46.6), avx512: dx(104.4, 154.4, 9.1) },
        { size: 128,  scalar: d(154.6, 123.1), sse: d(158.2, 119.3), avx2: dm(356.1, 201.8, 35.1, 105.5), avx512: dx(291.6, 215.4, 71.6) },
        { size: 256,  scalar: d(155.4, 187.3), sse: d(162.7, 190.4), avx2: dm(263.0, 319.7, 184.8, 242.4), avx512: dx(323.7, 428.4, 203.4) },
        { size: 512,  scalar: d(221.9, 215.1), sse: d(206.4, 214.0), avx2: dm(523.1, 440.8, 353.3, 364.8), avx512: dx(769.7, 676.7, 265.5) },
        { size: 1024, scalar: d(233.8, 224.2), sse: d(240.2, 222.7), avx2: dm(545.8, 531.1, 374.6, 484.0), avx512: dx(921.1, 962.8, 756.5) },
        { size: 2048, scalar: d(190.7, 226.3), sse: d(187.6, 238.7), avx2: dm(538.4, 574.6, 459.6, 476.1), avx512: dx(795.5, 900.4, 603.7) },
        { size: 4096, scalar: d(224.7, 221.0), sse: d(238.2, 241.7), avx2: dm(559.3, 582.3, 518.2, 503.7), avx512: dx(868.8, 951.7, 760.7) },
      ],
    },
    {
      threadCount: 16,
      label: 'Multi-threaded (16 threads)',
      data: [
        { size: 64,   scalar: d(26.6, 24.0), sse: d(27.6, 26.8), avx2: dm(58.8, 49.8, 4.5, 55.2), avx512: dx(108.8, 144.9, 8.5) },
        { size: 128,  scalar: d(149.9, 97.3), sse: d(143.6, 106.0), avx2: dm(214.1, 57.6, 30.1, 209.1), avx512: dx(280.7, 188.2, 55.6) },
        { size: 256,  scalar: d(122.2, 185.4), sse: d(123.3, 189.6), avx2: dm(151.4, 156.2, 141.0, 300.7), avx512: dx(172.3, 124.3, 241.8) },
        { size: 512,  scalar: d(303.8, 230.8), sse: d(286.9, 349.6), avx2: dm(485.5, 625.3, 323.7, 557.7), avx512: dx(567.0, 602.1, 220.6) },
        { size: 1024, scalar: d(398.3, 401.2), sse: d(381.1, 417.6), avx2: dm(801.3, 857.4, 470.3, 529.8), avx512: dx(1042.3, 845.7, 691.7) },
        { size: 2048, scalar: d(379.1, 407.2), sse: d(333.4, 379.0), avx2: dm(808.8, 963.6, 688.1, 857.3), avx512: dx(1040.7, 1205.0, 882.3) },
        { size: 4096, scalar: d(407.6, 339.9), sse: d(401.4, 431.5), avx2: dm(930.1, 964.5, 750.9, 830.8), avx512: dx(1255.1, 1291.4, 971.7) },
      ],
    },
  ],
};
