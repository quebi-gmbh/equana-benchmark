export interface ReferenceDataPoint {
  size: number;
  scalar: { numpy: number; openblas: number };
  sse: { numpy: number; openblas: number };
  avx2: { numpy: number; openblas: number };
  avx512: { numpy: number; openblas: number; matlab: number };
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

/** NumPy (n), native OpenBLAS (o), and MATLAB/MKL (m) GFLOPS */
function dm(n: number, o: number, m: number) {
  return { numpy: n, openblas: o, matlab: m };
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
        { size: 64,   scalar: d(23.3, 10.8), sse: d(25.0, 20.3), avx2: d(62.2, 37.4), avx512: dm(106.6, 122.3, 5.3) },
        { size: 128,  scalar: d(31.9, 23.2), sse: d(31.1, 23.2), avx2: d(72.3, 55.6), avx512: dm(97.2, 92.7, 26.3) },
        { size: 256,  scalar: d(29.4, 24.0), sse: d(28.9, 23.9), avx2: d(76.1, 59.7), avx512: dm(103.7, 99.6, 47.9) },
        { size: 512,  scalar: d(30.7, 29.2), sse: d(31.3, 31.8), avx2: d(78.4, 79.7), avx512: dm(119.3, 123.3, 69.2) },
        { size: 1024, scalar: d(30.9, 31.3), sse: d(31.9, 31.5), avx2: d(79.4, 77.9), avx512: dm(131.3, 131.0, 73.4) },
        { size: 2048, scalar: d(31.1, 30.1), sse: d(32.0, 32.3), avx2: d(77.7, 81.3), avx512: dm(125.4, 135.9, 77.3) },
        { size: 4096, scalar: d(31.1, 31.2), sse: d(32.3, 32.8), avx2: d(80.7, 81.6), avx512: dm(132.7, 138.7, 80.6) },
      ],
    },
    {
      threadCount: 2,
      label: 'Multi-threaded (2 threads)',
      data: [
        { size: 64,   scalar: d(26.4, 23.4), sse: d(27.0, 25.8), avx2: d(62.5, 49.9), avx512: dm(104.4, 165.6, 5.0) },
        { size: 128,  scalar: d(51.9, 45.4), sse: d(59.7, 52.9), avx2: d(130.2, 107.3), avx512: dm(182.2, 167.4, 33.9) },
        { size: 256,  scalar: d(56.3, 60.7), sse: d(55.4, 58.6), avx2: d(123.6, 132.5), avx512: dm(163.5, 113.9, 80.7) },
        { size: 512,  scalar: d(55.3, 60.5), sse: d(56.8, 62.8), avx2: d(154.2, 134.7), avx512: dm(220.7, 221.7, 122.4) },
        { size: 1024, scalar: d(60.3, 60.8), sse: d(63.2, 63.8), avx2: d(155.5, 158.3), avx512: dm(248.5, 263.7, 141.6) },
        { size: 2048, scalar: d(60.1, 59.7), sse: d(63.1, 64.2), avx2: d(154.9, 159.5), avx512: dm(247.1, 265.8, 149.8) },
        { size: 4096, scalar: d(59.6, 60.7), sse: d(64.2, 64.6), avx2: d(160.0, 161.3), avx512: dm(261.1, 270.0, 155.5) },
      ],
    },
    {
      threadCount: 4,
      label: 'Multi-threaded (4 threads)',
      data: [
        { size: 64,   scalar: d(28.3, 25.5), sse: d(28.1, 25.3), avx2: d(60.7, 51.8), avx512: dm(112.2, 152.4, 4.9) },
        { size: 128,  scalar: d(112.6, 77.3), sse: d(100.0, 78.7), avx2: d(176.3, 170.3), avx512: dm(303.8, 257.3, 33.0) },
        { size: 256,  scalar: d(108.5, 112.5), sse: d(96.1, 92.7), avx2: d(202.2, 230.5), avx512: dm(280.2, 293.8, 132.1) },
        { size: 512,  scalar: d(115.0, 114.9), sse: d(117.4, 118.8), avx2: d(250.2, 261.4), avx512: dm(413.6, 438.4, 213.8) },
        { size: 1024, scalar: d(117.9, 120.6), sse: d(122.9, 124.7), avx2: d(295.0, 310.8), avx512: dm(489.7, 499.8, 190.7) },
        { size: 2048, scalar: d(113.3, 118.8), sse: d(97.5, 125.6), avx2: d(293.1, 311.2), avx512: dm(460.9, 508.6, 278.7) },
        { size: 4096, scalar: d(121.0, 121.1), sse: d(126.9, 126.9), avx2: d(309.5, 309.5), avx512: dm(508.8, 515.8, 297.5) },
      ],
    },
    {
      threadCount: 8,
      label: 'Multi-threaded (8 threads)',
      data: [
        { size: 64,   scalar: d(27.8, 26.9), sse: d(28.0, 25.3), avx2: d(61.7, 51.5), avx512: dm(104.4, 154.4, 4.6) },
        { size: 128,  scalar: d(154.6, 123.1), sse: d(158.2, 119.3), avx2: d(356.1, 201.8), avx512: dm(291.6, 215.4, 37.3) },
        { size: 256,  scalar: d(155.4, 187.3), sse: d(162.7, 190.4), avx2: d(263.0, 319.7), avx512: dm(323.7, 428.4, 126.1) },
        { size: 512,  scalar: d(221.9, 215.1), sse: d(206.4, 214.0), avx2: d(523.1, 440.8), avx512: dm(769.7, 676.7, 208.2) },
        { size: 1024, scalar: d(233.8, 224.2), sse: d(240.2, 222.7), avx2: d(545.8, 531.1), avx512: dm(921.1, 962.8, 238.0) },
        { size: 2048, scalar: d(190.7, 226.3), sse: d(187.6, 238.7), avx2: d(538.4, 574.6), avx512: dm(795.5, 900.4, 441.9) },
        { size: 4096, scalar: d(224.7, 221.0), sse: d(238.2, 241.7), avx2: d(559.3, 582.3), avx512: dm(868.8, 951.7, 464.0) },
      ],
    },
    {
      threadCount: 16,
      label: 'Multi-threaded (16 threads)',
      data: [
        { size: 64,   scalar: d(26.6, 24.0), sse: d(27.6, 26.8), avx2: d(58.8, 49.8), avx512: dm(108.8, 144.9, 3.8) },
        { size: 128,  scalar: d(149.9, 97.3), sse: d(143.6, 106.0), avx2: d(214.1, 57.6), avx512: dm(280.7, 188.2, 33.1) },
        { size: 256,  scalar: d(122.2, 185.4), sse: d(123.3, 189.6), avx2: d(151.4, 156.2), avx512: dm(172.3, 124.3, 86.0) },
        { size: 512,  scalar: d(303.8, 230.8), sse: d(286.9, 349.6), avx2: d(485.5, 625.3), avx512: dm(567.0, 602.1, 243.6) },
        { size: 1024, scalar: d(398.3, 401.2), sse: d(381.1, 417.6), avx2: d(801.3, 857.4), avx512: dm(1042.3, 845.7, 329.9) },
        { size: 2048, scalar: d(379.1, 407.2), sse: d(333.4, 379.0), avx2: d(808.8, 963.6), avx512: dm(1040.7, 1205.0, 464.9) },
        { size: 4096, scalar: d(407.6, 339.9), sse: d(401.4, 431.5), avx2: d(930.1, 964.5), avx512: dm(1255.1, 1291.4, 595.7) },
      ],
    },
  ],
};
