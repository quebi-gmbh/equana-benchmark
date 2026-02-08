import { useReducer } from 'react';
import type { BenchmarkState, BenchmarkAction } from '../engine/types';

const initialState: BenchmarkState = {
  matrixSize: 512,
  rounds: 3,
  threadCount: typeof navigator !== 'undefined' ? (navigator.hardwareConcurrency ?? 4) : 4,
  results: {},
  errors: {},
  runningId: null,
  globalRunning: false,
};

function benchmarkReducer(state: BenchmarkState, action: BenchmarkAction): BenchmarkState {
  switch (action.type) {
    case 'SET_SIZE':
      return { ...state, matrixSize: action.payload, results: {}, errors: {} };
    case 'SET_ROUNDS':
      return { ...state, rounds: action.payload };
    case 'SET_THREADS':
      return { ...state, threadCount: action.payload, results: {}, errors: {} };
    case 'SET_RUNNING':
      return { ...state, runningId: action.payload, globalRunning: true };
    case 'SET_RESULT':
      return {
        ...state,
        runningId: null,
        results: { ...state.results, [action.payload.id]: action.payload.result },
      };
    case 'SET_ERROR':
      return {
        ...state,
        runningId: null,
        errors: { ...state.errors, [action.payload.id]: action.payload.error },
      };
    case 'SET_IDLE':
      return { ...state, runningId: null, globalRunning: false };
  }
}

export function useBenchmarkStore() {
  return useReducer(benchmarkReducer, initialState);
}
