// Accuracy Utilities for Matrix Multiplication Comparison

/**
 * Compute average relative error between reference and computed matrices
 * Error metric: (1/N²) × Σ|2(a-b)/(a+b)|
 *
 * @param {Float64Array} ref_f64 - Reference matrix (f64)
 * @param {Float32Array|Float64Array} result - Computed result
 * @param {number} componentsPerElement - 1 for f32, 2 for df64 (hi+lo), 3 for tf64
 * @returns {number} Average relative error
 */
export function computeRelativeError(ref_f64, result, componentsPerElement = 1) {
    const size = ref_f64.length;
    let totalError = 0;
    let validCount = 0;

    for (let i = 0; i < size; i++) {
        let computed;
        if (componentsPerElement === 1) {
            computed = result[i];
        } else if (componentsPerElement === 2) {
            // df64: hi + lo
            computed = result[i * 2] + result[i * 2 + 1];
        } else if (componentsPerElement === 3) {
            // tf64: hi + mid + lo
            computed = result[i * 3] + result[i * 3 + 1] + result[i * 3 + 2];
        }

        const a = ref_f64[i];
        const b = computed;
        const sum = a + b;

        // Avoid division by zero
        if (Math.abs(sum) > 1e-300) {
            totalError += Math.abs(2 * (a - b) / sum);
            validCount++;
        }
    }

    return validCount > 0 ? totalError / validCount : 0;
}

/**
 * Compute maximum absolute error
 */
export function computeMaxError(ref_f64, result, componentsPerElement = 1) {
    const size = ref_f64.length;
    let maxError = 0;

    for (let i = 0; i < size; i++) {
        let computed;
        if (componentsPerElement === 1) {
            computed = result[i];
        } else if (componentsPerElement === 2) {
            computed = result[i * 2] + result[i * 2 + 1];
        } else if (componentsPerElement === 3) {
            computed = result[i * 3] + result[i * 3 + 1] + result[i * 3 + 2];
        }

        const error = Math.abs(ref_f64[i] - computed);
        if (error > maxError) {
            maxError = error;
        }
    }

    return maxError;
}

/**
 * Split f64 value into hi + lo (double-float representation)
 */
export function splitF64(value) {
    const hi = Math.fround(value);
    const lo = Math.fround(value - hi);
    return { hi, lo };
}

/**
 * Split f64 value into hi + mid + lo (triple-float representation)
 */
export function splitF64Triple(value) {
    const hi = Math.fround(value);
    const r1 = value - hi;
    const mid = Math.fround(r1);
    const lo = Math.fround(r1 - mid);
    return { hi, mid, lo };
}

/**
 * Split Float64Array into two Float32Arrays (hi and lo components)
 */
export function splitMatrixDF64(f64Array) {
    const size = f64Array.length;
    const hi = new Float32Array(size);
    const lo = new Float32Array(size);

    for (let i = 0; i < size; i++) {
        hi[i] = Math.fround(f64Array[i]);
        lo[i] = Math.fround(f64Array[i] - hi[i]);
    }

    return { hi, lo };
}

/**
 * Assemble hi and lo Float32Arrays back into Float64Array
 */
export function assembleDF64(hi, lo) {
    const size = hi.length;
    const result = new Float64Array(size);

    for (let i = 0; i < size; i++) {
        result[i] = hi[i] + lo[i];
    }

    return result;
}

/**
 * Format a number in scientific notation with specified precision
 */
export function formatScientific(value, precision = 4) {
    return value.toExponential(precision);
}
