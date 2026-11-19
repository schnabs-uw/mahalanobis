import { describe, expect, test } from 'vitest';
import {
    type Point,
    calculateEigenvectors,
    calculateEllipse,
    calculateMahalanobisDistance,
    calculateMeanAndCovariance,
} from './math';

// Worked by hand: mean (2, 1); covariance divided by n = 4 is [[2, 1], [1, 1]];
// its inverse is [[1, -1], [-1, 2]]; its eigenvalues are (3 ± √5) / 2.
const tilted: Point[] = [
    { x: 0, y: 0 },
    { x: 2, y: 2 },
    { x: 2, y: 0 },
    { x: 4, y: 2 },
];

function stats(points: Point[]) {
    const result = calculateMeanAndCovariance(points);
    if (!result) throw new Error('expected at least 2 points');
    return result;
}

describe('calculateMeanAndCovariance', () => {
    test('returns null for fewer than 2 points', () => {
        expect(calculateMeanAndCovariance([])).toBeNull();
        expect(calculateMeanAndCovariance([{ x: 1, y: 2 }])).toBeNull();
    });

    test('returns the mean and the covariance divided by n', () => {
        const { mean, covariance } = stats(tilted);
        expect(mean).toEqual({ x: 2, y: 1 });
        expect(covariance[0][0]).toBeCloseTo(2, 5);
        expect(covariance[0][1]).toBeCloseTo(1, 5);
        expect(covariance[1][0]).toBeCloseTo(1, 5);
        expect(covariance[1][1]).toBeCloseTo(1, 5);
    });

    test('keeps points on a straight line usable for distances', () => {
        const line = stats([{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }]);
        expect(Number.isFinite(calculateMahalanobisDistance({ x: 1, y: 0 }, line))).toBe(true);
    });
});

describe('calculateMahalanobisDistance', () => {
    const s = stats(tilted);

    test('is 0 at the mean', () => {
        expect(calculateMahalanobisDistance({ x: 2, y: 1 }, s)).toBeCloseTo(0, 5);
    });

    test('matches hand-computed distances', () => {
        expect(calculateMahalanobisDistance({ x: 3, y: 1 }, s)).toBeCloseTo(1, 5);
        expect(calculateMahalanobisDistance({ x: 2, y: 2 }, s)).toBeCloseTo(Math.SQRT2, 5);
    });

    test('gives different distances to points the same Euclidean distance from the mean', () => {
        // Both points are 2 units from the mean; the cloud is more spread out along x.
        expect(calculateMahalanobisDistance({ x: 4, y: 1 }, s)).toBeCloseTo(2, 5);
        expect(calculateMahalanobisDistance({ x: 2, y: 3 }, s)).toBeCloseTo(2 * Math.SQRT2, 5);
    });
});

describe('calculateEigenvectors', () => {
    const matrix = [[2, 1], [1, 1]];
    const { values, vectors } = calculateEigenvectors(matrix);
    const column = (i: number) => [vectors[0][i], vectors[1][i]];

    test('returns the eigenvalues, largest first', () => {
        expect(values[0]).toBeCloseTo((3 + Math.sqrt(5)) / 2, 6);
        expect(values[1]).toBeCloseTo((3 - Math.sqrt(5)) / 2, 6);
    });

    test('returns unit eigenvectors as columns, with A·v = λ·v', () => {
        for (const i of [0, 1]) {
            const [vx, vy] = column(i);
            expect(Math.hypot(vx, vy)).toBeCloseTo(1, 6);
            expect(matrix[0][0] * vx + matrix[0][1] * vy).toBeCloseTo(values[i] * vx, 6);
            expect(matrix[1][0] * vx + matrix[1][1] * vy).toBeCloseTo(values[i] * vy, 6);
        }
    });

    test('orients each eigenvector so its first non-zero component is positive', () => {
        expect(column(0)[0]).toBeGreaterThan(0);
        expect(column(1)[0]).toBeGreaterThan(0);
    });
});

describe('calculateEllipse', () => {
    const s = stats(tilted);
    const eigen = calculateEigenvectors(s.covariance);

    test.each([1, 2, 3])('returns 100 points that are all %i Mahalanobis units from the mean', (nStd) => {
        const points = calculateEllipse(s.mean, eigen, nStd);
        expect(points).toHaveLength(100);
        for (const point of points) {
            expect(calculateMahalanobisDistance(point, s)).toBeCloseTo(nStd, 5);
        }
    });
});
