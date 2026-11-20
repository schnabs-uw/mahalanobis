import { describe, expect, test } from 'vitest';
import {
    type Point,
    calculateEigenvectors,
    calculateEllipse,
    calculateMahalanobisDistance,
    calculateMeanAndCovariance,
} from './math';

// Worked by hand: mean (2, 1); covariance divided by n−1 = 3 is [[8/3, 4/3], [4/3, 4/3]];
// its inverse is [[3/4, -3/4], [-3/4, 3/2]].
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

    test('returns the mean and the covariance divided by n−1', () => {
        const { mean, covariance } = stats(tilted);
        expect(mean).toEqual({ x: 2, y: 1 });
        expect(covariance[0][0]).toBeCloseTo(8 / 3, 5);
        expect(covariance[0][1]).toBeCloseTo(4 / 3, 5);
        expect(covariance[1][0]).toBeCloseTo(4 / 3, 5);
        expect(covariance[1][1]).toBeCloseTo(4 / 3, 5);
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
        expect(calculateMahalanobisDistance({ x: 3, y: 1 }, s)).toBeCloseTo(Math.sqrt(3) / 2, 5);
        expect(calculateMahalanobisDistance({ x: 2, y: 2 }, s)).toBeCloseTo(Math.sqrt(6) / 2, 5);
    });

    test('gives different distances to points the same Euclidean distance from the mean', () => {
        // Both points are 2 units from the mean; the cloud is more spread out along x.
        expect(calculateMahalanobisDistance({ x: 4, y: 1 }, s)).toBeCloseTo(Math.sqrt(3), 5);
        expect(calculateMahalanobisDistance({ x: 2, y: 3 }, s)).toBeCloseTo(Math.sqrt(6), 5);
    });
});

describe('calculateEigenvectors', () => {
    const column = (vectors: number[][], i: number) => [vectors[0][i], vectors[1][i]];

    // Checks A·v = λ·v, unit length, orthogonality, largest-first order and the sign rule.
    function expectValidDecomposition(matrix: number[][]) {
        const { values, vectors } = calculateEigenvectors(matrix);
        const scale = Math.max(Math.abs(values[0]), 1e-12);
        expect(values[0]).toBeGreaterThanOrEqual(values[1]);
        for (const i of [0, 1]) {
            const [vx, vy] = column(vectors, i);
            expect(Math.hypot(vx, vy)).toBeCloseTo(1, 9);
            expect(Math.abs(matrix[0][0] * vx + matrix[0][1] * vy - values[i] * vx) / scale).toBeLessThan(1e-9);
            expect(Math.abs(matrix[1][0] * vx + matrix[1][1] * vy - values[i] * vy) / scale).toBeLessThan(1e-9);
            const firstNonZero = [vx, vy].find(x => Math.abs(x) > 1e-10);
            expect(firstNonZero).toBeGreaterThan(0);
        }
        const [ax, ay] = column(vectors, 0);
        const [bx, by] = column(vectors, 1);
        expect(ax * bx + ay * by).toBeCloseTo(0, 9);
        return { values, vectors };
    }

    test('handles a tilted matrix', () => {
        const { values } = expectValidDecomposition([[2, 1], [1, 1]]);
        expect(values[0]).toBeCloseTo((3 + Math.sqrt(5)) / 2, 9);
        expect(values[1]).toBeCloseTo((3 - Math.sqrt(5)) / 2, 9);
    });

    test('handles a diagonal matrix with the larger spread along x', () => {
        const { values, vectors } = expectValidDecomposition([[3, 0], [0, 1]]);
        expect(values[0]).toBeCloseTo(3, 9);
        expect(values[1]).toBeCloseTo(1, 9);
        expect(column(vectors, 0)).toEqual([1, 0]);
        expect(column(vectors, 1)[0]).toBeCloseTo(0, 9);
        expect(column(vectors, 1)[1]).toBeCloseTo(1, 9);
    });

    test('handles a diagonal matrix with the larger spread along y', () => {
        const { values, vectors } = expectValidDecomposition([[1, 0], [0, 4]]);
        expect(values[0]).toBeCloseTo(4, 9);
        expect(values[1]).toBeCloseTo(1, 9);
        expect(column(vectors, 0)[0]).toBeCloseTo(0, 9);
        expect(column(vectors, 0)[1]).toBeCloseTo(1, 9);
        expect(column(vectors, 1)[0]).toBeCloseTo(1, 9);
        expect(column(vectors, 1)[1]).toBeCloseTo(0, 9);
    });

    test('handles a multiple of the identity, where every direction is an eigenvector', () => {
        const { values } = expectValidDecomposition([[2, 0], [0, 2]]);
        expect(values).toEqual([2, 2]);
    });

    test('finds the axes of a nearly circular tilted matrix', () => {
        // Eigenvalues 1.01 and 1 along axes rotated by 30°.
        const angle = Math.PI / 6;
        const c = Math.cos(angle);
        const s = Math.sin(angle);
        const matrix = [
            [1.01 * c * c + s * s, 0.01 * c * s],
            [0.01 * c * s, 1.01 * s * s + c * c],
        ];
        const { values, vectors } = expectValidDecomposition(matrix);
        expect(values[0]).toBeCloseTo(1.01, 9);
        expect(values[1]).toBeCloseTo(1, 9);
        expect(column(vectors, 0)[0]).toBeCloseTo(c, 9);
        expect(column(vectors, 0)[1]).toBeCloseTo(s, 9);
    });

    test('is valid for the covariance of many random datasets', () => {
        // Seeded so a failure is reproducible.
        let seed = 12345;
        const random = () => {
            seed = (seed * 1664525 + 1013904223) % 2 ** 32;
            return seed / 2 ** 32;
        };
        for (let run = 0; run < 500; run++) {
            const points = Array.from({ length: 20 }, () => ({ x: random() * 30 - 5, y: random() * 30 - 5 }));
            expectValidDecomposition(stats(points).covariance);
        }
    });
});

describe('calculateEllipse', () => {
    const datasets: [string, Point[]][] = [
        ['a tilted dataset', tilted],
        ['a wide axis-aligned rectangle', [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 0, y: 2 }, { x: 4, y: 2 }]],
        ['a tall axis-aligned rectangle', [{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 0, y: 4 }, { x: 2, y: 4 }]],
        ['two identical points', [{ x: 1, y: 1 }, { x: 1, y: 1 }]],
    ];

    describe.each(datasets)('for %s', (_, dataset) => {
        const s = stats(dataset);
        const eigen = calculateEigenvectors(s.covariance);

        test.each([1, 2, 3])('returns 100 points that are all %i Mahalanobis units from the mean', (nStd) => {
            const points = calculateEllipse(s.mean, eigen, nStd);
            expect(points).toHaveLength(100);
            for (const point of points) {
                expect(calculateMahalanobisDistance(point, s)).toBeCloseTo(nStd, 5);
            }
        });
    });
});
