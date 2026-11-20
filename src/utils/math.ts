import { inv } from 'mathjs';

export interface Point {
    x: number;
    y: number;
}

export interface MeanAndCovariance {
    mean: Point;
    covariance: number[][];
}

export interface EigenDecomposition {
    values: number[];
    vectors: number[][];
}

/** Returns null for fewer than 2 points, where covariance is undefined. */
export function calculateMeanAndCovariance(points: Point[]): MeanAndCovariance | null {
    if (points.length < 2) return null;
    const n = points.length;

    const mean = points.reduce(
        (acc, p) => ({ x: acc.x + p.x / n, y: acc.y + p.y / n }),
        { x: 0, y: 0 }
    );

    // Sample covariance (divided by n−1), the same as NumPy's np.cov.
    const covariance = [[0, 0], [0, 0]];
    points.forEach(p => {
        const dx = p.x - mean.x;
        const dy = p.y - mean.y;
        covariance[0][0] += dx * dx / (n - 1);
        covariance[0][1] += dx * dy / (n - 1);
        covariance[1][0] += dx * dy / (n - 1);
        covariance[1][1] += dy * dy / (n - 1);
    });

    // Keeps the matrix invertible when all points lie on one line.
    covariance[0][0] += 1e-6;
    covariance[1][1] += 1e-6;

    return { mean, covariance };
}

export function calculateMahalanobisDistance(point: Point, { mean, covariance }: MeanAndCovariance): number {
    try {
        const invCov = inv(covariance);
        const dx = point.x - mean.x;
        const dy = point.y - mean.y;
        return Math.sqrt(
            dx * dx * invCov[0][0] +
            2 * dx * dy * invCov[0][1] +
            dy * dy * invCov[1][1]
        );
    } catch (e) {
        console.warn('Error calculating Mahalanobis distance:', e);
        return Infinity;
    }
}

export function calculateEigenvectors(matrix: number[][]): EigenDecomposition {
    // Closed form for a symmetric 2×2 matrix [[a, b], [b, d]].
    const [[a, b], [, d]] = matrix;
    const center = (a + d) / 2;
    const radius = Math.hypot((a - d) / 2, b);
    // Angle of the major axis; atan2 also covers diagonal matrices, including a > d, a < d and a = d.
    const angle = Math.atan2(2 * b, a - d) / 2;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);

    // Orient each eigenvector so its first non-zero component is positive, so the arrows don't flip.
    const normalizeDirection = (vector: number[]): number[] => {
        const firstNonZeroIndex = vector.findIndex(x => Math.abs(x) > 1e-10);
        if (firstNonZeroIndex !== -1 && vector[firstNonZeroIndex] < 0) {
            return vector.map(x => -x);
        }
        return vector;
    };

    const [v1x, v1y] = normalizeDirection([cos, sin]);
    const [v2x, v2y] = normalizeDirection([-sin, cos]);

    return {
        values: [center + radius, center - radius],
        vectors: [
            [v1x, v2x],
            [v1y, v2y],
        ],
    };
}

export function calculateEllipse(mean: Point, { values: eigenvalues, vectors: eigenvectors }: EigenDecomposition, nStd: number = 1): Point[] {
    const points: Point[] = [];
    const steps = 100;
    for (let i = 0; i < steps; i++) {
        const angle = (2 * Math.PI * i) / steps;
        const x = Math.cos(angle) * nStd * Math.sqrt(Math.abs(eigenvalues[0]));
        const y = Math.sin(angle) * nStd * Math.sqrt(Math.abs(eigenvalues[1]));

        // Transform point
        const transformedX = mean.x + x * eigenvectors[0][0] + y * eigenvectors[0][1];
        const transformedY = mean.y + x * eigenvectors[1][0] + y * eigenvectors[1][1];

        points.push({ x: transformedX, y: transformedY });
    }
    return points;
}
