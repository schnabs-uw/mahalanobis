import { Matrix, det, inv, multiply, subtract, transpose } from 'mathjs';

export interface Point {
    x: number;
    y: number;
}

export function calculateMahalanobisDistance(point: Point, dataset: Point[]): number {
    if (dataset.length < 2) return 0;

    // Calculate mean
    const mean = dataset.reduce(
        (acc, p) => ({ x: acc.x + p.x / dataset.length, y: acc.y + p.y / dataset.length }),
        { x: 0, y: 0 }
    );

    // Calculate covariance matrix
    const covMatrix = [[0, 0], [0, 0]];
    dataset.forEach(p => {
        const dx = p.x - mean.x;
        const dy = p.y - mean.y;
        covMatrix[0][0] += dx * dx / dataset.length;
        covMatrix[0][1] += dx * dy / dataset.length;
        covMatrix[1][0] += dx * dy / dataset.length;
        covMatrix[1][1] += dy * dy / dataset.length;
    });

    // Add small regularization term
    covMatrix[0][0] += 1e-6;
    covMatrix[1][1] += 1e-6;

    try {
        const invCov = inv(covMatrix);
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

export function calculateEigenvectors(matrix: number[][]): { values: number[], vectors: number[][] } {
    const n = matrix.length;
    const identity = Array(n).fill(0).map((_, i) => Array(n).fill(0).map((_, j) => i === j ? 1 : 0));
    
    // Power iteration method for largest eigenvalue/vector
    const powerIteration = (matrix: number[][], maxIter: number = 100): { value: number, vector: number[] } => {
        let vector = Array(n).fill(0).map(() => Math.random());
        let value = 0;
        
        for (let i = 0; i < maxIter; i++) {
            const newVector = multiply(matrix, vector) as number[];            const norm = Math.sqrt(newVector.reduce((sum, x) => sum + x * x, 0));
            vector = newVector.map(x => x / norm);
            const vectorMatrix = [vector]; // Create a 1xN matrix
            const vectorMatrixTranspose = transpose(vectorMatrix); // Create an Nx1 matrix
            const temp = multiply(vectorMatrix, matrix) as number[][]; // 1xN * NxN = 1xN
            value = (multiply(temp, vectorMatrixTranspose) as number[][])[0][0]; // 1xN * Nx1 = 1x1
        }
        
        return { value, vector };
    };
    
    const result1 = powerIteration(matrix);
    
    // Deflate matrix to find second eigenvalue/vector
    const deflated = subtract(
        matrix,
        multiply(
            multiply(transpose([result1.vector]), [result1.vector]),
            result1.value
        )
    ) as number[][];
    
    const result2 = powerIteration(deflated);
    
    return {
        values: [result1.value, result2.value],
        vectors: transpose([result1.vector, result2.vector]) as number[][]
    };
}

export function calculateEllipse(dataset: Point[], nStd: number = 1): Point[] {
    if (dataset.length < 2) return [];

    // Calculate mean
    const mean = dataset.reduce(
        (acc, p) => ({ x: acc.x + p.x / dataset.length, y: acc.y + p.y / dataset.length }),
        { x: 0, y: 0 }
    );

    // Calculate covariance matrix
    const covMatrix = [[0, 0], [0, 0]];
    dataset.forEach(p => {
        const dx = p.x - mean.x;
        const dy = p.y - mean.y;
        covMatrix[0][0] += dx * dx / dataset.length;
        covMatrix[0][1] += dx * dy / dataset.length;
        covMatrix[1][0] += dx * dy / dataset.length;
        covMatrix[1][1] += dy * dy / dataset.length;
    });

    // Add small regularization term
    covMatrix[0][0] += 1e-6;
    covMatrix[1][1] += 1e-6;

    try {
        const { values: eigenvalues, vectors: eigenvectors } = calculateEigenvectors(covMatrix);

        // Generate ellipse points
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
    } catch (e) {
        console.warn('Error calculating ellipse:', e);
        return [];
    }
}
