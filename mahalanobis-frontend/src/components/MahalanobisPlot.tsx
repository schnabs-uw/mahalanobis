import React, { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import { Point, calculateMahalanobisDistance, calculateEllipse, calculateEigenvectors } from '../utils/math';
import { 
    Button, 
    Stack, 
    Box, 
    Typography, 
    Table, 
    TableBody, 
    TableCell, 
    TableContainer, 
    TableHead, 
    TableRow, 
    Paper, 
    TextField,
    IconButton
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';

export interface MahalanobisPlotProps {
    initialDatasetPoints?: number;
    initialTestPoints?: number;
}

export const MahalanobisPlot: React.FC<MahalanobisPlotProps> = ({
    initialDatasetPoints = 20,
    initialTestPoints = 16
}) => {
    const [datasetPoints, setDatasetPoints] = useState<Point[]>([]);
    const [testPoints, setTestPoints] = useState<Point[]>([]);

    const generateRandomPoint = (min: number, max: number): Point => ({
        x: Math.random() * (max - min) + min,
        y: Math.random() * (max - min) + min
    });

    const generateRandomPoints = (count: number): Point[] => {
        return Array.from({ length: count }, () => generateRandomPoint(-5, 25));
    };

    const addRandomPoint = (points: Point[], setPoints: React.Dispatch<React.SetStateAction<Point[]>>) => {
        if (points.length === 0) {
            setPoints([generateRandomPoint(-5, 5)]);
            return;
        }

        const minX = Math.min(...points.map(p => p.x));
        const maxX = Math.max(...points.map(p => p.x));
        const minY = Math.min(...points.map(p => p.y));
        const maxY = Math.max(...points.map(p => p.y));

        const rangeX = maxX - minX;
        const rangeY = maxY - minY;

        const newPoint = {
            x: Math.random() * (rangeX * 1.4) + (minX - rangeX * 0.2),
            y: Math.random() * (rangeY * 1.4) + (minY - rangeY * 0.2)
        };

        setPoints([...points, newPoint]);
    };

    useEffect(() => {
        setDatasetPoints(generateRandomPoints(initialDatasetPoints));
        setTestPoints(generateRandomPoints(initialTestPoints));
    }, [initialDatasetPoints, initialTestPoints]);    const getChartOptions = () => {
        // Calculate mean, covariance matrix, and eigenvectors
        let mean = { x: 0, y: 0 };
        let covMatrix = [[0, 0], [0, 0]];
        
        if (datasetPoints.length >= 2) {
            // Calculate mean
            mean = datasetPoints.reduce(
                (acc, p) => ({ 
                    x: acc.x + p.x / datasetPoints.length, 
                    y: acc.y + p.y / datasetPoints.length 
                }),
                { x: 0, y: 0 }
            );

            // Calculate covariance matrix
            datasetPoints.forEach(p => {
                const dx = p.x - mean.x;
                const dy = p.y - mean.y;
                covMatrix[0][0] += dx * dx / datasetPoints.length;
                covMatrix[0][1] += dx * dy / datasetPoints.length;
                covMatrix[1][0] += dx * dy / datasetPoints.length;
                covMatrix[1][1] += dy * dy / datasetPoints.length;
            });

            // Add small regularization term
            covMatrix[0][0] += 1e-6;
            covMatrix[1][1] += 1e-6;
        }

        const ellipses = datasetPoints.length >= 2 
            ? [1, 2, 3].map(std => calculateEllipse(datasetPoints, std))
            : [];

        const distances = testPoints.map(point => 
            calculateMahalanobisDistance(point, datasetPoints)
        );

        // Calculate eigenvectors and scale them for visualization
        const { values: eigenvalues, vectors: eigenvectors } = 
            datasetPoints.length >= 2 ? calculateEigenvectors(covMatrix) : { values: [], vectors: [] };

        // Scale factor for eigenvector visualization
        const scaleFactor = 5;
        const eigenvectorLines = eigenvalues.map((value, i) => ({            name: `Eigenvector ${i + 1} (λ=${value.toFixed(2)})`,
            type: 'line',
            data: [
                [mean.x, mean.y],
                [
                    mean.x + eigenvectors[0][i] * Math.sqrt(Math.abs(value)) * scaleFactor,
                    mean.y + eigenvectors[1][i] * Math.sqrt(Math.abs(value)) * scaleFactor
                ]
            ],
            symbol: ['none', 'arrow'],
            symbolSize: 10,
            lineStyle: {
                color: i === 0 ? '#FF7F0E' : '#2CA02C', // Orange for first, green for second
                width: 2
            },
            label: {
                show: true,
                formatter: `λ${i + 1}=${value.toFixed(2)}`,
                position: 'end'
            },
            legend: {
                show: true
            }
        }));        return {
            title: {
                text: 'Mahalanobis Distance Interactive Demo'
            },            legend: {
                show: true,
                top: 'bottom',
                padding: [5, 5],
                itemGap: 20,
                data: [
                    { name: 'Dataset Points', icon: 'circle' },
                    { name: 'Test Points', icon: 'circle' },
                    ...eigenvalues.map((value, i) => ({ 
                        name: `Eigenvector ${i + 1} (λ=${value.toFixed(2)})`, 
                        icon: 'line',
                        itemStyle: {
                            color: i === 0 ? '#FF7F0E' : '#2CA02C'
                        }
                    })),
                    ...Array(3).fill(0).map((_, i) => ({ 
                        name: `${i + 1}-std Ellipse`, 
                        icon: 'line',
                        itemStyle: {
                            color: ['green', 'blue', 'purple'][i]
                        }
                    }))
                ]
            },            tooltip: {
                trigger: 'item',
                formatter: function(params: any) {
                    if (params.seriesName.includes('Eigenvector')) {
                        return params.seriesName;
                    }
                    return params.value;
                }
            },
            xAxis: {
                type: 'value',
                min: -12,
                max: 36
            },
            yAxis: {
                type: 'value',
                min: -12,
                max: 30
            },
            series: [
                {
                    name: 'Dataset Points',
                    type: 'scatter',
                    data: datasetPoints.map(p => [p.x, p.y]),
                    itemStyle: { color: 'blue' }
                },
                {
                    name: 'Test Points',
                    type: 'scatter',
                    data: testPoints.map((p, i) => ({
                        value: [p.x, p.y],
                        label: {
                            show: true,
                            position: 'top',
                            formatter: distances[i].toFixed(2)
                        }
                    })),
                    itemStyle: { color: 'red' }
                },
                ...eigenvectorLines,
                ...ellipses.map((points, i) => ({
                    name: `${i + 1}-std Ellipse`,
                    type: 'line',
                    data: points.map(p => [p.x, p.y]),
                    symbol: 'none',
                    animation: false,
                    itemStyle: {
                        color: ['green', 'blue', 'purple'][i]
                    },
                    lineStyle: {
                        type: 'dashed'
                    }
                }))
            ]
        };
    };    return (
        <div>
            <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
                <Button variant="contained" onClick={() => addRandomPoint(datasetPoints, setDatasetPoints)}>
                    Add Dataset Point
                </Button>
                <Button variant="contained" onClick={() => addRandomPoint(testPoints, setTestPoints)}>
                    Add Test Point
                </Button>
                <Button 
                    variant="contained" 
                    onClick={() => {
                        setDatasetPoints(generateRandomPoints(initialDatasetPoints));
                        setTestPoints(generateRandomPoints(initialTestPoints));
                    }}
                >
                    Regenerate All Points
                </Button>
                <Button 
                    variant="contained" 
                    onClick={() => setDatasetPoints(generateRandomPoints(initialDatasetPoints))}
                >
                    Regenerate Dataset
                </Button>
                <Button 
                    variant="contained" 
                    onClick={() => setTestPoints(generateRandomPoints(initialTestPoints))}
                >
                    Regenerate Test Points
                </Button>
            </Stack>
            <Box sx={{ display: 'flex', gap: 2 }}>
                {/* Chart Column */}
                <Box sx={{ width: '65%' }}>
                    <ReactECharts 
                        option={getChartOptions()} 
                        style={{ height: '600px' }} 
                        notMerge={true}
                    />
                </Box>
                {/* Tables Column */}
                <Box sx={{ width: '30%' }}>
                    <Typography variant="h6" gutterBottom>Dataset Points</Typography>
                    <TableContainer 
                        component={Paper} 
                        sx={{ maxHeight: 250, marginBottom: 3 }}
                    >
                        <Table size="small" stickyHeader>
                            <TableHead>
                                <TableRow>
                                    <TableCell>X</TableCell>
                                    <TableCell>Y</TableCell>
                                    <TableCell padding="checkbox"></TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {datasetPoints.map((point, index) => (
                                    <TableRow key={index}>
                                        <TableCell>
                                            <TextField
                                                size="small"
                                                value={point.x}
                                                onChange={(e) => {
                                                    const newPoints = [...datasetPoints];
                                                    newPoints[index] = { ...point, x: Number(e.target.value) };
                                                    setDatasetPoints(newPoints);
                                                }}
                                                type="number"
                                                variant="standard"
                                            />
                                        </TableCell>
                                        <TableCell>
                                            <TextField
                                                size="small"
                                                value={point.y}
                                                onChange={(e) => {
                                                    const newPoints = [...datasetPoints];
                                                    newPoints[index] = { ...point, y: Number(e.target.value) };
                                                    setDatasetPoints(newPoints);
                                                }}
                                                type="number"
                                                variant="standard"
                                            />
                                        </TableCell>
                                        <TableCell>
                                            <IconButton
                                                size="small"
                                                onClick={() => {
                                                    const newPoints = datasetPoints.filter((_, i) => i !== index);
                                                    setDatasetPoints(newPoints);
                                                }}
                                            >
                                                <DeleteIcon />
                                            </IconButton>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>

                    <Typography variant="h6" gutterBottom>Test Points</Typography>
                    <TableContainer 
                        component={Paper} 
                        sx={{ maxHeight: 250 }}
                    >
                        <Table size="small" stickyHeader>
                            <TableHead>
                                <TableRow>
                                    <TableCell>X</TableCell>
                                    <TableCell>Y</TableCell>
                                    <TableCell padding="checkbox"></TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {testPoints.map((point, index) => (
                                    <TableRow key={index}>
                                        <TableCell>
                                            <TextField
                                                size="small"
                                                value={point.x}
                                                onChange={(e) => {
                                                    const newPoints = [...testPoints];
                                                    newPoints[index] = { ...point, x: Number(e.target.value) };
                                                    setTestPoints(newPoints);
                                                }}
                                                type="number"
                                                variant="standard"
                                            />
                                        </TableCell>
                                        <TableCell>
                                            <TextField
                                                size="small"
                                                value={point.y}
                                                onChange={(e) => {
                                                    const newPoints = [...testPoints];
                                                    newPoints[index] = { ...point, y: Number(e.target.value) };
                                                    setTestPoints(newPoints);
                                                }}
                                                type="number"
                                                variant="standard"
                                            />
                                        </TableCell>
                                        <TableCell>
                                            <IconButton
                                                size="small"
                                                onClick={() => {
                                                    const newPoints = testPoints.filter((_, i) => i !== index);
                                                    setTestPoints(newPoints);
                                                }}
                                            >
                                                <DeleteIcon />
                                            </IconButton>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                </Box>
            </Box>
        </div>
    );
};
