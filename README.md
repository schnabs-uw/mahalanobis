# Mahalanobis Distance Interactive Demo

This repository contains both Python (Dash) and React implementations of an interactive Mahalanobis Distance visualization tool.

## Python Version (Dash)

### Requirements
- Python 3.x
- Required packages are listed in `requirements.txt`

### Installation
```powershell
pip install -r requirements.txt
```

### Running the App
```powershell
python mahalanobis_distance_app.py
```

The app will be available at `http://localhost:8050`

## React Version (TypeScript)

### Requirements
- Node.js (LTS version)
- npm (comes with Node.js)

### Installation
```powershell
npm install
```

### Running the App
```powershell
npm start
```

The app will be available at `http://localhost:3000`

## Features (Both Versions)

- Interactive visualization of Mahalanobis Distance
- Dataset and Test point manipulation
- Real-time calculation and display of:
  - Mahalanobis distances
  - Covariance ellipses (1, 2, and 3 standard deviations)
  - Dataset and test point coordinates

### Interactive Elements
- Add individual points to either dataset
- Regenerate random points (all, dataset, or test points)
- Edit point coordinates through tables
- Delete individual points
- Visual feedback with distances displayed above test points

### Mathematical Components
- Covariance matrix calculation
- Eigenvalue/eigenvector computation
- Mahalanobis distance calculation
- Ellipse generation based on covariance

## Implementation Details

### Python Version
- Built with Dash and Plotly
- Uses NumPy for mathematical operations
- Dash callbacks for reactivity

### React Version
- Built with React and TypeScript
- Uses ECharts for visualization
- Uses mathjs for mathematical operations
- Material-UI for modern UI components
- Client-side calculations for improved responsiveness

## License
MIT License