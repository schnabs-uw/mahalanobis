# Mahalanobis Distance Demo Repository

This repository contains Python scripts demonstrating the calculation of Euclidean and Mahalanobis distances. Follow the steps below to set up and run the scripts.

## Prerequisites

Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

## Setup

1. Clone the repository or download the source code.
2. Navigate to the project directory:
   ```bash
   cd pylearn
   ```
3. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Virtual Environment Setup

It is recommended to use a virtual environment to manage dependencies. Follow these steps to create and activate a virtual environment:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

3. To deactivate the virtual environment, run:
   ```bash
   deactivate
   ```

## Running the Scripts

### Euclidean Distance Demo

To run the Euclidean distance demonstration script, execute the following command:
```bash
python euclidean_distance_demo.py
```

### Mahalanobis Distance Demo

To run the Mahalanobis distance demonstration script, execute the following command:
```bash
python mahalanobis_distance_demo.py
```

## Additional Information

- The `requirements.txt` file lists the dependencies required for the scripts, including `numpy`, `matplotlib`, and `scipy`.
- The scripts generate visualizations and outputs to help understand the distance calculations.

Feel free to explore and modify the scripts to suit your needs.
