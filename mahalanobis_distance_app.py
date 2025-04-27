import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table # Added dash_table
import plotly.graph_objs as go
from scipy.spatial.distance import mahalanobis

def calculate_mahalanobis_distance(point, data):
    """Calculate the Mahalanobis distance of a point from a dataset."""
    data = np.array(data)
    if data.ndim == 1: # Handle case with only one data point
        data = data.reshape(1, -1)
    if data.shape[0] < 2: # Not enough points for covariance
        return 0.0 # Or handle as appropriate
    point = np.array(point)
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    # Add a small value to the diagonal to regularize the covariance matrix
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Handle singular matrix case if regularization is not enough
        print("Warning: Covariance matrix is singular.")
        return np.inf # Or some other indicator of error
    return mahalanobis(point, mean, inv_cov_matrix)

def get_cov_ellipse(mean, cov, n_std=1.0, num_points=100):
    """Plot an ellipse representing the covariance matrix."""
    # Ensure covariance matrix is valid for eigenvalue decomposition
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        print("Warning: Invalid values in covariance matrix.")
        return np.array([]), np.array([]) # Return empty arrays

    try:
        vals, vecs = np.linalg.eigh(cov)
        # Check for non-positive eigenvalues which can occur with near-singular matrices
        if np.any(vals <= 1e-6): # Use a small threshold instead of zero
             print(f"Warning: Non-positive eigenvalues detected: {vals}. Ellipse may be inaccurate.")
             vals = np.maximum(vals, 1e-6) # Clamp small/negative eigenvalues

    except np.linalg.LinAlgError:
        print("Warning: Eigenvalue decomposition failed.")
        return np.array([]), np.array([]) # Return empty arrays

    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    t = np.linspace(0, 2*np.pi, num_points)
    ellipse_pts = np.array([np.cos(t), np.sin(t)])
    scale = n_std * np.sqrt(vals) # Ensure vals are non-negative before sqrt
    ellipse_transform = vecs @ np.diag(scale) @ ellipse_pts
    return ellipse_transform[0] + mean[0], ellipse_transform[1] + mean[1]


def get_eigenvectors(mean, cov):
    """Get eigenvectors and eigenvalues."""
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)) or cov.shape[0] < 2:
        print("Warning: Invalid covariance matrix for eigenvectors.")
        return []
    try:
        vals, vecs = np.linalg.eigh(cov)
        # Clamp small/negative eigenvalues
        vals = np.maximum(vals, 1e-9)
    except np.linalg.LinAlgError:
        print("Warning: Eigenvalue decomposition failed for eigenvectors.")
        return []

    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    return [(mean, vecs[:,i], vals[i]) for i in range(len(vals))]

def make_figure(dataset_points, test_points):
    """Generates the Plotly figure."""
    fig_data = []

    # Ensure dataset_points is a 2D array
    if dataset_points.ndim == 1:
        dataset_points = dataset_points.reshape(1, -1)
    if test_points.ndim == 1:
        test_points = test_points.reshape(1, -1)

    # --- Dataset Points ---
    if dataset_points.shape[0] > 0:
        dataset_trace = go.Scatter(
            x=dataset_points[:,0], y=dataset_points[:,1],
            mode='markers', marker=dict(color='blue', size=8),
            name='Dataset Points',
            customdata=[f'dataset_{i}' for i in range(len(dataset_points))],
            selected=dict(marker=dict(color='cyan', size=12)),
            unselected=dict(marker=dict(opacity=0.7)),
            hoverinfo='text',
            text=[f"({x:.2f}, {y:.2f})" for x, y in dataset_points]
        )
        fig_data.append(dataset_trace)

        # --- Calculations requiring dataset points ---
        if dataset_points.shape[0] >= 2: # Need at least 2 points for covariance
            mean = np.mean(dataset_points, axis=0)
            cov = np.cov(dataset_points, rowvar=False)
            cov += np.eye(cov.shape[0]) * 1e-6 # Regularization

            # --- Test Points ---
            if test_points.shape[0] > 0:
                distances = [calculate_mahalanobis_distance(pt, dataset_points) for pt in test_points]
                test_trace = go.Scatter(
                    x=test_points[:,0], y=test_points[:,1],
                    mode='markers+text', marker=dict(color='red', size=10),
                    name='Test Points',
                    customdata=[f'test_{i}' for i in range(len(test_points))],
                    text=[f"{d:.2f}" for d in distances],
                    textposition='top center',
                    hoverinfo='text',
                    selected=dict(marker=dict(color='orange', size=14)),
                    unselected=dict(marker=dict(opacity=0.7)),
                )
                fig_data.append(test_trace)

            # --- Covariance Ellipses ---
            colors = ['green', 'blue', 'purple']
            for idx, n_std in enumerate([1, 2, 3]):
                ex, ey = get_cov_ellipse(mean, cov, n_std)
                if len(ex) > 0: # Check if ellipse calculation was successful
                    fig_data.append(go.Scatter(
                        x=ex, y=ey, mode='lines',
                        line=dict(color=colors[idx], dash='dash'),
                        name=f'{n_std}-std Covariance Ellipse',
                        hoverinfo='skip',
                    ))

            # --- Eigenvectors ---
            eigenvectors = get_eigenvectors(mean, cov)
            for i, (m, vec, val) in enumerate(eigenvectors):
                scale = np.sqrt(val) * 3 # Ensure val is non-negative
                fig_data.append(go.Scatter(
                    x=[mean[0], mean[0]+vec[0]*scale],
                    y=[mean[1], mean[1]+vec[1]*scale],
                    mode='lines+markers',
                    line=dict(color='orange', width=4),
                    marker=dict(size=1),
                    name=f'Eigenvector {i+1} (Eigenvalue: {val:.2f})',
                    hoverinfo='skip',
                ))
        else:
             # Handle case with 0 or 1 dataset point (no covariance/eigenvectors)
             if test_points.shape[0] > 0:
                 # Plot test points without distances if dataset is too small
                 test_trace = go.Scatter(
                     x=test_points[:,0], y=test_points[:,1],
                     mode='markers', marker=dict(color='red', size=10),
                     name='Test Points (No Distances)',
                     customdata=[f'test_{i}' for i in range(len(test_points))],
                     hoverinfo='text',
                     text=[f"({x:.2f}, {y:.2f})" for x, y in test_points],
                     selected=dict(marker=dict(color='orange', size=14)),
                     unselected=dict(marker=dict(opacity=0.7)),
                 )
                 fig_data.append(test_trace)

    elif test_points.shape[0] > 0: # Only test points exist
        test_trace = go.Scatter(
            x=test_points[:,0], y=test_points[:,1],
            mode='markers', marker=dict(color='red', size=10),
            name='Test Points',
            customdata=[f'test_{i}' for i in range(len(test_points))],
            hoverinfo='text',
            text=[f"({x:.2f}, {y:.2f})" for x, y in test_points],
            selected=dict(marker=dict(color='orange', size=14)),
            unselected=dict(marker=dict(opacity=0.7)),
        )
        fig_data.append(test_trace)


    # --- Layout ---
    layout = go.Layout(
        title='Dataset with Eigenvectors and Covariance Ellipses',
        xaxis=dict(title='X-axis', zeroline=True, range=[-10, 30]), # Example fixed range
        yaxis=dict(title='Y-axis', zeroline=True, range=[-10, 25]), # Example fixed range
        showlegend=True,
        dragmode='select', # Changed dragmode for selection
        clickmode='event+select',
        margin=dict(l=40, r=40, t=60, b=40),
        height=600, # Adjusted height
        # width=900, # Let width be more flexible or set percentage
    )
    fig = go.Figure(data=fig_data, layout=layout)
    # Ensure axes are scaled equally if desired
    fig.update_layout(yaxis_scaleanchor="x", yaxis_scaleratio=1)
    return fig


# Initial data - randomly generated
np.random.seed()  # Reset random seed each time
num_dataset_points = 20
num_test_points = 16

# Generate random dataset points in a reasonable range
initial_dataset_points = np.array([
    [np.random.uniform(-5, 25), np.random.uniform(-5, 15)] 
    for _ in range(num_dataset_points)
])

# Generate random test points in a similar range
initial_test_points = np.array([
    [np.random.uniform(-5, 25), np.random.uniform(-5, 15)] 
    for _ in range(num_test_points)
])


# Dash app
app = dash.Dash(__name__)

# Function to create data for tables
def create_table_data(points_list):
    return [{'X': p[0], 'Y': p[1]} for p in points_list]

app.layout = html.Div([
    html.H2('Mahalanobis Distance Interactive Demo'),
    html.Div([
        html.Button('Add Dataset Point', id='add-dataset-btn', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Add Test Point', id='add-test-btn', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Regenerate All Points', id='regenerate-btn', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Regenerate Dataset', id='regenerate-dataset-btn', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('Regenerate Test Points', id='regenerate-test-btn', n_clicks=0, style={'marginRight': '10px'}),
    ], style={'marginBottom': '20px'}),

    html.Div([ # Main container for graph and tables
        # Graph Column
        html.Div([
            dcc.Graph(
                id='main-graph',
                # figure=make_figure(initial_dataset_points, initial_test_points), # Initial figure set in callback
                config={
                    # 'editable': True, # Disable direct graph editing for now
                    # 'edits': {'shapePosition': True, 'annotationPosition': True},
                    'displayModeBar': True,
                    'scrollZoom': True,
                },
            )
        ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Tables Column
        html.Div([
            html.H4("Dataset Points"),
            dash_table.DataTable(
                id='dataset-table',
                columns=[
                    {'name': 'X', 'id': 'X', 'type': 'numeric', 'editable': True},
                    {'name': 'Y', 'id': 'Y', 'type': 'numeric', 'editable': True},
                ],
                row_deletable=True,
                style_table={'height': '250px', 'overflowY': 'auto'},
                 style_cell={'textAlign': 'left'},
                 # Removed row_selectable='single',
            ),
            html.H4("Test Points", style={'marginTop': '20px'}),
            dash_table.DataTable(
                id='test-table',
                columns=[
                    {'name': 'X', 'id': 'X', 'type': 'numeric', 'editable': True},
                    {'name': 'Y', 'id': 'Y', 'type': 'numeric', 'editable': True},
                ],
                row_deletable=True,
                 style_table={'height': '250px', 'overflowY': 'auto'},
                 style_cell={'textAlign': 'left'},
                 # Removed row_selectable='single',
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),

    ], style={'display': 'flex', 'flexDirection': 'row'}), # Use flexbox for layout

    # Stores
    dcc.Store(id='dataset-points', data=initial_dataset_points.tolist()),
    dcc.Store(id='test-points', data=initial_test_points.tolist()),
    dcc.Store(id='selected-dataset-index', data=None), # Store index
    dcc.Store(id='selected-test-index', data=None), # Store index
    html.Div(id='debug-output', style={'display': 'none'}) # Hidden div for debugging if needed
])

# Combined Callback for updating stores from buttons, selection, and tables
@app.callback(
    Output('dataset-points', 'data'),
    Output('test-points', 'data'),
    Output('selected-dataset-index', 'data'),
    Output('selected-test-index', 'data'),
    Input('add-dataset-btn', 'n_clicks'),
    Input('add-test-btn', 'n_clicks'),
    Input('regenerate-btn', 'n_clicks'),
    Input('regenerate-dataset-btn', 'n_clicks'),
    Input('regenerate-test-btn', 'n_clicks'),
    Input('main-graph', 'selectedData'),
    Input('dataset-table', 'data'),
    Input('test-table', 'data'),
    State('dataset-points', 'data'),
    State('test-points', 'data'),
    State('selected-dataset-index', 'data'),
    State('selected-test-index', 'data'),
    prevent_initial_call=True,
    allow_duplicate=True
)
def sync_data_sources(
    add_ds_clicks,
    add_test_clicks,
    regenerate_clicks,
    regenerate_dataset_clicks,
    regenerate_test_clicks,
    selected_graph_data,
    ds_table_data,
    test_table_data,
    current_ds_points,
    current_test_points,
    selected_ds_idx,
    selected_test_idx):

    trigger = ctx.triggered_id
    ds_points_updated = list(current_ds_points)
    test_points_updated = list(current_test_points)
    new_selected_ds_idx = selected_ds_idx
    new_selected_test_idx = selected_test_idx

    # Handle regenerate buttons
    if trigger == 'regenerate-btn':
        # Generate new random dataset and test points
        ds_points_updated = [
            [np.random.uniform(-5, 25), np.random.uniform(-5, 15)]
            for _ in range(num_dataset_points)
        ]
        test_points_updated = [
            [np.random.uniform(-5, 25), np.random.uniform(-5, 15)]
            for _ in range(num_test_points)
        ]
        new_selected_ds_idx = None
        new_selected_test_idx = None
        return ds_points_updated, test_points_updated, new_selected_ds_idx, new_selected_test_idx
    
    elif trigger == 'regenerate-dataset-btn':
        # Generate only new random dataset points
        ds_points_updated = [
            [np.random.uniform(-5, 25), np.random.uniform(-5, 15)]
            for _ in range(num_dataset_points)
        ]
        new_selected_ds_idx = None
        new_selected_test_idx = None
    
    elif trigger == 'regenerate-test-btn':
        # Generate only new random test points
        test_points_updated = [
            [np.random.uniform(-5, 25), np.random.uniform(-5, 15)]
            for _ in range(num_test_points)
        ]
        new_selected_ds_idx = None
        new_selected_test_idx = None

    # --- Handle other Button Clicks ---
    if trigger == 'add-dataset-btn':
        if ds_points_updated:
            # Calculate ranges from existing points
            points_array = np.array(ds_points_updated)
            min_x, min_y = np.min(points_array, axis=0)
            max_x, max_y = np.max(points_array, axis=0)
            # Expand range by 20%
            x_range = max_x - min_x
            y_range = max_y - min_y
            if x_range == 0: x_range = 2  # Default range if all points have same x
            if y_range == 0: y_range = 2  # Default range if all points have same y
            # Generate random point within expanded range
            new_x = np.random.uniform(min_x - 0.2 * x_range, max_x + 0.2 * x_range)
            new_y = np.random.uniform(min_y - 0.2 * y_range, max_y + 0.2 * y_range)
        else:
            # If no points exist, use default range
            new_x = np.random.uniform(-5, 5)
            new_y = np.random.uniform(-5, 5)
        
        ds_points_updated.append([float(new_x), float(new_y)])
        new_selected_ds_idx = None # Deselect after adding
        new_selected_test_idx = None
    elif trigger == 'add-test-btn':
        if test_points_updated:
            # Calculate ranges from existing points
            points_array = np.array(test_points_updated)
            min_x, min_y = np.min(points_array, axis=0)
            max_x, max_y = np.max(points_array, axis=0)
            # Expand range by 20%
            x_range = max_x - min_x
            y_range = max_y - min_y
            if x_range == 0: x_range = 2  # Default range if all points have same x
            if y_range == 0: y_range = 2  # Default range if all points have same y
            # Generate random point within expanded range
            new_x = np.random.uniform(min_x - 0.2 * x_range, max_x + 0.2 * x_range)
            new_y = np.random.uniform(min_y - 0.2 * y_range, max_y + 0.2 * y_range)
        else:
            # If no points exist, use default range
            new_x = np.random.uniform(-5, 5)
            new_y = np.random.uniform(-5, 5)
        
        test_points_updated.append([float(new_x), float(new_y)])
        new_selected_ds_idx = None
        new_selected_test_idx = None # Deselect after adding

    # --- Handle Graph Selection ---
    elif trigger == 'main-graph':
        new_selected_ds_idx = None # Reset selection first
        new_selected_test_idx = None
        if selected_graph_data and 'points' in selected_graph_data and selected_graph_data['points']:
            pt = selected_graph_data['points'][0]
            if pt['curveNumber'] == 0: # Dataset point trace
                 new_selected_ds_idx = pt['pointIndex']
            elif pt['curveNumber'] == 1: # Test point trace (adjust curveNumber if needed)
                 # Ensure test curve number is correct based on dataset presence
                 test_curve_number = 1 if len(ds_points_updated) > 0 else 0
                 if pt['curveNumber'] == test_curve_number:
                     new_selected_test_idx = pt['pointIndex']

    # Ensure points are valid numbers before returning
    try:
        ds_points_final = [[float(p[0]), float(p[1])] for p in ds_points_updated]
        test_points_final = [[float(p[0]), float(p[1])] for p in test_points_updated]
    except (ValueError, TypeError):
        print("Invalid data detected during final conversion. Reverting changes.")
        return current_ds_points, current_test_points, selected_ds_idx, selected_test_idx # Revert    # Final check on selected indices validity
    if new_selected_ds_idx is not None and new_selected_ds_idx >= len(ds_points_final):
        new_selected_ds_idx = None
    if new_selected_test_idx is not None and new_selected_test_idx >= len(test_points_final):
        new_selected_test_idx = None

    # Handle table data updates
    if ctx.triggered_id == 'dataset-table':
        ds_points_final = [[row['X'], row['Y']] for row in ds_table_data] if ds_table_data else []
    elif ctx.triggered_id == 'test-table':
        test_points_final = [[row['X'], row['Y']] for row in test_table_data] if test_table_data else []

    return ds_points_final, test_points_final, new_selected_ds_idx, new_selected_test_idx


# Callback to update Graph and Tables from Stores
@app.callback(
    Output('main-graph', 'figure'),
    Output('dataset-table', 'data'),
    Output('test-table', 'data'),
    Input('dataset-points', 'data'),
    Input('test-points', 'data'),
    Input('selected-dataset-index', 'data'),
    Input('selected-test-index', 'data')
)
def update_outputs(ds_points_data, test_points_data, selected_ds_idx, selected_test_idx):
    # Convert store data (list of lists) to numpy arrays for figure generation
    ds_np = np.array(ds_points_data) if ds_points_data else np.empty((0, 2))
    test_np = np.array(test_points_data) if test_points_data else np.empty((0, 2))

    # Generate the figure
    figure = make_figure(ds_np, test_np)

    # Create data for tables
    ds_table_output = create_table_data(ds_points_data)
    test_table_output = create_table_data(test_points_data)

    # Removed selectedData output and related logic
    return figure, ds_table_output, test_table_output


if __name__ == '__main__':
    app.run(debug=True)
