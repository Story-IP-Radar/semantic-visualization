import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from google.cloud import bigquery
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize BigQuery client
def init_bigquery():
    try:
        # Create credentials dict from environment variables
        credentials_info = {
            "type": os.getenv("GOOGLE_CLOUD_SERVICE_ACCOUNT_TYPE"),
            "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
            "private_key_id": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.getenv("GOOGLE_CLOUD_CLIENT_EMAIL"),
            "client_id": os.getenv("GOOGLE_CLOUD_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        
        return bigquery.Client.from_service_account_info(
            credentials_info,
            project=os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        )
    except Exception as e:
        logger.error(f"Failed to initialize BigQuery client: {e}")
        raise

# Similarity calculation functions
def cosine_similarity_matrix(embeddings):
    """Calculate cosine similarity matrix for embeddings"""
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.dot(embeddings_norm, embeddings_norm.T)

def euclidean_distance_matrix(embeddings):
    """Calculate euclidean distance matrix for embeddings"""
    from sklearn.metrics.pairwise import euclidean_distances
    return euclidean_distances(embeddings)

# Data fetching and processing
def fetch_data_from_bigquery(limit=1000):
    """Fetch vector embeddings and metadata from BigQuery"""
    try:
        client = init_bigquery()
        
        query = f"""
        SELECT 
            v.id, 
            v.embedding, 
            v.descriptionText,
            a.nftMetadata
        FROM `storygraph-462415.storygraph.vector_embeddings_external` v
        LEFT JOIN `storygraph-462415.storygraph.assets_external` a ON v.id = a.id
        LIMIT {limit}
        """
        
        logger.info(f"Fetching {limit} records from BigQuery...")
        
        query_job = client.query(query)
        results = query_job.result()
        
        data = []
        for row in results:
            # Parse nftMetadata if it exists
            metadata = {}
            if row.nftMetadata:
                try:
                    if isinstance(row.nftMetadata, str):
                        metadata = json.loads(row.nftMetadata)
                    else:
                        metadata = row.nftMetadata
                except:
                    metadata = {}
            
            data.append({
                'id': row.id,
                'embedding': row.embedding,
                'description': row.descriptionText or '',
                'name': metadata.get('name', f'Item {row.id}'),
                'image_url': metadata.get('imageUrl', '')
            })
        
        logger.info(f"Successfully fetched {len(data)} records")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise

def filter_by_similarity(data, embeddings, similarity_threshold=0.7):
    """Filter data points based on similarity to others"""
    try:
        logger.info(f"Filtering data with similarity threshold {similarity_threshold}...")
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity_matrix(embeddings)
        
        # For each point, count how many other points it's similar to
        similar_counts = []
        for i in range(len(similarity_matrix)):
            # Count similarities above threshold (excluding self)
            similar_count = np.sum(similarity_matrix[i] > similarity_threshold) - 1
            similar_counts.append(similar_count)
        
        # Add similarity info to data
        filtered_data = []
        for i, item in enumerate(data):
            item_copy = item.copy()
            item_copy['similar_count'] = similar_counts[i]
            item_copy['max_similarity'] = float(np.max(similarity_matrix[i][np.arange(len(similarity_matrix)) != i]))
            filtered_data.append(item_copy)
        
        logger.info(f"Added similarity metrics to {len(filtered_data)} items")
        return filtered_data, similarity_matrix
        
    except Exception as e:
        logger.error(f"Error filtering by similarity: {e}")
        raise

def process_embeddings(data, method='umap', n_components=2):
    """Process embeddings using dimensionality reduction"""
    try:
        embeddings = np.array([item['embedding'] for item in data])
        
        logger.info(f"Processing {len(embeddings)} embeddings with {method}...")
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
        else:  # umap
            reducer = umap.UMAP(n_components=n_components, random_state=42, n_neighbors=min(15, len(embeddings)-1))
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Add reduced coordinates to data
        for i, item in enumerate(data):
            item['x'] = float(reduced_embeddings[i][0])
            item['y'] = float(reduced_embeddings[i][1])
            if n_components == 3:
                item['z'] = float(reduced_embeddings[i][2])
        
        return data, embeddings
        
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        raise

# Initialize Dash app with custom styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For Gunicorn

# Custom CSS styling to match your D3 aesthetic
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
                color: #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                margin: 0;
                padding: 0;
            }
            
            .main-container {
                background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .card-custom {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid #475569;
                border-radius: 12px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(12px);
            }
            
            .glass-effect {
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
            }
            
            .btn-primary-custom {
                background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
                border: none;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                transition: all 0.2s ease;
            }
            
            .btn-primary-custom:hover {
                box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
                transform: translateY(-1px);
            }
            
            .btn-secondary-custom {
                background: rgba(71, 85, 105, 0.8);
                border: 1px solid #475569;
                color: #e2e8f0;
                border-radius: 8px;
                padding: 8px 16px;
                transition: all 0.2s ease;
            }
            
            .btn-secondary-custom:hover {
                background: rgba(71, 85, 105, 1);
                border-color: #8b5cf6;
            }
            
            .plot-container {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border: 1px solid #475569;
                border-radius: 12px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }
            
            .info-card {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border: 1px solid #475569;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            
            .metric-value {
                color: #8b5cf6;
                font-weight: bold;
                font-size: 1.5rem;
            }
            
            .metric-label {
                color: #94a3b8;
                font-size: 0.875rem;
                margin-bottom: 4px;
            }
            
            .similarity-indicator {
                color: #06b6d4;
                font-weight: 600;
            }
            
            .status-success {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: #10b981;
                border-radius: 8px;
            }
            
            .status-info {
                background: rgba(6, 182, 212, 0.1);
                border: 1px solid rgba(6, 182, 212, 0.3);
                color: #06b6d4;
                border-radius: 8px;
            }
            
            .status-warning {
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.3);
                color: #f59e0b;
                border-radius: 8px;
            }
            
            .status-error {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                color: #ef4444;
                border-radius: 8px;
            }
            
            /* Custom slider styling */
            .rc-slider {
                background-color: #475569 !important;
            }
            
            .rc-slider-track {
                background: linear-gradient(90deg, #8b5cf6, #a855f7) !important;
            }
            
            .rc-slider-handle {
                background: linear-gradient(135deg, #8b5cf6, #a855f7) !important;
                border: 2px solid #a855f7 !important;
                box-shadow: 0 0 10px rgba(139, 92, 246, 0.3) !important;
            }
            
            /* Dropdown styling */
            .Select-control {
                background: rgba(30, 41, 59, 0.8) !important;
                border: 1px solid #475569 !important;
                color: #e2e8f0 !important;
            }
            
            .Select-menu-outer {
                background: #1e293b !important;
                border: 1px solid #475569 !important;
            }
            
            .Select-option {
                background: #1e293b !important;
                color: #e2e8f0 !important;
            }
            
            .Select-option:hover {
                background: #334155 !important;
            }
            
            /* Loading spinner */
            .loading-spinner {
                border: 3px solid #475569;
                border-top: 3px solid #8b5cf6;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Glow effects */
            .glow-primary {
                box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
            }
            
            .glow-hover:hover {
                box-shadow: 0 0 25px rgba(139, 92, 246, 0.4);
                transition: box-shadow 0.3s ease;
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Global data storage
app_data = {}

# Layout
app.layout = html.Div([
    # Header Section
    html.Div([
        html.H1("Semantic Visualization Dashboard", 
                style={'color': '#e2e8f0', 'margin-bottom': '8px', 'font-weight': '700', 'font-size': '2.5rem'}),
        html.P("Explore semantic relationships in your data through interactive vector embeddings visualization.",
               style={'color': '#94a3b8', 'margin-bottom': '24px', 'font-size': '1.1rem'}),
        
        # Info card explaining how it works
        html.Div([
            html.H3("How it works", style={'color': '#8b5cf6', 'margin-bottom': '12px', 'font-weight': '600'}),
            html.P([
                "Each item is represented by a high-dimensional vector embedding that captures its semantic meaning. ",
                "We calculate cosine similarity between these vectors to determine relationships, then use dimensionality ",
                "reduction (UMAP, PCA, or t-SNE) to visualize these relationships in 2D space."
            ], style={'color': '#cbd5e1', 'font-size': '0.95rem', 'line-height': '1.5'})
        ], className="info-card", style={'margin-bottom': '24px'})
    ], style={'margin-bottom': '32px'}),
    
    # Controls Section
    html.Div([
        html.Div([
            html.H5("Visualization Controls", style={'color': '#e2e8f0', 'margin-bottom': '20px', 'font-weight': '600'}),
            
            # First row of controls
            dbc.Row([
                dbc.Col([
                    html.Label("Dataset Size:", style={'color': '#94a3b8', 'font-weight': '500', 'margin-bottom': '8px'}),
                    dcc.Dropdown(
                        id='limit-dropdown',
                        options=[
                            {'label': '100 items', 'value': 100},
                            {'label': '250 items', 'value': 250},
                            {'label': '500 items', 'value': 500},
                            {'label': '1,000 items', 'value': 1000},
                            {'label': '2,000 items', 'value': 2000},
                            {'label': '5,000 items', 'value': 5000}
                        ],
                        value=1000,
                        clearable=False,
                        style={'margin-bottom': '16px'}
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Reduction Method:", style={'color': '#94a3b8', 'font-weight': '500', 'margin-bottom': '8px'}),
                    dcc.Dropdown(
                        id='method-dropdown',
                        options=[
                            {'label': 'UMAP (Recommended)', 'value': 'umap'},
                            {'label': 'PCA (Fastest)', 'value': 'pca'},
                            {'label': 't-SNE (High Quality)', 'value': 'tsne'}
                        ],
                        value='umap',
                        clearable=False,
                        style={'margin-bottom': '16px'}
                    )
                ], width=4),
                
                dbc.Col([
                    html.Label("Similarity Threshold:", style={'color': '#94a3b8', 'font-weight': '500', 'margin-bottom': '8px'}),
                    html.Div([
                        dcc.Slider(
                            id='similarity-slider',
                            min=0.5,
                            max=0.95,
                            step=0.01,
                            value=0.7,
                            marks={
                                0.5: {'label': '0.5', 'style': {'color': '#94a3b8'}},
                                0.6: {'label': '0.6', 'style': {'color': '#94a3b8'}},
                                0.7: {'label': '0.7', 'style': {'color': '#8b5cf6'}},
                                0.8: {'label': '0.8', 'style': {'color': '#94a3b8'}},
                                0.9: {'label': '0.9', 'style': {'color': '#94a3b8'}}
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'padding': '0 10px', 'margin-bottom': '16px'})
                ], width=4)
            ]),
            
            # Action buttons
            html.Div([
                html.Button(
                    "Load Data", 
                    id="load-button", 
                    className="btn-primary-custom glow-hover",
                    style={'margin-right': '12px', 'font-weight': '500'}
                ),
                html.Button(
                    "Apply Similarity Filter", 
                    id="filter-button", 
                    className="btn-secondary-custom",
                    disabled=True,
                    style={'margin-right': '12px', 'font-weight': '500'}
                ),
                html.Button(
                    "Download Data", 
                    id="download-button", 
                    className="btn-secondary-custom",
                    disabled=True,
                    style={'font-weight': '500'}
                )
            ], style={'margin-top': '20px'})
        ], className="card-custom", style={'padding': '24px'})
    ], style={'margin-bottom': '24px'}),
    
    # Visualization Section
    dbc.Row([
        # Main plot
        dbc.Col([
            html.Div([
                dcc.Loading(
                    id="loading",
                    children=[
                        dcc.Graph(
                            id='scatter-plot',
                            style={'height': '75vh', 'background': 'transparent'},
                            config={
                                'displayModeBar': True, 
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                            }
                        )
                    ],
                    type="circle",
                    color="#8b5cf6"
                )
            ], className="plot-container glow-primary")
        ], width=8),
        
        # Details sidebar
        dbc.Col([
            html.Div([
                html.H5("Item Details", style={'color': '#e2e8f0', 'margin-bottom': '16px', 'font-weight': '600'}),
                html.Div(id="item-details", children=[
                    html.Div([
                        html.P("Click on any point in the visualization to explore its details and relationships.",
                               style={'color': '#94a3b8', 'text-align': 'center', 'padding': '20px', 'font-style': 'italic'})
                    ])
                ])
            ], className="card-custom", style={'padding': '20px', 'height': '75vh', 'overflow-y': 'auto'})
        ], width=4)
    ]),
    
    # Status message
    html.Div(id="status-message", style={'margin-top': '24px'}),
    
    # Hidden storage
    html.Div(id='stored-data', style={'display': 'none'}),
    dcc.Download(id="download-data")
    
], style={'max-width': '1400px', 'margin': '0 auto', 'padding': '20px'})

# Callbacks
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('stored-data', 'children'),
     Output('status-message', 'children'),
     Output('download-button', 'disabled'),
     Output('filter-button', 'disabled')],
    [Input('load-button', 'n_clicks')],
    [State('limit-dropdown', 'value'),
     State('method-dropdown', 'value')]
)
def update_plot(n_clicks, limit, method):
    if n_clicks is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Click 'Load Data' to start visualization",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2"
        )
        return empty_fig, "", "", True, True
    
    try:
        # Fetch and process data
        status = dbc.Alert("Loading data from BigQuery...", color="info")
        
        raw_data = fetch_data_from_bigquery(limit)
        processed_data, embeddings = process_embeddings(raw_data, method)
        
        # Store data globally
        app_data['processed_data'] = processed_data
        app_data['embeddings'] = embeddings
        app_data['original_data'] = raw_data
        
        # Create scatter plot
        df = pd.DataFrame(processed_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                size=10,
                color=np.arange(len(df)),
                colorscale='Viridis',
                opacity=0.8,
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                symbol='circle'
            ),
            text=df['name'],
            hovertemplate='<b style="color: #8b5cf6;">%{text}</b><br>' +
                         '<span style="color: #94a3b8;">ID:</span> %{customdata[0]}<br>' +
                         '<span style="color: #94a3b8;">Description:</span> %{customdata[1]}<br>' +
                         '<extra></extra>',
            customdata=np.column_stack((df['id'], df['description'])),
            name='Items',
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Semantic Visualization ({method.upper()}) - {len(processed_data)} items",
                x=0.5,
                font=dict(size=20, color='#e2e8f0', family='Arial, sans-serif')
            ),
            xaxis=dict(
                title=f"{method.upper()} Dimension 1",
                titlefont=dict(color='#94a3b8', size=14),
                tickfont=dict(color='#94a3b8'),
                gridcolor='rgba(148, 163, 184, 0.2)',
                zerolinecolor='rgba(148, 163, 184, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                title=f"{method.upper()} Dimension 2",
                titlefont=dict(color='#94a3b8', size=14),
                tickfont=dict(color='#94a3b8'),
                gridcolor='rgba(148, 163, 184, 0.2)',
                zerolinecolor='rgba(148, 163, 184, 0.3)',
                showgrid=True
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            height=600,
            font=dict(family='Arial, sans-serif')
        )
        
        success_status = html.Div([
            html.I(className="fas fa-check-circle", style={'margin-right': '8px'}),
            f"Successfully loaded {len(processed_data)} items using {method.upper()}"
        ], className="status-success", style={'padding': '12px', 'margin': '8px 0'})
        
        return fig, json.dumps(processed_data), success_status, False, False
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            title=dict(
                text=f"Error: {str(e)}",
                x=0.5,
                font=dict(size=18, color='#ef4444', family='Arial, sans-serif')
            ),
            xaxis=dict(
                title="Dimension 1",
                titlefont=dict(color='#94a3b8'),
                tickfont=dict(color='#94a3b8'),
                gridcolor='rgba(148, 163, 184, 0.2)'
            ),
            yaxis=dict(
                title="Dimension 2", 
                titlefont=dict(color='#94a3b8'),
                tickfont=dict(color='#94a3b8'),
                gridcolor='rgba(148, 163, 184, 0.2)'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        error_status = html.Div([
            html.I(className="fas fa-exclamation-triangle", style={'margin-right': '8px'}),
            f"Error loading data: {str(e)}"
        ], className="status-error", style={'padding': '12px', 'margin': '8px 0'})
        
        return error_fig, "", error_status, True, True

@app.callback(
    [Output('scatter-plot', 'figure', allow_duplicate=True),
     Output('stored-data', 'children', allow_duplicate=True),
     Output('status-message', 'children', allow_duplicate=True)],
    [Input('filter-button', 'n_clicks')],
    [State('similarity-slider', 'value'),
     State('method-dropdown', 'value')],
    prevent_initial_call=True
)
def apply_similarity_filter(n_clicks, similarity_threshold, method):
    if n_clicks is None or 'processed_data' not in app_data:
        return dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Get stored data
        original_data = app_data['original_data']
        embeddings = app_data['embeddings']
        
        # Apply similarity filtering
        filtered_data, similarity_matrix = filter_by_similarity(
            original_data, embeddings, similarity_threshold
        )
        
        # Re-process embeddings for visualization
        processed_data, _ = process_embeddings(filtered_data, method)
        
        # Update stored data
        app_data['processed_data'] = processed_data
        app_data['similarity_matrix'] = similarity_matrix
        
        # Create updated scatter plot
        df = pd.DataFrame(processed_data)
        
        fig = go.Figure()
        
        # Color by similarity count or max similarity
        color_values = [item.get('similar_count', 0) for item in processed_data]
        
        fig.add_trace(go.Scatter(
            x=df['x'],
            y=df['y'],
            mode='markers',
            marker=dict(
                size=10,
                color=color_values,
                colorscale='Plasma',
                opacity=0.8,
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                colorbar=dict(
                    title=dict(text="Similar Items", font=dict(color='#94a3b8')),
                    titleside="right",
                    tickfont=dict(color='#94a3b8'),
                    outlinecolor='rgba(148, 163, 184, 0.3)',
                    bgcolor='rgba(0,0,0,0)'
                ),
                cmin=0,
                cmax=max(color_values) if color_values else 1,
                symbol='circle'
            ),
            text=df['name'],
            hovertemplate='<b style="color: #8b5cf6;">%{text}</b><br>' +
                         '<span style="color: #94a3b8;">ID:</span> %{customdata[0]}<br>' +
                         '<span style="color: #94a3b8;">Description:</span> %{customdata[1]}<br>' +
                         '<span style="color: #06b6d4;">Similar Items:</span> %{customdata[2]}<br>' +
                         '<span style="color: #06b6d4;">Max Similarity:</span> %{customdata[3]:.3f}<br>' +
                         '<extra></extra>',
            customdata=np.column_stack((
                df['id'], 
                df['description'],
                [item.get('similar_count', 0) for item in processed_data],
                [item.get('max_similarity', 0) for item in processed_data]
            )),
            name='Items',
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text=f"Semantic Visualization ({method.upper()}) - Similarity ≥ {similarity_threshold:.2f}",
                x=0.5,
                font=dict(size=20, color='#e2e8f0', family='Arial, sans-serif')
            ),
            xaxis=dict(
                title=f"{method.upper()} Dimension 1",
                titlefont=dict(color='#94a3b8', size=14),
                tickfont=dict(color='#94a3b8'),
                gridcolor='rgba(148, 163, 184, 0.2)',
                zerolinecolor='rgba(148, 163, 184, 0.3)',
                showgrid=True
            ),
            yaxis=dict(
                title=f"{method.upper()} Dimension 2",
                titlefont=dict(color='#94a3b8', size=14),
                tickfont=dict(color='#94a3b8'),
                gridcolor='rgba(148, 163, 184, 0.2)',
                zerolinecolor='rgba(148, 163, 184, 0.3)',
                showgrid=True
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='closest',
            height=600,
            font=dict(family='Arial, sans-serif')
        )
        
        success_status = html.Div([
            html.I(className="fas fa-info-circle", style={'margin-right': '8px'}),
            f"Applied similarity filter ≥ {similarity_threshold:.2f}. Points colored by similar item count."
        ], className="status-info", style={'padding': '12px', 'margin': '8px 0'})
        
        return fig, json.dumps(processed_data), success_status
        
    except Exception as e:
        error_status = dbc.Alert(f"Error applying similarity filter: {str(e)}", color="danger")
        return dash.no_update, dash.no_update, error_status

@app.callback(
    Output('item-details', 'children'),
    [Input('scatter-plot', 'clickData')],
    [State('stored-data', 'children')]
)
def update_item_details(clickData, stored_data):
    if clickData is None or not stored_data:
        return "Click on a point to see details"
    
    try:
        data = json.loads(stored_data)
        point_index = clickData['points'][0]['pointIndex']
        item = data[point_index]
        


        details = [
            html.H6(item['name'], style={'color': '#8b5cf6', 'margin-bottom': '8px', 'font-weight': '600'}),
            html.P(f"ID: {item['id']}", style={'color': '#94a3b8', 'font-size': '0.85rem', 'margin-bottom': '12px', 'font-family': 'monospace'}),
            html.Hr(style={'border-color': '#475569', 'margin': '12px 0'}),
            html.Div([
                html.P("Description:", style={'color': '#94a3b8', 'font-weight': '500', 'margin-bottom': '4px'}),
                html.P(item['description'], style={'color': '#e2e8f0', 'line-height': '1.5', 'margin-bottom': '12px'})
            ]),
            html.Hr(style={'border-color': '#475569', 'margin': '12px 0'}),
            html.P(f"Coordinates: ({item['x']:.3f}, {item['y']:.3f})", 
                   style={'color': '#94a3b8', 'font-size': '0.8rem', 'font-family': 'monospace'})
        ]
        
        # Add similarity info if available
        if 'similar_count' in item:
            details.insert(-2, html.Div([
                html.P("Similar Items:", style={'color': '#06b6d4', 'font-weight': '500', 'margin-bottom': '4px'}),
                html.P(str(item['similar_count']), style={'color': '#e2e8f0', 'font-size': '1.2rem', 'font-weight': '600'})
            ], style={'margin-bottom': '8px'}))
            
        if 'max_similarity' in item:
            details.insert(-2, html.Div([
                html.P("Max Similarity:", style={'color': '#06b6d4', 'font-weight': '500', 'margin-bottom': '4px'}),
                html.P(f"{item['max_similarity']:.3f}", style={'color': '#e2e8f0', 'font-size': '1.2rem', 'font-weight': '600'})
            ], style={'margin-bottom': '12px'}))
        
        if item.get('image_url'):
            details.insert(2, html.Img(
                src=item['image_url'], 
                style={
                    'width': '100%', 
                    'max-width': '200px', 
                    'height': 'auto',
                    'border-radius': '8px',
                    'border': '1px solid #475569',
                    'margin-bottom': '12px'
                }
            ))
        
        return details
        
    except Exception as e:
        return html.Div([
            html.P("Error displaying details", style={'color': '#ef4444', 'font-weight': '500'}),
            html.P(str(e), style={'color': '#94a3b8', 'font-size': '0.85rem'})
        ])

@app.callback(
    Output("download-data", "data"),
    [Input("download-button", "n_clicks")],
    [State('stored-data', 'children')],
    prevent_initial_call=True
)
def download_data(n_clicks, stored_data):
    if n_clicks and stored_data:
        try:
            data = json.loads(stored_data)
            df = pd.DataFrame(data)
            return dcc.send_data_frame(df.to_csv, "semantic_visualization_data.csv")
        except:
            return None
    return None

if __name__ == '__main__':
    # For local development
    app.run(debug=os.getenv('DASH_DEBUG', 'False').lower() == 'true', 
            host='0.0.0.0', 
            port=int(os.getenv('PORT', 8050)))