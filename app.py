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

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For Gunicorn

# Global data storage
app_data = {}

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Semantic Visualization Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Data Controls", className="card-title"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Number of Items:"),
                            dcc.Dropdown(
                                id='limit-dropdown',
                                options=[
                                    {'label': '100', 'value': 100},
                                    {'label': '250', 'value': 250},
                                    {'label': '500', 'value': 500},
                                    {'label': '1000', 'value': 1000},
                                    {'label': '2000', 'value': 2000},
                                    {'label': '5000', 'value': 5000}
                                ],
                                value=1000,
                                clearable=False
                            )
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Label("Visualization Method:"),
                            dcc.Dropdown(
                                id='method-dropdown',
                                options=[
                                    {'label': 'UMAP', 'value': 'umap'},
                                    {'label': 'PCA', 'value': 'pca'},
                                    {'label': 't-SNE', 'value': 'tsne'}
                                ],
                                value='umap',
                                clearable=False
                            )
                        ], width=4),
                        
                        dbc.Col([
                            dbc.Label("Similarity Threshold:"),
                            html.Div([
                                dcc.Slider(
                                    id='similarity-slider',
                                    min=0.5,
                                    max=0.95,
                                    step=0.01,
                                    value=0.7,
                                    marks={
                                        0.5: '0.5',
                                        0.6: '0.6',
                                        0.7: '0.7',
                                        0.8: '0.8',
                                        0.9: '0.9'
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ])
                        ], width=4)
                    ]),
                    
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Load Data", 
                                id="load-button", 
                                color="primary", 
                                className="me-2"
                            ),
                            dbc.Button(
                                "Apply Similarity Filter", 
                                id="filter-button", 
                                color="info", 
                                className="me-2",
                                disabled=True
                            ),
                            dbc.Button(
                                "Download Data", 
                                id="download-button", 
                                color="secondary",
                                disabled=True
                            )
                        ])
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading",
                children=[
                    dcc.Graph(
                        id='scatter-plot',
                        style={'height': '70vh'},
                        config={'displayModeBar': True, 'displaylogo': False}
                    )
                ],
                type="default"
            )
        ], width=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Selected Item Details"),
                dbc.CardBody([
                    html.Div(id="item-details", children="Click on a point to see details")
                ])
            ])
        ], width=4)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="status-message", className="mt-3")
        ])
    ]),
    
    # Hidden div to store data
    html.Div(id='stored-data', style={'display': 'none'}),
    
    # Download component
    dcc.Download(id="download-data")
    
], fluid=True)

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
                size=8,
                color=np.arange(len(df)),
                colorscale='Viridis',
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=df['name'],
            hovertemplate='<b>%{text}</b><br>' +
                         'ID: %{customdata[0]}<br>' +
                         'Description: %{customdata[1]}<br>' +
                         '<extra></extra>',
            customdata=np.column_stack((df['id'], df['description'])),
            name='Items'
        ))
        
        fig.update_layout(
            title=f"Semantic Visualization ({method.upper()}) - {len(processed_data)} items",
            xaxis_title=f"{method.upper()} Dimension 1",
            yaxis_title=f"{method.upper()} Dimension 2",
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        success_status = dbc.Alert(
            f"Successfully loaded {len(processed_data)} items using {method.upper()}", 
            color="success"
        )
        
        return fig, json.dumps(processed_data), success_status, False, False
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            title=f"Error: {str(e)}",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2"
        )
        
        error_status = dbc.Alert(f"Error loading data: {str(e)}", color="danger")
        
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
                size=8,
                color=color_values,
                colorscale='Plasma',
                opacity=0.7,
                line=dict(width=0.5, color='white'),
                colorbar=dict(title="Similar Items Count"),
                cmin=0,
                cmax=max(color_values) if color_values else 1
            ),
            text=df['name'],
            hovertemplate='<b>%{text}</b><br>' +
                         'ID: %{customdata[0]}<br>' +
                         'Description: %{customdata[1]}<br>' +
                         'Similar Items: %{customdata[2]}<br>' +
                         'Max Similarity: %{customdata[3]:.3f}<br>' +
                         '<extra></extra>',
            customdata=np.column_stack((
                df['id'], 
                df['description'],
                [item.get('similar_count', 0) for item in processed_data],
                [item.get('max_similarity', 0) for item in processed_data]
            )),
            name='Items'
        ))
        
        fig.update_layout(
            title=f"Semantic Visualization ({method.upper()}) - Similarity ≥ {similarity_threshold:.2f}",
            xaxis_title=f"{method.upper()} Dimension 1",
            yaxis_title=f"{method.upper()} Dimension 2",
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        success_status = dbc.Alert(
            f"Applied similarity filter ≥ {similarity_threshold:.2f}. Points colored by similar item count.", 
            color="info"
        )
        
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
            html.H6(item['name'], className="text-primary"),
            html.P(f"ID: {item['id']}", className="text-muted small"),
            html.Hr(),
            html.P(item['description']),
            html.Hr(),
            html.P(f"Coordinates: ({item['x']:.3f}, {item['y']:.3f})", className="small text-muted")
        ]
        
        # Add similarity info if available
        if 'similar_count' in item:
            details.insert(-1, html.P(f"Similar Items: {item['similar_count']}", className="small text-info"))
        if 'max_similarity' in item:
            details.insert(-1, html.P(f"Max Similarity: {item['max_similarity']:.3f}", className="small text-info"))
        
        if item.get('image_url'):
            details.insert(2, html.Img(
                src=item['image_url'], 
                style={'width': '100%', 'max-width': '200px', 'height': 'auto'},
                className="mb-2"
            ))
        
        return details
        
    except Exception as e:
        return f"Error displaying details: {str(e)}"

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