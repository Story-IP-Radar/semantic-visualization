import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import json
import os
from google.cloud import bigquery
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticVisualization:
    def __init__(self):
        self.bigquery_client = None
        self.nodes_df = None
        self.links_df = None
        self.graph = None
        self.positions = None
        
    def init_bigquery(self):
        """Initialize BigQuery client"""
        try:
            credentials = {
                "type": os.getenv("GOOGLE_CLOUD_SERVICE_ACCOUNT_TYPE"),
                "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
                "private_key_id": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY_ID"),
                "private_key": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY", "").replace("\\n", "\n"),
                "client_email": os.getenv("GOOGLE_CLOUD_CLIENT_EMAIL"),
                "client_id": os.getenv("GOOGLE_CLOUD_CLIENT_ID"),
            }
            
            self.bigquery_client = bigquery.Client.from_service_account_info(credentials)
            logger.info("BigQuery client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery: {e}")
            # For development, you can load from local file
            self.load_sample_data()
    
    def load_sample_data(self):
        """Load sample data for development/testing"""
        logger.info("Loading sample data...")
        np.random.seed(42)
        
        # Generate sample embeddings and metadata
        n_samples = 100
        embedding_dim = 384
        
        embeddings = np.random.randn(n_samples, embedding_dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        self.nodes_df = pd.DataFrame({
            'id': [f'asset_{i}' for i in range(n_samples)],
            'label': [f'Asset {i}' for i in range(n_samples)],
            'description': [f'Description for asset {i}' for i in range(n_samples)],
            'imageUrl': [f'https://via.placeholder.com/150?text=Asset{i}' for i in range(n_samples)],
            'embedding': [emb.tolist() for emb in embeddings]
        })
    
    def fetch_data_from_bigquery(self, limit=500):
        """Fetch data from BigQuery"""
        if not self.bigquery_client:
            self.load_sample_data()
            return
            
        try:
            query = f"""
            SELECT 
                v.id, v.embedding, v.descriptionText, a.nftMetadata
            FROM `storygraph-462415.storygraph.vector_embeddings_external` v
            LEFT JOIN `storygraph-462415.storygraph.assets_external` a ON v.id = a.id
            LIMIT {limit}
            """
            
            logger.info(f"Fetching {limit} records from BigQuery...")
            query_job = self.bigquery_client.query(query)
            rows = query_job.result()
            
            data = []
            for row in rows:
                nft_metadata = json.loads(row.nftMetadata) if row.nftMetadata else {}
                data.append({
                    'id': row.id,
                    'label': nft_metadata.get('name', str(row.id)),
                    'description': row.descriptionText or 'No description',
                    'imageUrl': nft_metadata.get('imageUrl', ''),
                    'embedding': row.embedding
                })
            
            self.nodes_df = pd.DataFrame(data)
            logger.info(f"Loaded {len(self.nodes_df)} nodes from BigQuery")
            
        except Exception as e:
            logger.error(f"Error fetching from BigQuery: {e}")
            self.load_sample_data()
    
    def calculate_similarities_and_links(self, threshold=0.7, max_links=7500):
        """Calculate cosine similarities and create links"""
        if self.nodes_df is None:
            return
            
        logger.info("Calculating similarities...")
        embeddings = np.array(self.nodes_df['embedding'].tolist())
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create links dataframe
        links = []
        node_ids = self.nodes_df['id'].tolist()
        
        for i in range(len(node_ids)):
            if len(links) >= max_links:
                break
            for j in range(i + 1, len(node_ids)):
                if len(links) >= max_links:
                    break
                
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    links.append({
                        'source': node_ids[i],
                        'target': node_ids[j],
                        'similarity': similarity,
                        'distance': 1 - similarity
                    })
        
        self.links_df = pd.DataFrame(links)
        logger.info(f"Created {len(self.links_df)} links with threshold {threshold}")
    
    def create_network_layout(self):
        """Create network layout using NetworkX and force-directed positioning"""
        if self.nodes_df is None or self.links_df is None:
            return
            
        logger.info("Creating network layout...")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for _, node in self.nodes_df.iterrows():
            G.add_node(node['id'], **node.to_dict())
        
        # Add edges
        for _, link in self.links_df.iterrows():
            G.add_edge(link['source'], link['target'], 
                      weight=link['similarity'], distance=link['distance'])
        
        # Calculate layout
        if len(G.nodes()) > 0:
            # Use spring layout with distance as edge weight
            pos = nx.spring_layout(G, weight='distance', k=3, iterations=50)
            self.positions = pos
            self.graph = G
        
        logger.info("Network layout created")
    
    def create_plotly_figure(self, selected_node=None):
        """Create the main Plotly figure"""
        if self.positions is None:
            return go.Figure()
        
        # Prepare node traces
        node_x = [self.positions[node][0] for node in self.graph.nodes()]
        node_y = [self.positions[node][1] for node in self.graph.nodes()]
        
        node_info = []
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_info.append(f"<b>{node_data['label']}</b><br>" +
                           f"ID: {node_data['id']}<br>" +
                           f"Description: {node_data['description'][:100]}...")
            
            # Color and size based on selection
            if selected_node and node == selected_node:
                node_colors.append('red')
                node_sizes.append(15)
            elif selected_node:
                # Check if connected to selected node
                if self.graph.has_edge(node, selected_node):
                    node_colors.append('orange')
                    node_sizes.append(10)
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(6)
            else:
                node_colors.append('lightblue')
                node_sizes.append(8)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[self.graph.nodes[node]['label'] for node in self.graph.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hoverinfo='text',
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='white')
            ),
            customdata=list(self.graph.nodes()),
            name="Nodes"
        )
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges():
            x0, y0 = self.positions[edge[0]]
            x1, y1 = self.positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            opacity=0.6 if not selected_node else 0.2,
            name="Links"
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                            title='Semantic Visualization',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="Click on nodes to explore connections",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor="left", yanchor="bottom",
                                font=dict(color="#888", size=12)
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='rgba(0,0,0,0.05)',
                            height=600
                        ))
        
        return fig

# Initialize the visualization
viz = SemanticVisualization()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Semantic Visualization"

# Define the layout
app.layout = html.Div([
    html.Div([
        html.H1("Semantic Visualization", className="text-3xl font-bold mb-2"),
        html.P("Explore IP assets based on their semantic similarity using vector embeddings. "
               "Assets with similar content are positioned closer together in this interactive visualization.",
               className="text-gray-600 mb-4"),
        
        html.Div([
            html.H3("How it works", className="font-medium text-blue-900 mb-2"),
            html.P("Each IP asset is represented by a high-dimensional vector embedding that captures its semantic meaning. "
                   "We calculate cosine similarity between these vectors to determine how related different assets are, "
                   "then use a force-directed graph to visualize these relationships spatially.",
                   className="text-sm text-blue-800")
        ], className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6"),
        
        # Controls
        html.Div([
            html.Div([
                html.Label("Max Nodes:", className="text-sm font-medium mr-2"),
                dcc.Dropdown(
                    id='max-nodes-dropdown',
                    options=[
                        {'label': '100', 'value': 100},
                        {'label': '250', 'value': 250},
                        {'label': '500', 'value': 500},
                        {'label': '1000', 'value': 1000},
                        {'label': '2000', 'value': 2000},
                        {'label': '5000', 'value': 5000}
                    ],
                    value=500,
                    style={'width': '120px', 'display': 'inline-block'}
                )
            ], className="flex items-center mr-4"),
            
            html.Div([
                html.Label("Similarity Threshold:", className="text-sm font-medium mr-2"),
                dcc.Slider(
                    id='threshold-slider',
                    min=0.6,
                    max=0.95,
                    step=0.01,
                    value=0.7,
                    marks={i/100: f'{i/100:.2f}' for i in range(60, 96, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '300px', 'display': 'inline-block'})
        ], className="flex items-center justify-between mb-4"),
        
        # Loading indicator
        dcc.Loading(
            id="loading",
            children=[
                dcc.Graph(id='network-graph', style={'height': '600px'})
            ],
            type="circle"
        ),
        
        # Selected node info
        html.Div(id='node-info', className="mt-4")
        
    ], className="container mx-auto p-6"),
    
    # Store for data
    dcc.Store(id='graph-data'),
    dcc.Store(id='selected-node')
])

@app.callback(
    Output('graph-data', 'data'),
    [Input('max-nodes-dropdown', 'value'),
     Input('threshold-slider', 'value')]
)
def update_graph_data(max_nodes, threshold):
    """Update graph data when parameters change"""
    try:
        # Fetch data
        viz.fetch_data_from_bigquery(limit=max_nodes)
        
        # Calculate similarities and create network
        viz.calculate_similarities_and_links(threshold=threshold)
        viz.create_network_layout()
        
        return {'status': 'success', 'timestamp': pd.Timestamp.now().isoformat()}
    except Exception as e:
        logger.error(f"Error updating graph data: {e}")
        return {'status': 'error', 'message': str(e)}

@app.callback(
    [Output('network-graph', 'figure'),
     Output('selected-node', 'data')],
    [Input('graph-data', 'data'),
     Input('network-graph', 'clickData')],
    [State('selected-node', 'data')]
)
def update_graph(graph_data, click_data, current_selected):
    """Update the network graph"""
    ctx = callback_context
    
    selected_node = current_selected
    
    # Handle node clicks
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'network-graph.clickData':
        if click_data and 'points' in click_data:
            point = click_data['points'][0]
            if 'customdata' in point:
                new_selected = point['customdata']
                selected_node = new_selected if new_selected != current_selected else None
    
    # Create figure
    fig = viz.create_plotly_figure(selected_node=selected_node)
    
    return fig, selected_node

@app.callback(
    Output('node-info', 'children'),
    [Input('selected-node', 'data')]
)
def update_node_info(selected_node):
    """Update selected node information panel"""
    if not selected_node or viz.graph is None:
        return html.Div()
    
    if selected_node in viz.graph.nodes():
        node_data = viz.graph.nodes[selected_node]
        
        return html.Div([
            html.H3(f"Selected: {node_data['label']}", className="text-lg font-bold mb-2"),
            html.P(f"ID: {node_data['id']}", className="text-sm text-gray-600 mb-1"),
            html.P(f"Description: {node_data['description']}", className="text-sm mb-2"),
            html.P(f"Connections: {len(list(viz.graph.neighbors(selected_node)))}", className="text-sm text-blue-600")
        ], className="border rounded-lg p-4 bg-gray-50")
    
    return html.Div()

if __name__ == '__main__':
    # Initialize BigQuery connection
    viz.init_bigquery()
    
    # Run the app
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)