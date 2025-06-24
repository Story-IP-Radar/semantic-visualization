import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from google.cloud import bigquery
import logging
import time
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manual .env file loading
def load_env_file():
    """Load .env file manually"""
    try:
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"').strip("'")
                    os.environ[key] = value
        logger.info("Successfully loaded .env file")
    except FileNotFoundError:
        logger.error("No .env file found")
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")

load_env_file()

def cosine_similarity_custom(vec_a, vec_b):
    """Calculate cosine similarity - matches your JS implementation exactly"""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length")
    
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def similarity_to_distance(similarity):
    """Convert similarity to distance - matches your JS implementation exactly"""
    return 1 - similarity

class ExactD3ForceSimulation:
    """EXACT replica of your D3 force simulation - creates the ring pattern"""
    
    def __init__(self, nodes, links, width=1200, height=800):
        self.width = width
        self.height = height
        
        # Initialize nodes with D3's exact initialization pattern
        self.nodes = []
        for i, node in enumerate(nodes):
            # D3 starts nodes randomly but we can improve with slight structure
            x = width/2 + (np.random.random() - 0.5) * width * 0.8
            y = height/2 + (np.random.random() - 0.5) * height * 0.8
            
            self.nodes.append({
                'id': node['id'],
                'x': x,
                'y': y,
                'vx': 0,
                'vy': 0,
                'fx': None,  # Fixed position (for dragging)
                'fy': None,
                **node
            })
        
        self.links = links
        
        # EXACT D3 force parameters from your code
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 1 - pow(self.alpha_min, 1 / 300)  # D3's default
        self.velocity_decay = 0.6
        
        # Create node lookup
        self.node_by_id = {node['id']: i for i, node in enumerate(self.nodes)}
        
    def run_simulation(self, iterations=300):
        """Run D3 force simulation with exact D3 behavior"""
        
        for iteration in range(iterations):
            # D3's alpha decay
            self.alpha += (self.alpha_min - self.alpha) * self.alpha_decay
            
            # Apply forces in D3 order
            self._apply_link_force()
            self._apply_charge_force()
            self._apply_center_force()
            self._apply_collision_force()
            
            # Integrate positions (D3's verlet integration)
            self._integrate_positions()
            
            # Early termination if alpha is small enough
            if self.alpha < self.alpha_min:
                break
                
        return [(node['x'], node['y']) for node in self.nodes]
    
    def _apply_link_force(self):
        """Apply link forces - EXACTLY like your D3 forceLink"""
        strength = 0.5  # Your D3: .strength(0.5)
        
        for link in self.links:
            source_idx = self.node_by_id.get(link['source'])
            target_idx = self.node_by_id.get(link['target'])
            
            if source_idx is None or target_idx is None:
                continue
                
            source = self.nodes[source_idx]
            target = self.nodes[target_idx]
            
            dx = target['x'] - source['x']
            dy = target['y'] - source['y']
            distance = max(1e-6, math.sqrt(dx*dx + dy*dy))
            
            # EXACT D3 calculation: similarityToDistance(d.similarity) * 250
            target_distance = similarity_to_distance(link['similarity']) * 250
            
            # D3's force calculation
            force = (distance - target_distance) / distance * self.alpha * strength
            
            fx = dx * force
            fy = dy * force
            
            # Apply forces (D3 style)
            target['vx'] -= fx
            target['vy'] -= fy
            source['vx'] += fx
            source['vy'] += fy
    
    def _apply_charge_force(self):
        """Apply charge forces - EXACTLY like your D3 forceManyBody"""
        strength = -400  # Your D3: .strength(-400)
        
        for i in range(len(self.nodes)):
            node_i = self.nodes[i]
            for j in range(i + 1, len(self.nodes)):
                node_j = self.nodes[j]
                
                dx = node_j['x'] - node_i['x']
                dy = node_j['y'] - node_i['y']
                distance_sq = dx*dx + dy*dy
                
                if distance_sq == 0:
                    # D3's handling of zero distance
                    dx = np.random.random() * 1e-6
                    dy = np.random.random() * 1e-6
                    distance_sq = dx*dx + dy*dy
                
                distance = math.sqrt(distance_sq)
                
                # D3's many-body force calculation
                force = strength * self.alpha / distance_sq
                
                fx = dx * force
                fy = dy * force
                
                # Apply forces
                node_i['vx'] -= fx
                node_i['vy'] -= fy
                node_j['vx'] += fx
                node_j['vy'] += fy
    
    def _apply_center_force(self):
        """Apply center force - EXACTLY like your D3 forceCenter"""
        center_x = self.width / 2
        center_y = self.height / 2
        
        # Calculate center of mass
        sx = sum(node['x'] for node in self.nodes)
        sy = sum(node['y'] for node in self.nodes)
        n = len(self.nodes)
        
        if n == 0:
            return
        
        # D3's centering calculation
        sx = (sx / n - center_x) * self.alpha
        sy = (sy / n - center_y) * self.alpha
        
        # Apply to all nodes
        for node in self.nodes:
            node['vx'] -= sx
            node['vy'] -= sy
    
    def _apply_collision_force(self):
        """Apply collision force - EXACTLY like your D3 forceCollide"""
        radius = 25  # Your D3: .radius(25)
        strength = 0.7
        
        for i in range(len(self.nodes)):
            node_i = self.nodes[i]
            for j in range(i + 1, len(self.nodes)):
                node_j = self.nodes[j]
                
                dx = node_j['x'] - node_i['x']
                dy = node_j['y'] - node_i['y']
                distance = math.sqrt(dx*dx + dy*dy)
                
                combined_radius = radius * 2
                
                if distance < combined_radius:
                    if distance == 0:
                        distance = 1e-6
                        dx = np.random.random() * 1e-6
                        dy = np.random.random() * 1e-6
                    
                    # D3's collision force
                    force = (combined_radius - distance) / distance * self.alpha * strength
                    
                    fx = dx * force * 0.5
                    fy = dy * force * 0.5
                    
                    node_j['vx'] += fx
                    node_j['vy'] += fy
                    node_i['vx'] -= fx
                    node_i['vy'] -= fy
    
    def _integrate_positions(self):
        """Integrate positions - D3's verlet integration"""
        for node in self.nodes:
            # Skip if position is fixed
            if node['fx'] is not None:
                node['x'] = node['fx']
                node['vx'] = 0
            else:
                # Apply velocity decay and update position
                node['vx'] *= self.velocity_decay
                node['x'] += node['vx']
                
            if node['fy'] is not None:
                node['y'] = node['fy']
                node['vy'] = 0
            else:
                node['vy'] *= self.velocity_decay
                node['y'] += node['vy']

class SemanticVisualization:
    def __init__(self):
        self.bigquery_client = None
        self.raw_data_cache = {}
        
    def init_bigquery(self):
        """Initialize BigQuery client"""
        try:
            required_vars = {
                "GOOGLE_CLOUD_PROJECT_ID": os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
                "GOOGLE_CLOUD_CLIENT_EMAIL": os.getenv("GOOGLE_CLOUD_CLIENT_EMAIL"),
                "GOOGLE_CLOUD_PRIVATE_KEY": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY")
            }
            
            missing_vars = [k for k, v in required_vars.items() if not v]
            
            if missing_vars:
                logger.error(f"Missing BigQuery environment variables: {missing_vars}")
                logger.warning("Using sample data")
                return
            
            credentials = {
                "type": os.getenv("GOOGLE_CLOUD_SERVICE_ACCOUNT_TYPE", "service_account"),
                "project_id": os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
                "private_key_id": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY_ID"),
                "private_key": os.getenv("GOOGLE_CLOUD_PRIVATE_KEY").replace("\\n", "\n"),
                "client_email": os.getenv("GOOGLE_CLOUD_CLIENT_EMAIL"),
                "client_id": os.getenv("GOOGLE_CLOUD_CLIENT_ID"),
                "auth_uri": os.getenv("GOOGLE_CLOUD_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
                "token_uri": os.getenv("GOOGLE_CLOUD_TOKEN_URI", "https://oauth2.googleapis.com/token"),
                "auth_provider_x509_cert_url": os.getenv("GOOGLE_CLOUD_AUTH_PROVIDER_X509_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
                "client_x509_cert_url": os.getenv("GOOGLE_CLOUD_CLIENT_X509_CERT_URL"),
                "universe_domain": os.getenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN", "googleapis.com")
            }
            
            logger.info("Initializing BigQuery client...")
            self.bigquery_client = bigquery.Client.from_service_account_info(credentials)
            
            # Test connection
            test_query = "SELECT COUNT(*) as count FROM `storygraph-462415.storygraph.vector_embeddings_external` LIMIT 1"
            result = list(self.bigquery_client.query(test_query).result())
            logger.info(f"BigQuery connection successful! Found {result[0].count} total records")
                
        except Exception as e:
            logger.error(f"BigQuery initialization failed: {e}")
            self.bigquery_client = None
    
    def fetch_graph_data(self, limit=500, threshold=0.7):
        """Fetch graph data"""
        if self.bigquery_client:
            try:
                return self._fetch_from_bigquery(limit, threshold)
            except Exception as e:
                logger.error(f"BigQuery fetch failed: {e}")
        
        logger.warning("Using sample data")
        return self._generate_sample_data(limit, threshold)
    
    def _fetch_from_bigquery(self, limit, threshold):
        """Fetch real data from BigQuery"""
        cache_key = f"bigquery_{limit}"
        
        if cache_key not in self.raw_data_cache:
            logger.info(f"Fetching {limit} records from BigQuery...")
            
            query = f"""
            SELECT 
                v.id, v.embedding, v.descriptionText, a.nftMetadata
            FROM `storygraph-462415.storygraph.vector_embeddings_external` v
            LEFT JOIN `storygraph-462415.storygraph.assets_external` a ON v.id = a.id
            LIMIT {limit}
            """
            
            start_time = time.time()
            query_job = self.bigquery_client.query(query)
            rows = list(query_job.result())
            fetch_time = time.time() - start_time
            
            logger.info(f"Fetched {len(rows)} rows from BigQuery in {fetch_time:.2f}s")
            
            raw_data = []
            for row in rows:
                try:
                    nft_metadata = json.loads(row.nftMetadata) if row.nftMetadata else {}
                except:
                    nft_metadata = {}
                    
                raw_data.append({
                    'id': row.id,
                    'embedding': row.embedding,
                    'descriptionText': row.descriptionText,
                    'nftMetadata': nft_metadata
                })
            
            self.raw_data_cache[cache_key] = raw_data
            logger.info(f"Processed and cached {len(raw_data)} records")
        
        raw_data = self.raw_data_cache[cache_key]
        return self._process_raw_data(raw_data, threshold)
    
    def _generate_sample_data(self, limit, threshold):
        """Generate sample data"""
        logger.info(f"Generating sample data for {limit} nodes...")
        np.random.seed(42)
        
        embedding_dim = 384
        n_clusters = max(5, min(15, limit // 50))
        
        cluster_centers = np.random.randn(n_clusters, embedding_dim)
        for i in range(n_clusters):
            cluster_centers[i] = cluster_centers[i] / np.linalg.norm(cluster_centers[i])
        
        raw_data = []
        for i in range(limit):
            cluster_id = i % n_clusters
            
            noise_factor = 0.2 + 0.3 * np.random.random()
            noise = np.random.randn(embedding_dim) * noise_factor
            embedding = cluster_centers[cluster_id] + noise
            embedding = embedding / np.linalg.norm(embedding)
            
            asset_types = ['Art', 'Music', 'Game', 'Video', 'Text', 'Code', 'Design', 'Data']
            asset_type = asset_types[cluster_id % len(asset_types)]
            
            raw_data.append({
                'id': f'sample_{i:04d}',
                'embedding': embedding.tolist(),
                'descriptionText': f'Sample {asset_type.lower()} asset representing cluster {cluster_id} content.',
                'nftMetadata': {
                    'name': f'{asset_type} #{i}',
                    'imageUrl': f'https://via.placeholder.com/150/{["FF6B6B","4ECDC4","45B7D1","96CEB4","FFEAA7","DDA0DD","98D8C8","F7DC6F"][cluster_id%8]}/FFFFFF?text={asset_type[0]}{i}'
                }
            })
        
        return self._process_raw_data(raw_data, threshold)
    
    def _process_raw_data(self, raw_data, threshold):
        """Process raw data - EXACTLY like your Next.js"""
        start_time = time.time()
        
        # Create nodes exactly like your Next.js
        nodes = []
        for row in raw_data:
            nft_metadata = row['nftMetadata'] if row['nftMetadata'] else {}
            nodes.append({
                'id': row['id'],
                'label': nft_metadata.get('name', str(row['id'] or 'Unknown')),
                'description': row['descriptionText'] or 'No description',
                'imageUrl': nft_metadata.get('imageUrl', '')
            })
        
        # Calculate links EXACTLY like your Next.js
        links = []
        max_links = 7500
        
        logger.info(f"Calculating similarities for {len(raw_data)} nodes with threshold {threshold}...")
        
        for i in range(len(raw_data)):
            if len(links) >= max_links:
                break
            for j in range(i + 1, len(raw_data)):
                if len(links) >= max_links:
                    break
                
                # Use EXACT same similarity calculation as your JS
                similarity = cosine_similarity_custom(raw_data[i]['embedding'], raw_data[j]['embedding'])
                
                if similarity > threshold:
                    links.append({
                        'source': raw_data[i]['id'],
                        'target': raw_data[j]['id'],
                        'similarity': similarity
                    })
        
        processing_time = time.time() - start_time
        logger.info(f"Processed {len(nodes)} nodes and {len(links)} links in {processing_time:.2f}s")
        
        return {'nodes': nodes, 'links': links}

# Initialize
viz = SemanticVisualization()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Semantic Visualization"
server = app.server

def create_d3_visualization(nodes, links, selected_node=None):
    """Create visualization with EXACT D3 ring pattern"""
    if not nodes:
        logger.warning("No nodes provided to create_d3_visualization")
        return go.Figure()
    
    logger.info(f"Creating EXACT D3 force simulation for {len(nodes)} nodes, {len(links)} links...")
    
    # Debug: Check if we have too many links (indicates threshold too low)
    if len(links) > 5000:
        logger.warning(f"Very dense network with {len(links)} links - may need higher similarity threshold")
    
    # Run the EXACT D3 force simulation
    simulation = ExactD3ForceSimulation(nodes, links, width=1200, height=800)
    positions = simulation.run_simulation(iterations=300)
    
    # Debug: Check positions
    logger.info(f"Simulation complete. Sample positions: {positions[:3]}")
    
    # Create position mapping
    pos = {nodes[i]['id']: positions[i] for i in range(len(nodes))}
    
    # Create edge traces FIRST (behind nodes)
    edge_x = []
    edge_y = []
    
    edges_rendered = 0
    max_edges_to_render = 3000  # Limit edges for performance
    
    for link in links:
        if edges_rendered >= max_edges_to_render:
            break
            
        if link['source'] in pos and link['target'] in pos:
            x0, y0 = pos[link['source']]
            x1, y1 = pos[link['target']]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edges_rendered += 1
    
    logger.info(f"Rendering {edges_rendered} edges out of {len(links)} total")
    
    # Only create edge trace if we have edges
    edge_traces = []
    if edge_x:
        edge_trace = go.Scattergl(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#999999'),
            hoverinfo='none',
            mode='lines',
            opacity=0.6 if not selected_node else 0.2,
            name="Links",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node data
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    node_info = []
    
    # EXACT D3 colors - d3.schemeCategory10
    d3_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Build connection map for selection
    connections = {}
    for link in links:
        if link['source'] not in connections:
            connections[link['source']] = set()
        if link['target'] not in connections:
            connections[link['target']] = set()
        connections[link['source']].add(link['target'])
        connections[link['target']].add(link['source'])
    
    for i, node in enumerate(nodes):
        node_id = node['id']
        
        if node_id in pos:
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
        else:
            # Fallback position
            node_x.append(600)
            node_y.append(400)
        
        # EXACT D3 selection coloring and sizing
        if selected_node and node_id == selected_node:
            node_colors.append('red')    # Selected node
            node_sizes.append(12)        # Your D3: r = 12 for selected
        elif selected_node and node_id in connections.get(selected_node, set()):
            node_colors.append('orange') # Connected nodes
            node_sizes.append(10)        # Your D3: r = 10 for connected
        elif selected_node:
            node_colors.append('lightgray')  # Dimmed nodes
            node_sizes.append(6)
        else:
            # EXACT D3 coloring: colorScale(String(i % 10))
            node_colors.append(d3_colors[i % len(d3_colors)])
            node_sizes.append(8)         # Your D3: .attr("r", 8)
        
        # Labels for smaller graphs (like your D3)
        if len(nodes) <= 200:
            node_text.append(node['label'][:12])
        else:
            node_text.append('')
        
        node_info.append(f"<b>{node['label']}</b><br>ID: {node['id']}<br>{node['description'][:100]}...")
    
    logger.info(f"Rendering {len(node_x)} nodes. Sample positions: x={node_x[:3]}, y={node_y[:3]}")
    logger.info(f"Node colors: {node_colors[:3]}, sizes: {node_sizes[:3]}")
    
    # Create node trace
    node_trace = go.Scattergl(
        x=node_x, y=node_y,
        mode='markers+text' if any(node_text) else 'markers',
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color='white'),  # Your D3: white text with shadow
        hoverinfo='text',
        hovertext=node_info,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.5, color='white'),  # Your D3: stroke="#fff" stroke-width=1.5
            opacity=0.9
        ),
        customdata=[node['id'] for node in nodes],
        name="Nodes"
    )
    
    # Create figure with EXACT D3 dark styling
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=dict(
                text=f'Semantic Visualization ({len(nodes)} nodes, {len(links)} links)',
                x=0.5,
                font=dict(size=16, color='white')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#374151',  # Dark background like your D3
            paper_bgcolor='#4b5563',
            height=800,
            width=1200,
            font=dict(color='white')
        )
    )
    
    logger.info("Figure created successfully")
    return fig

# Layout
app.layout = html.Div([
    html.H1("Semantic View", style={'color': 'white', 'text-align': 'center', 'font-size': '2rem', 'margin-bottom': '10px'}),
    
    html.P("Explore IP assets based on their semantic similarity using vector embeddings. "
           "Assets with similar content are positioned closer together in this interactive visualization.",
           style={'color': '#9CA3AF', 'text-align': 'center', 'margin-bottom': '20px'}),
    
    html.Div([
        html.H3("How it works", style={'color': '#1E3A8A', 'margin-bottom': '10px'}),
        html.P("Each IP asset is represented by a high-dimensional vector embedding that captures its semantic meaning. "
               "We calculate cosine similarity between these vectors to determine how related different assets are, "
               "then use a force-directed graph to visualize these relationships spatially.",
               style={'color': '#1E40AF', 'font-size': '14px'})
    ], style={'background': '#DBEAFE', 'border': '1px solid #93C5FD', 'border-radius': '8px', 
              'padding': '16px', 'margin-bottom': '24px'}),
    
    html.Div([
        html.Label("Max Nodes:", style={'color': 'white', 'margin-right': '10px'}),
        dcc.Dropdown(
            id='max-nodes',
            options=[
                {'label': '100', 'value': 100},
                {'label': '250', 'value': 250},
                {'label': '500', 'value': 500},
                {'label': '1000', 'value': 1000},
                {'label': '2000', 'value': 2000}
            ],
            value=500,
            style={'width': '120px', 'display': 'inline-block', 'margin-right': '20px'}
        ),
        
        html.Label("Similarity:", style={'color': 'white', 'margin-right': '10px'}),
        dcc.Slider(
            id='threshold',
            min=0.85, max=0.99, step=0.005, value=0.92,
            marks={i/100: f'{i/100:.2f}' for i in range(85, 100, 2)},
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'text-align': 'center', 'margin': '20px'}),
    
    dcc.Loading(
        dcc.Graph(id='graph', style={'height': '800px'}),
        type="circle"
    ),
    
    html.Div(id='node-info'),
    
    dcc.Store(id='data'),
    dcc.Store(id='selected')
    
], style={'background': 'linear-gradient(135deg, #374151, #4b5563)', 'min-height': '100vh', 'padding': '20px'})

# Callbacks
@app.callback(
    Output('data', 'data'),
    [Input('max-nodes', 'value'), Input('threshold', 'value')]
)
def update_data(max_nodes, threshold):
    return viz.fetch_graph_data(limit=max_nodes, threshold=threshold)

@app.callback(
    [Output('graph', 'figure'), Output('selected', 'data')],
    [Input('data', 'data'), Input('graph', 'clickData')],
    [State('selected', 'data')]
)
def update_graph(data, click_data, current_selected):
    ctx = callback_context
    
    if not data or 'nodes' not in data:
        return go.Figure(), None
    
    selected = current_selected
    if ctx.triggered and 'clickData' in ctx.triggered[0]['prop_id']:
        if click_data and 'points' in click_data:
            point = click_data['points'][0]
            if 'customdata' in point:
                new_selected = point['customdata']
                selected = new_selected if new_selected != current_selected else None
    
    fig = create_d3_visualization(data['nodes'], data['links'], selected)
    return fig, selected

@app.callback(
    Output('node-info', 'children'),
    [Input('selected', 'data'), Input('data', 'data')]
)
def update_info(selected, data):
    if not selected or not data:
        return html.Div()
    
    node = next((n for n in data['nodes'] if n['id'] == selected), None)
    if not node:
        return html.Div()
    
    connections = len([l for l in data['links'] if l['source'] == selected or l['target'] == selected])
    
    return html.Div([
        html.H3(node['label'], style={'color': '#1f2937', 'margin-bottom': '15px'}),
        html.P(f"ID: {node['id']}", style={'font-family': 'monospace', 'font-size': '12px', 'color': '#374151'}),
        html.P(node['description'], style={'font-size': '14px', 'color': '#374151', 'margin': '10px 0'}),
        html.P(f"Connections: {connections}", style={'color': '#2563eb', 'font-weight': 'bold'})
    ], style={'background': 'white', 'padding': '20px', 'border-radius': '8px', 'margin-top': '20px'})

if __name__ == '__main__':
    viz.init_bigquery()
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)