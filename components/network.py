import streamlit as st
import streamlit.components.v1 as components
import json

def render_network_graph(nodes, edges):
    """
    Renders a kinetic network graph using Vis.js via HTML injection.
    """
    
    # Convert data to JSON for JS injection
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    
    # VIS.JS HTML Template
    # We use a Dark Neon theme consistent with the app
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            body {{ margin: 0; background: transparent; overflow: hidden; }}
            #mynetwork {{
                width: 100%;
                height: 600px;
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 16px;
                background: rgba(20, 20, 30, 0.4);
                backdrop-filter: blur(10px);
            }}
        </style>
    </head>
    <body>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        // Data from Streamlit
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        
        var container = document.getElementById('mynetwork');
        var data = {{
            nodes: nodes,
            edges: edges
        }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                size: 16,
                font: {{
                    size: 14,
                    color: '#e0e0e0',
                    face: 'Inter'
                }},
                borderWidth: 2,
                shadow: true,
                color: {{
                    background: '#0a0a12',
                    border: '#00f3ff',
                    highlight: {{
                        background: '#00f3ff',
                        border: '#bd00ff'
                    }}
                }}
            }},
            edges: {{
                width: 1,
                color: {{
                    color: 'rgba(255, 255, 255, 0.1)',
                    highlight: '#bd00ff',
                    hover: '#00f3ff'
                }},
                smooth: {{
                    type: 'continuous'
                }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{
                    gravitationalConstant: -2000, // Spreads nodes out
                    springConstant: 0.04,
                    springLength: 95
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                zoomView: true
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Add "breathing" animation simulation by tweaking physics occasionally
        // or just let the natural float happen.
        
        // Cursor interaction for "alive" feel
        // We can't easily pass mouse pos from Python continuously here without lag,
        // so we rely on Vis.js native hover events which are very smooth.
        
    </script>
    </body>
    </html>
    """
    
    # Inject the graph
    components.html(html_code, height=600, scrolling=False)
