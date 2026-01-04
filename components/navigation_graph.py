import streamlit as st
import streamlit.components.v1 as components
import json

def render_sidebar_graph(current_page):
    """
    Renders a small interactive graph in the sidebar for navigation.
    Updates the URL query param 'page' when a node is clicked.
    Highlights the 'current_page' node.
    """
    
    # Navigation structure
    # We define nodes and simple edges to make it look connected like a "constellation"
    nav_items = ["Upload & Scan", "Deep Analysis", "Explainability", "Graph Intelligence", "History", "Settings"]
    
    nodes = []
    edges = []
    
    # Create simple linear or star topology for nav
    # Let's do a meaningful structure: Upload -> Analysis -> Explainability -> Graph. Settings typically separate.
    
    # Define visual properties
    active_color = "#00f3ff"   # Cyan for active
    default_color = "#2d2d3d"  # Dark Grey for inactive
    text_active = "#ffffff"
    text_default = "#888888"
    
    # Layout positions (hardcoded specific x,y for a nice vertical or tree layout in the narrow sidebar)
    # We will let physics handle it briefly or separate them manually.
    # Linear vertical stack might be boring. Let's do a zig-zag or "constellation" vertical flow.
    
    for i, item in enumerate(nav_items):
        is_active = (item == current_page)
        
        node_color = active_color if is_active else default_color
        font_color = text_active if is_active else text_default
        border_width = 3 if is_active else 1
        size = 20 if is_active else 12
        
        nodes.append({
            "id": item,
            "label": item,
            "color": {
                "background": node_color,
                "border": active_color if is_active else "#444"
            },
            "font": {"color": font_color, "size": 14, "face": "Inter"},
            "size": size,
            "borderWidth": border_width,
            "shadow": True
        })
        
        # Connect sequentially for the "flow" metaphor
        if i > 0:
            edges.append({
                "from": nav_items[i-1],
                "to": item,
                "color": {"color": "#444", "opacity": 0.3}
            })

    # Convert to JSON
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            body {{ margin: 0; padding: 0; overflow: hidden; background: transparent; }}
            #nav-network {{
                width: 100%;
                height: 350px; /* Specific height for sidebar */
            }}
        </style>
    </head>
    <body>
    <div id="nav-network"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        
        var container = document.getElementById('nav-network');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                fixed: false, # Allow movement for fun
            }},
            edges: {{
                width: 1,
                smooth: {{ type: 'curvedCW', roundness: 0.2 }}
            }},
            physics: {{
                stabilization: false,
                barnesHut: {{
                    gravitationalConstant: -2000,
                    springConstant: 0.05,
                    springLength: 50
                }},
                minVelocity: 0.75
            }},
            interaction: {{
                hover: true,
                zoomView: false,
                dragView: false
            }}
        }};
        
        var network = new vis.Network(container, data, options);
        
        // Handle Clicks
        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                var selectedNodeId = params.nodes[0];
                
                // Encode for URL safely
                var encodedPage = encodeURIComponent(selectedNodeId);
                
                // Force reload/navigation by updating parent URL
                // Streamlit will catch this query param on reload
                window.parent.location.search = '?page=' + encodedPage;
            }}
        }});
        
        // Fit initial view
        network.once("stabilizationIterationsDone", function() {{
             network.fit();
        }});
        
    </script>
    </body>
    </html>
    """
    
    # Render with component
    components.html(html_code, height=350)
