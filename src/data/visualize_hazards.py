"""
Hazard Visualization Script

Creates an interactive HTML map showing the road network colored by hazard level.
Use this to verify that the hazard overlay worked correctly.

Usage:
    python visualize_hazards.py data/la_trinidad_hazard_graph.gpickle

Output:
    Opens hazard_map.html in your browser

"""

import sys
import pickle
import webbrowser
from pathlib import Path

import folium
import numpy as np


def get_color(score: float) -> str:
    """Convert a hazard score (0-1) to a color."""
    if score == 0:
        return '#2E7D32'  # Green - safe
    elif score < 0.3:
        return '#8BC34A'  # Light green - very low
    elif score < 0.5:
        return '#FFEB3B'  # Yellow - low
    elif score < 0.7:
        return '#FF9800'  # Orange - moderate
    elif score < 0.9:
        return '#F44336'  # Red - high
    else:
        return '#B71C1C'  # Dark red - very high


def create_hazard_map(graph_path: str, output_path: str = "hazard_map.html"):
    """
    Create an interactive map showing hazard levels on the road network.
    
    Each road segment is colored based on its combined hazard score:
    - Green: No hazard (score = 0)
    - Yellow: Low hazard (score < 0.5)
    - Orange: Moderate hazard (score 0.5-0.7)
    - Red: High hazard (score > 0.7)
    
    Click on any road to see detailed hazard information.
    """
    
    print(f"Loading graph from: {graph_path}")
    
    # Load the graph
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate map center
    lats = [G.nodes[n]['y'] for n in G.nodes()]
    lons = [G.nodes[n]['x'] for n in G.nodes()]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='cartodbpositron'
    )
    
    # Add each edge to the map
    print("Adding edges to map...")
    
    for u, v, data in G.edges(data=True):
        # Get hazard scores
        flood = data.get('flood_hazard', 0)
        landslide = data.get('landslide_hazard', 0)
        combined = data.get('combined_hazard', 0)
        
        # Get geometry
        geometry = data.get('geometry')
        if geometry is None:
            coords = [
                [G.nodes[u]['y'], G.nodes[u]['x']],
                [G.nodes[v]['y'], G.nodes[v]['x']]
            ]
        else:
            coords = [[lat, lon] for lon, lat in geometry.coords]
        
        # Create popup with hazard info
        popup_text = f"""
        <b>Road Segment</b><br>
        Length: {data.get('length', 0):.0f} m<br>
        Type: {data.get('highway', 'unknown')}<br>
        <hr>
        <b>Hazard Scores:</b><br>
        Flood: {flood:.2f}<br>
        Landslide: {landslide:.2f}<br>
        Combined: {combined:.2f}
        """
        
        # Add the line
        folium.PolyLine(
            locations=coords,
            color=get_color(combined),
            weight=3 if combined > 0 else 2,
            opacity=0.8,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; 
                background: white; padding: 10px; border: 2px solid gray;
                border-radius: 5px; font-size: 12px; z-index: 9999;">
        <b>Hazard Level</b><br>
        <span style="color: #2E7D32;">■</span> Safe (0)<br>
        <span style="color: #8BC34A;">■</span> Very Low (&lt;0.3)<br>
        <span style="color: #FFEB3B;">■</span> Low (0.3-0.5)<br>
        <span style="color: #FF9800;">■</span> Moderate (0.5-0.7)<br>
        <span style="color: #F44336;">■</span> High (0.7-0.9)<br>
        <span style="color: #B71C1C;">■</span> Very High (&gt;0.9)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save and open
    m.save(output_path)
    print(f"\nMap saved to: {output_path}")
    
    # Try to open in browser
    try:
        webbrowser.open(f"file://{Path(output_path).absolute()}")
        print("Opening map in browser...")
    except Exception:
        print(f"Open {output_path} in your browser to view the map.")
    
    # Print summary statistics
    combined_scores = [d.get('combined_hazard', 0) for _, _, d in G.edges(data=True)]
    
    print("\n" + "=" * 40)
    print("HAZARD SUMMARY")
    print("=" * 40)
    print(f"Total edges: {len(combined_scores)}")
    print(f"Edges with hazard: {sum(1 for s in combined_scores if s > 0)}")
    print(f"Safe edges: {sum(1 for s in combined_scores if s == 0)}")
    print(f"High hazard edges (>0.7): {sum(1 for s in combined_scores if s > 0.7)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_hazards.py <path_to_graph.gpickle>")
        print("Example: python visualize_hazards.py data/la_trinidad_hazard_graph.gpickle")
        sys.exit(1)
    
    graph_path = sys.argv[1]
    
    if not Path(graph_path).exists():
        print(f"Error: File not found: {graph_path}")
        sys.exit(1)
    
    create_hazard_map(graph_path)


if __name__ == "__main__":
    main()