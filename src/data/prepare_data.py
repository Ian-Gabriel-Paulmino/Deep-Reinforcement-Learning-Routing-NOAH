"""
Hazard-Aware Routing: Data Preprocessing Script

Prepares the road network data for the DRL routing framework.
Downloads the La Trinidad road network and overlays NOAH hazard data.

Usage:
    python preprocess_data.py

Output:
    data/la_trinidad_hazard_graph.gpickle

"""

import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """All configuration"""
    
    # Study area
    place_name: str = "La Trinidad, Benguet, Philippines"

    # Options: drive, walk, bike, all
    network_type: str = "drive"
    
    # Paths to NOAH hazard shapefiles
    flood_shapefile: str = "data/raw/noah/flood_hazard/Benguet_Flood_25year.shp"
    landslide_shapefile: str = "data/raw/noah/landslide_susceptibility/Benguet_LandslideHazards.shp"
    
    # Column names in the shapefiles that contain hazard classification
    flood_column: str = "Var"      # Values: 1, 2, 3 (low, moderate, high)
    landslide_column: str = "LH"   # Values: 1, 2, 3 (low, moderate, high)
    
    # Hazard score mapping (raw value -> normalized score)
    flood_scores: dict = None
    landslide_scores: dict = None
    
    # Sampling: how often to check hazard along each road (in meters)
    sample_interval_meters: float = 10.0
    
    # Combined hazard weights (should sum to 1.0)
    flood_weight: float = 0.6
    landslide_weight: float = 0.4
    
    # Output path
    output_path: str = "data/la_trinidad_hazard_graph.gpickle"
    
    def __post_init__(self):
        # Default score mappings if not provided
        if self.flood_scores is None:
            self.flood_scores = {
                1: 0.2,   # Low flood hazard
                2: 0.6,   # Moderate flood hazard
                3: 1.0,   # High flood hazard
            }
        if self.landslide_scores is None:
            self.landslide_scores = {
                1: 0.3,   # Low landslide susceptibility
                2: 0.6,   # Moderate landslide susceptibility
                3: 1.0,   # High landslide susceptibility
            }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text: str) -> None:
    """Print a visible section header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_step(step_num: int, text: str) -> None:
    """Print a step indicator."""
    print(f"\n[Step {step_num}] {text}")
    print("-" * 40)


def print_result(label: str, value) -> None:
    """Print a result in a consistent format."""
    print(f"  {label}: {value}")


# =============================================================================
# STEP 1: DOWNLOAD ROAD NETWORK
# =============================================================================

def download_road_network(config: Config) -> nx.MultiDiGraph:
    """
    Download the road network for La Trinidad from OpenStreetMap.
    
    Uses OSMnx to query the Overpass API. The result is a NetworkX
    graph where nodes are intersections and edges are road segments.
    """
    print_step(1, "Downloading road network from OpenStreetMap")
    
    print(f"  Querying: {config.place_name}")
    print(f"  Network type: {config.network_type}")
    print("  This may take a few seconds...")
    
    # Download the network
    G = ox.graph_from_place(
        config.place_name,
        network_type=config.network_type,
        simplify=True
    )
    
    # Add basic attributes we'll need later
    for u, v, key, data in G.edges(keys=True, data=True):
        # Ensure every edge has a geometry
        if 'geometry' not in data or data['geometry'] is None:
            u_coords = (G.nodes[u]['x'], G.nodes[u]['y'])
            v_coords = (G.nodes[v]['x'], G.nodes[v]['y'])
            data['geometry'] = LineString([u_coords, v_coords])
        
        # Add travel time based on length and assumed speed
        length_m = data.get('length', 0)
        speed_kph = 30  # Conservative default for rural roads
        data['travel_time_min'] = (length_m / 1000) / speed_kph * 60
    
    print_result("Nodes (intersections)", G.number_of_nodes())
    print_result("Edges (road segments)", G.number_of_edges())
    
    # Calculate total road length
    total_length_km = sum(d.get('length', 0) for _, _, d in G.edges(data=True)) / 1000
    print_result("Total road length", f"{total_length_km:.1f} km")
    
    return G


# =============================================================================
# STEP 2: LOAD HAZARD DATA
# =============================================================================

@dataclass
class HazardData:
    """Container for loaded hazard data with spatial index."""
    name: str
    polygons: gpd.GeoDataFrame
    spatial_index: STRtree

    # Same order as spatial_index for lookups
    geometries: list
    column: str
    scores: dict


def load_hazard_shapefile(
    filepath: str,
    name: str,
    column: str,
    scores: dict
) -> Optional[HazardData]:
    """
    Load a NOAH hazard shapefile and build a spatial index for fast queries.
    
    The spatial index (R-tree) allows us to quickly find which hazard polygon
    contains a given point, without checking every polygon individually.
    """
    path = Path(filepath)
    
    if not path.exists():
        print(f"  WARNING: {name} shapefile not found at {filepath}")
        print(f"           Skipping {name} hazard data.")
        return None
    
    print(f"  Loading {name} data from: {filepath}")
    
    # Load the shapefile
    gdf = gpd.read_file(filepath)
    print(f"    Loaded {len(gdf)} polygons")
    
    # Check CRS (coordinate reference system)
    if gdf.crs is None:
        print(f"    WARNING: No CRS defined, assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs) != "EPSG:4326":
        print(f"    Reprojecting from {gdf.crs} to EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    
    # Check for the classification column
    if column not in gdf.columns:
        print(f"    ERROR: Column '{column}' not found!")
        print(f"    Available columns: {list(gdf.columns)}")
        return None
    
    # Show distribution of hazard levels
    print(f"    Hazard distribution ({column}):")
    for value, count in gdf[column].value_counts().items():
        score = scores.get(int(value), "?")
        print(f"      Level {int(value)} (score={score}): {count} polygons")
    
    # Build spatial index for fast point-in-polygon queries
    # We store geometries in a list to map back from index results
    geometries = list(gdf.geometry)
    spatial_index = STRtree(geometries)
    print(f"    Built spatial index for {len(geometries)} geometries")
    
    return HazardData(
        name=name,
        polygons=gdf,
        spatial_index=spatial_index,
        geometries=geometries,
        column=column,
        scores=scores
    )


def load_all_hazard_data(config: Config) -> tuple:
    """Load both flood and landslide hazard data."""
    print_step(2, "Loading NOAH hazard data")
    
    flood_data = load_hazard_shapefile(
        config.flood_shapefile,
        "flood",
        config.flood_column,
        config.flood_scores
    )
    
    landslide_data = load_hazard_shapefile(
        config.landslide_shapefile,
        "landslide",
        config.landslide_column,
        config.landslide_scores
    )
    
    return flood_data, landslide_data


# =============================================================================
# STEP 3: SPATIAL OVERLAY (THE CORE ALGORITHM)
# =============================================================================

def query_hazard_at_point(
    point: Point,
    hazard_data: Optional[HazardData]
) -> float:
    """
    Query the hazard score at a specific point.
    
    Core spatial query. 
    Given a point on a road, check if it
    falls inside any hazard polygon. If it's in multiple polygons (overlapping
    hazard zones), return the maximum (worst-case) score.
    
    How it works:
    1. Use the spatial index to find candidate polygons (fast bounding box check)
    2. For each candidate, test if the point is actually inside (exact check)
    3. If inside, look up the hazard classification and convert to score
    4. Return the maximum score across all containing polygons
    
    Note: Shapely 2.0+ STRtree.query() returns indices, not geometries.
    We use these indices to look up the actual geometry objects.
    """
    if hazard_data is None:
        return 0.0
    
    # Query spatial index for nearby polygons
    # In Shapely 2.0+, this returns INDICES into the geometries list, not geometries
    candidate_indices = hazard_data.spatial_index.query(point)
    
    max_score = 0.0
    
    for idx in candidate_indices:
        # Look up the actual geometry using the index
        candidate_geom = hazard_data.geometries[idx]
        
        # Check if point is actually inside this polygon (not just bounding box)
        if candidate_geom.contains(point):
            # Get the hazard classification value from the GeoDataFrame
            try:
                raw_value = hazard_data.polygons.iloc[idx][hazard_data.column]
                score = hazard_data.scores.get(int(raw_value), 0.0)
                max_score = max(max_score, score)
            except (ValueError, KeyError):
                continue
    
    return max_score


def sample_points_along_edge(geometry: LineString, interval_m: float) -> list:
    """
    Generate sample points along a road edge at regular intervals.
    """
    # Estimate length in meters (approximate for geographic coordinates)
    coords = list(geometry.coords)
    
    # Calculate approximate length using simple distance formula
    total_length = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        # Rough conversion: 1 degree ≈ 111km at equator, adjust for latitude
        dlat = (lat2 - lat1) * 111000
        dlon = (lon2 - lon1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
        total_length += np.sqrt(dlat**2 + dlon**2)
    
    # Determine number of samples
    num_samples = max(3, int(total_length / interval_m) + 1)
    
    # Generate evenly spaced points using Shapely's interpolate
    points = []
    for i in range(num_samples):
        fraction = i / (num_samples - 1) if num_samples > 1 else 0.5
        point = geometry.interpolate(fraction, normalized=True)
        points.append(point)
    
    return points


def enrich_graph_with_hazards(
    G: nx.MultiDiGraph,
    flood_data: Optional[HazardData],
    landslide_data: Optional[HazardData],
    config: Config
) -> nx.MultiDiGraph:
    """
    Add hazard attributes to every edge in the road network.
    
    This is the spatial overlay process. For each road segment:
    1. Sample points along the road at regular intervals
    2. Query flood and landslide hazard at each point
    3. Take the maximum hazard across all points (worst-case approach)
    4. Store the results as edge attributes
    
    After this function, every edge will have:
    - flood_hazard: 0.0 to 1.0
    - landslide_hazard: 0.0 to 1.0  
    - combined_hazard: weighted combination of both
    """
    print_step(3, "Overlaying hazard data on road network")
    
    print(f"  Sampling interval: {config.sample_interval_meters} meters")
    print(f"  Processing {G.number_of_edges()} edges...")
    
    # Track statistics
    edges_with_flood = 0
    edges_with_landslide = 0
    edges_with_any_hazard = 0
    total_samples = 0
    
    # Process each edge with a progress bar
    edges = list(G.edges(keys=True, data=True))
    
    for u, v, key, data in tqdm(edges, desc="  Enriching edges"):
        geometry = data.get('geometry')
        
        if geometry is None:
            # This shouldn't happen, but handle it gracefully
            data['flood_hazard'] = 0.0
            data['landslide_hazard'] = 0.0
            data['combined_hazard'] = 0.0
            continue
        
        # Sample points along this edge
        sample_points = sample_points_along_edge(
            geometry, 
            config.sample_interval_meters
        )
        total_samples += len(sample_points)
        
        # Query hazard at each sample point
        flood_scores = [query_hazard_at_point(p, flood_data) for p in sample_points]
        landslide_scores = [query_hazard_at_point(p, landslide_data) for p in sample_points]
        
        # Aggregate: take maximum (worst-case) hazard
        flood_max = max(flood_scores) if flood_scores else 0.0
        landslide_max = max(landslide_scores) if landslide_scores else 0.0
        
        # Compute combined hazard
        combined = (config.flood_weight * flood_max + 
                   config.landslide_weight * landslide_max)
        
        # Store on edge
        data['flood_hazard'] = flood_max
        data['landslide_hazard'] = landslide_max
        data['combined_hazard'] = combined
        
        # Update statistics
        if flood_max > 0:
            edges_with_flood += 1
        if landslide_max > 0:
            edges_with_landslide += 1
        if combined > 0:
            edges_with_any_hazard += 1
    
    # Print summary
    total_edges = G.number_of_edges()
    print(f"\n  Overlay complete!")
    print_result("Total sample points queried", f"{total_samples:,}")
    print_result("Edges with flood hazard", 
                f"{edges_with_flood} ({100*edges_with_flood/total_edges:.1f}%)")
    print_result("Edges with landslide hazard", 
                f"{edges_with_landslide} ({100*edges_with_landslide/total_edges:.1f}%)")
    print_result("Edges with any hazard", 
                f"{edges_with_any_hazard} ({100*edges_with_any_hazard/total_edges:.1f}%)")
    
    return G


# =============================================================================
# STEP 4: SAVE RESULTS
# =============================================================================

def save_graph(G: nx.MultiDiGraph, filepath: str) -> None:
    """Save the enriched graph to a pickle file."""
    print_step(4, "Saving enriched graph")
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    
    size_mb = path.stat().st_size / (1024 * 1024)
    print_result("Output file", filepath)
    print_result("File size", f"{size_mb:.2f} MB")


# =============================================================================
# STEP 5: VERIFY RESULTS
# =============================================================================

def verify_results(G: nx.MultiDiGraph) -> None:
    """Print a summary to verify the preprocessing worked correctly."""
    print_step(5, "Verification summary")
    
    # Check that hazard attributes exist
    sample_edge = list(G.edges(data=True))[0]
    edge_data = sample_edge[2]
    
    required_attrs = ['flood_hazard', 'landslide_hazard', 'combined_hazard']
    missing = [attr for attr in required_attrs if attr not in edge_data]
    
    if missing:
        print(f"  ERROR: Missing attributes on edges: {missing}")
        return
    
    print("  ✓ All required hazard attributes present")
    
    # Compute hazard statistics
    flood_scores = [d['flood_hazard'] for _, _, d in G.edges(data=True)]
    landslide_scores = [d['landslide_hazard'] for _, _, d in G.edges(data=True)]
    combined_scores = [d['combined_hazard'] for _, _, d in G.edges(data=True)]
    
    print("\n  Hazard score statistics:")
    print(f"    Flood     - Mean: {np.mean(flood_scores):.4f}, "
          f"Max: {max(flood_scores):.2f}, "
          f"Non-zero: {sum(1 for s in flood_scores if s > 0)}")
    print(f"    Landslide - Mean: {np.mean(landslide_scores):.4f}, "
          f"Max: {max(landslide_scores):.2f}, "
          f"Non-zero: {sum(1 for s in landslide_scores if s > 0)}")
    print(f"    Combined  - Mean: {np.mean(combined_scores):.4f}, "
          f"Max: {max(combined_scores):.2f}, "
          f"Non-zero: {sum(1 for s in combined_scores if s > 0)}")
    
    # Warning if no hazards found
    if sum(combined_scores) == 0:
        print("\n  ⚠ WARNING: No edges have hazard exposure!")
        print("    This could mean:")
        print("    1. The hazard polygons don't overlap with La Trinidad")
        print("    2. The coordinate systems aren't aligned")
        print("    3. The shapefile paths are incorrect")
        print("    Please verify with the visualization script.")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run the complete preprocessing pipeline."""
    
    print_header("HAZARD-AWARE ROUTING: DATA PREPROCESSING")
    
    # Initialize configuration
    config = Config()
    
    print(f"\nStudy area: {config.place_name}")
    print(f"Flood data: {config.flood_shapefile}")
    print(f"Landslide data: {config.landslide_shapefile}")
    print(f"Output: {config.output_path}")
    
    # Step 1: Download road network
    G = download_road_network(config)
    
    # Step 2: Load hazard data
    flood_data, landslide_data = load_all_hazard_data(config)
    
    # Step 3: Overlay hazards on road network
    G = enrich_graph_with_hazards(G, flood_data, landslide_data, config)
    
    # Step 4: Save results
    save_graph(G, config.output_path)
    
    # Step 5: Verify
    verify_results(G)
    
    print_header("PREPROCESSING COMPLETE")
    print(f"\nYour enriched graph is ready at: {config.output_path}")
    print("You can now proceed to the training phase.")
    print("\nTo visualize the results, run:")
    print(f"  python visualize_hazards.py {config.output_path}")


if __name__ == "__main__":
    main()