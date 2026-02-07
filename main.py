import geopandas as gpd
import pandas as pd

def analyze_hazard_data(shapefile_path, hazard_type):
    print("=" * 60)
    print(f"{hazard_type.upper()} HAZARD DATA SUMMARY")
    print("=" * 60)

    gdf = gpd.read_file(shapefile_path)

    print(f"Source file: {shapefile_path}")
    print(f"CRS: {gdf.crs}")
    print(f"Total features: {len(gdf)}")
    print(f"Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")

    # Reproject to metric CRS for area computation (UTM Zone 51N – Philippines)
    gdf_m = gdf.to_crs(epsg=32651)
    gdf_m["area_km2"] = gdf_m.geometry.area / 1e6

    print(f"\nArea statistics (km²):")
    print(f"  Total area: {gdf_m['area_km2'].sum():.2f}")
    print(f"  Mean area:  {gdf_m['area_km2'].mean():.2f}")
    print(f"  Max area:   {gdf_m['area_km2'].max():.2f}")

    # Flood-specific stats
    if hazard_type == "flood":
        possible_cols = ["FLOOD_LEVEL", "Var", "DEPTH", "LEVEL", "CLASS"]
        col = next((c for c in possible_cols if c in gdf.columns), None)

        if col:
            print(f"\nFlood classification column: {col}")
            print("Flood level distribution:")
            print(gdf[col].value_counts())
        else:
            print("\nNo flood classification column found")

    # Landslide-specific stats
    if hazard_type == "landslide":
        possible_cols = ["SUSCEPT", "LH", "RATING", "CLASS"]
        col = next((c for c in possible_cols if c in gdf.columns), None)

        if col:
            print(f"\nLandslide susceptibility column: {col}")
            print("Susceptibility distribution:")
            print(gdf[col].value_counts())
        else:
            print("\nNo landslide susceptibility column found")

    # Spatial extent
    bounds = gdf.total_bounds
    print("\nSpatial extent (lon/lat):")
    print(f"  Min Lon: {bounds[0]:.4f}, Min Lat: {bounds[1]:.4f}")
    print(f"  Max Lon: {bounds[2]:.4f}, Max Lat: {bounds[3]:.4f}")

    print("\n")


if __name__ == "__main__":
    analyze_hazard_data(
        "./benguet-flood/Benguet_Flood_25year.shp",
        hazard_type="flood"
    )

    analyze_hazard_data(
        "./benguet-landslide/Benguet_LandslideHazards.shp",
        hazard_type="landslide"
    )
