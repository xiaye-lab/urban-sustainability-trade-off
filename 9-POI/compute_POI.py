"""
compute_poi_metrics.py
======================
Computes Walkable POI Density and POI Shannon Diversity (Entropy) per 200m 
fishnet cell using a 600m walkable buffer.

Data sources
------------
  OSM Points of Interest
    Input      : hotosm_sgp_points_of_interest_points_shp.shp
    Processing : Filters out infrastructure/utilities. Categorises remaining 
                 socioeconomic points into 5 macro-functions (F&B, Retail, 
                 Civic, Leisure, Tourism) to calculate functional diversity.

Metrics
-------
  POI_COUNT  = Total valid socioeconomic POIs within a 600m radius of cell centroid.
  POI_DIVERS = Shannon Entropy of the 5 macro-categories within the 600m radius.
               H = -sum(p * ln(p))

Outputs
-------
  poi_metrics.csv               Raw per-node metrics (OBJECTID key)
  fishnet_final_poi_sevi.shp    Input shapefile + POI_COUNT & POI_DIVERS columns
"""

import os
import math
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

# ── PROJ conflict fix ─────────────────────────────────────────────────────────
try:
    import pyproj as _pp
    _pd = os.path.join(os.path.dirname(_pp.__file__), "proj_dir", "share", "proj")
    if os.path.isdir(_pd):
        os.environ["PROJ_DATA"] = _pd
        os.environ["PROJ_LIB"]  = _pd
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ────────────────────────────────────────────────────────────
GRID_SHP      = "fishnet_final_price.shp"
POI_SHP       = "hotosm_sgp_points_of_interest_points_shp.shp"
OUT_SHP       = "fishnet_final_POI.shp"  # The final target for GNN
BUFFER_RADIUS = 600.0  # meters

# SVY21 WKT — bypasses proj.db conflicts
SVY21_WKT = (
    'PROJCS["SVY21 / Singapore TM",'
    'GEOGCS["SVY21",DATUM["SVY21",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",1.36666666666667],'
    'PARAMETER["central_meridian",103.833333333333],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",28001.642],'
    'PARAMETER["false_northing",38744.572],'
    'UNIT["metre",1]]'
)


# ── Step 1: Load and project data ────────────────────────────────────────────
def load_data():
    print(f"  Loading Grid : {GRID_SHP}")
    grid = gpd.read_file(GRID_SHP)
    
    print(f"  Loading POIs : {POI_SHP}")
    pois = gpd.read_file(POI_SHP)
    
    # Force SVY21 projection for accurate 600m buffering
    print("  Projecting to SVY21 (meters)...")
    if grid.crs != SVY21_WKT:
        grid = grid.to_crs(SVY21_WKT)
    if pois.crs != SVY21_WKT:
        pois = pois.to_crs(SVY21_WKT)
        
    return grid, pois


# ── Step 2: Clean and Categorize POIs ────────────────────────────────────────
def clean_and_categorize_pois(pois):
    initial_count = len(pois)
    
    # 1. Define infrastructural noise to exclude
    exclusions = [
        'waste_basket', 'bench', 'post_box', 'recycling', 'vending_machine', 
        'telephone', 'parking', 'parking_entrance', 'parking_space', 'bicycle_parking', 
        'toilets', 'fire_hydrant', 'fuel', 'car_wash', 'police', 'fire_station',
        'substation', 'water_tower', 'drinking_water', 'shower'
    ]
    
    # Filter out exclusions
    if 'amenity' in pois.columns:
        pois = pois[~pois['amenity'].isin(exclusions)]
    
    def assign_category(row):
        amenity = str(row.get('amenity', '')).lower()
        shop    = str(row.get('shop', '')).lower()
        tourism = str(row.get('tourism', '')).lower()
        
        # 1. Food & Beverage (F&B)
        if amenity in ['restaurant', 'cafe', 'fast_food', 'bar', 'pub', 'food_court', 'ice_cream']:
            return 'F&B'
            
        # 2. Retail & Commercial
        elif shop != 'nan' and shop != 'none' and shop != '':
            return 'Retail'
        elif amenity in ['bank', 'marketplace', 'pharmacy', 'atm']:
            return 'Retail'
            
        # 3. Institutional & Civic
        elif amenity in ['school', 'university', 'college', 'kindergarten', 'library', 'hospital', 
                         'clinic', 'dentist', 'post_office', 'place_of_worship', 'community_centre', 'social_facility']:
            return 'Civic'
            
        # 4. Leisure & Entertainment
        elif amenity in ['cinema', 'theatre', 'nightclub', 'arts_centre', 'sports_centre', 'pitch', 'swimming_pool']:
            return 'Leisure'
            
        # 5. Tourism & Hospitality
        elif tourism != 'nan' and tourism != 'none' and tourism != '':
            return 'Tourism'
            
        else:
            return 'Drop'

    print("  Categorizing into 5 socioeconomic macro-functions...")
    pois['macro_category'] = pois.apply(assign_category, axis=1)
    
    # Drop uncategorized or infrastructural noise
    pois_clean = pois[pois['macro_category'] != 'Drop'].copy()
    
    print(f"  Original OSM POIs        : {initial_count:,}")
    print(f"  Valid Socioeconomic POIs : {len(pois_clean):,}")
    print("\n  Category Breakdown:")
    print(pois_clean['macro_category'].value_counts().to_string())
    
    return pois_clean


# ── Step 3: Spatial Join and Entropy Calculation ─────────────────────────────
def compute_metrics(grid, pois_clean):
    print(f"\n  Creating {BUFFER_RADIUS}m walkable catchments around grid centroids...")
    
    # Create 600m buffers around cell centroids
    grid_buffers = grid[['OBJECTID', 'geometry']].copy()
    grid_buffers['geometry'] = grid_buffers.geometry.centroid.buffer(BUFFER_RADIUS)
    
    print("  Spatial join: POIs within catchments...")
    joined = gpd.sjoin(pois_clean, grid_buffers, how="inner", predicate="intersects")
    
    print("  Calculating Shannon Diversity (Entropy)...")
    results = []
    
    # Group by the grid's OBJECTID
    for obj_id, group in joined.groupby('OBJECTID'):
        total_pois = len(group)
        
        # Calculate Shannon Entropy
        category_counts = group['macro_category'].value_counts()
        entropy = 0.0
        for count in category_counts:
            p = count / total_pois
            entropy -= p * math.log(p)
            
        results.append({
            'OBJECTID': obj_id,
            'POI_COUNT': total_pois,
            'POI_DIVERS': entropy
        })
        
    metrics_df = pd.DataFrame(results)
    return metrics_df


# ── Step 4: Merge and Save ───────────────────────────────────────────────────
def save_results(grid, metrics_df):
    print("\n[4/4] Saving Results...")
    
    # Ensure columns exist and fill with 0
    grid = grid.merge(metrics_df, on='OBJECTID', how='left')
    grid['POI_COUNT']  = grid['POI_COUNT'].fillna(0).astype(int)
    grid['POI_DIVERS'] = grid['POI_DIVERS'].fillna(0.0).round(4)
    
    s_valid_density = grid.loc[grid['POI_COUNT'] > 0, 'POI_COUNT']
    s_valid_divers  = grid.loc[grid['POI_DIVERS'] > 0, 'POI_DIVERS']

    print(f"\n{'='*60}")
    print(f"  POI METRICS SUMMARY (600m Walkable Catchment)")
    print(f"{'='*60}")
    print(f"  Cells with >0 POIs : {len(s_valid_density):,}")
    print(f"  Cells with 0 POIs  : {(grid['POI_COUNT']==0).sum():,}")
    print(f"")
    print(f"  POI DENSITY (Count per 600m)")
    print(f"    Median : {s_valid_density.median():.0f}")
    print(f"    Max    : {s_valid_density.max():.0f}")
    print(f"")
    print(f"  POI DIVERSITY (Shannon Entropy)")
    print(f"    Median : {s_valid_divers.median():.3f}")
    print(f"    Max    : {s_valid_divers.max():.3f} (Max theoretical for 5 bins ~ 1.609)")
    print(f"{'='*60}")

    # Save CSV
    grid[['OBJECTID', 'POI_COUNT', 'POI_DIVERS']].to_csv("poi_metrics.csv", index=False)
    print("\nSaved → poi_metrics.csv")
    
    # Save Shapefile
    grid.to_file(OUT_SHP)
    print(f"Saved → {OUT_SHP}")
    print(f"Columns ready for SEVI composite: ['POI_COUNT', 'POI_DIVERS']")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  WALKABLE POI DENSITY & FUNCTIONAL DIVERSITY (SHANNON)")
    print("=" * 60)
    
    print("\n[1/4] Loading Data...")
    grid, pois = load_data()
    
    print("\n[2/4] Cleaning & Categorizing OSM POIs...")
    pois_clean = clean_and_categorize_pois(pois)
    
    print("\n[3/4] Computing Spatial Metrics...")
    metrics_df = compute_metrics(grid, pois_clean)
    
    save_results(grid, metrics_df)