import geopandas as gpd

# --- Load shapefile ---
gdf = gpd.read_file("fishnet.shp")

# Rename truncated DBF column names to readable ones
# DBF format caps column names at 10 characters, so the original field names are truncated
#   SHP name (truncated)  →  Clean name
gdf = gdf.rename(columns={
    'SG_200m_Fe': 'OBJECTID',  # SG_200m_FeatureID  →  OBJECTID   (Feature ID)
    'SG_200m__1': 'BCR',       # SG_200m_BCR        →  BCR        (Building Coverage Ratio)
    'SG_200m__2': 'FAR',       # SG_200m_FAR        →  FAR        (Floor Area Ratio)
    'SG_200m__3': 'AHI',       # SG_200m_AHI        →  AHI        (Average Height Index)
    'SG_200m__4': 'ABV',       # SG_200m_ABV        →  ABV        (Average Building Volume)
    'SG_200m__5': 'NB_BCR',    # SG_200m_NB_BCR     →  NB_BCR     (Neighbour BCR)
    'SG_200m__6': 'NB_FAR',    # SG_200m_NB_FAR     →  NB_FAR     (Neighbour FAR)
    'SG_200m__7': 'NB_AHI',    # SG_200m_NB_AHI     →  NB_AHI     (Neighbour AHI)
    'SG_200m__8': 'NB_ABV',    # SG_200m_NB_ABV     →  NB_ABV     (Neighbour ABV)
    'SG_200m__9': 'NDVI',      # SG_200m_NDVI       →  NDVI       (Normalised Difference Vegetation Index)
    'SG_200m_10': 'NDWI',      # SG_200m_NDWI       →  NDWI       (Normalised Difference Water Index)
    'SG_200m_11': 'DIS_MRT',   # SG_200m_DIS_MRT    →  DIS_MRT    (Distance to MRT)
    'SG_200m_12': 'DIS_CBD',   # SG_200m_DIS_CBD    →  DIS_CBD    (Distance to CBD)
    'SG_200m_13': 'LST',       # SG_200m_LST        →  LST        (Land Surface Temperature)
    'SG_200m_14': 'NTL',       # SG_200m_NTL        →  NTL        (Nighttime Light)
    # The following columns are already correctly named in the DBF:
    # 'POI'        →  POI        (Point of Interest count)
    # 'RND'        →  RND        (Road Network Density)
    # 'Shape_Leng' →  Shape_Leng (Polygon perimeter)
    # 'Shape_Area' →  Shape_Area (Polygon area)
    # 'PM25'       →  PM25       (PM2.5 air quality)
    # 'CO2'        →  CO2        (CO2 emissions)
    # 'POPU'       →  POPU       (Population)
})

# --- Filter rule ---
# Exclude rows where any of these key environmental fields are zero,
# as they indicate cells with no valid data (e.g. water bodies, offshore areas)
exclude = (gdf['LST'] == 0) | (gdf['PM25'] == 0) | (gdf['NTL'] == 0)
gdf_subset = gdf[~exclude].copy()

print(f"Original fishnet : {len(gdf):,} rows")
print(f"Excluded         : {exclude.sum():,} rows  (LST==0 OR PM25==0 OR NTL==0)")
print(f"Subset result    : {len(gdf_subset):,} rows")

# --- Save ---
gdf_subset.to_file("fishnet_subset.shp")
print("Saved → fishnet_subset.shp")
