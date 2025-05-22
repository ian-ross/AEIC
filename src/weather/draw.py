import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.geodesic import Geodesic

def plot_flight_arc(mission):
    
    # Set up map
    
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-130, -65, 22, 50], crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
    
    # Plot flights
    gd = Geodesic()
    
    lon_dep, lat_dep, _ = mission["dep_location"]
    lon_arr, lat_arr, _ = mission["arr_location"]
    
    # Draw great circle arc
    
    arc = gd.inverse([lon_dep, lon_arr], [lat_dep, lat_arr])
    arc_coords = arc["coordinates"]
    ax.plot(arc_coords[:, 0], arc_coords[:, 1], 'k-', linewidth=0.7, alpha=0.6)
    
    # Plot dep/arr points
    ax.plot(lon_dep, lat_dep, 'go', markersize=3)
    ax.plot(lon_arr, lat_arr, 'ro', markersize=3)
    
    plt.tight_layout()
    plt.savefig("sample.png", dpi = 300)
    plt.close()