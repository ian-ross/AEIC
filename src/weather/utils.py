from pyproj import Geod
import numpy as np

def get_flight_track(mission):
    
    # Instantiate WGS84 ellipsoid
    geod = Geod(ellps ="WGS84")

    # Extract OD lat-lon 
    lon_dep, lat_dep, _ = mission["dep_location"]
    lon_arr, lat_arr, _ = mission["arr_location"]
    
    # Assume 100 points for discretization (for demo only)
    # Note: This will change when flying actual missions
    
    points = geod.npts(lon_dep, lat_dep, lon_arr, lat_arr, 100)
    
    lons = [lon_dep] + [pt[0] for pt in points] + [lon_arr]
    lats = [lat_dep] + [pt[1] for pt in points] + [lat_arr]
    
    print(lons)
    print(lats)
    
    

