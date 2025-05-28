from pyproj import Geod
import numpy as np

def compute_headings(lons, lats):
    """Compute aircraft heading between each point."""
    geod = Geod(ellps="WGS84")
    headings = []
    for i in range(len(lons) - 1):
        azimuth, _, _ = geod.inv(lons[i], lats[i], lons[i + 1], lats[i + 1])
        headings.append(np.deg2rad(azimuth))
    headings.append(headings[-1])  # Repeat last for equal length
    return np.array(headings)


