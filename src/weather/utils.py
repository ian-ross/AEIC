from pyproj import Geod
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

def altitude_to_pressure_hpa(alt_ft):
    """Convert altitude in feet to pressure in hPa using ISA approximation."""
    alt_m = np.array(alt_ft) * 0.3048
    pressure = 1013.25 * (1 - (0.0065 * alt_m) / 288.15) ** 5.2561
    return pressure
    

def get_wind_at_points(mission_data, era5_path):
    """
    Interpolates U and V wind components from ERA5 at each lat/lon/alt point.

    Parameters
    ----------
    mission_data : dict
        Output from get_mission_points(); must contain 'lons', 'lats', and 'H'.
    era5_path : str
        Path to ERA5 GRIB file.

    Returns
    -------
    u_vals : np.ndarray
        Zonal wind (eastward) component at each point [m/s]
    v_vals : np.ndarray
        Meridional wind (northward) component at each point [m/s]
    wind_speed : np.ndarray
        Wind speed magnitude [m/s]
    """
    
    # Load ERA4 GRIB file (single time slice)
    ds = xr.open_dataset(era5_path, engine="cfgrib")
    
    # Ensure proper dimension order
    lats_era = ds.latitude.values
    lons_era = ds.longitude.values
    levels_era = ds.isobaricInhPa.values
    
    # Sort lattitude in ascending order
    if lats_era[0] > lats_era[-1]:
        lats_era = lats_era[::-1]
        ds['u'] = ds['u'][::-1, :, :]
        ds['v'] = ds['v'][::-1, :, :]

    # Prepare 3D interpolators
    u_interp = RegularGridInterpolator(
        (levels_era, lats_era, lons_era),
        ds['u'].values,
        bounds_error=False,
        fill_value=np.nan
    )
    
    v_interp = RegularGridInterpolator(
        (levels_era, lats_era, lons_era),
        ds['v'].values,
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Convert altitude (ft) to pressure (hPa)
    pressures = altitude_to_pressure_hpa(mission_data['H'])
    
    # Convert lon to [0, 360] for ERA5 grid compatibility
    lons_360 = [(lon + 360) if lon < 0 else lon for lon in mission_data['lons']]
    
    # Form (pressure, lat, lon) points for interpolation
    points = np.array([
        (p, lat, lon) for p, lat, lon in zip(pressures, mission_data['lats'], lons_360)
    ])
    
    # Interpolate
    u_vals = u_interp(points)
    v_vals = v_interp(points)
    
    wind_speed = np.sqrt(u_vals**2 + v_vals**2)
    
    return u_vals, v_vals, wind_speed


def get_tas(mission_data, era5_path):

    # Get wind components based on path
    u_vals, v_vals, wind_speed = get_wind_at_points(mission_data, era5_path)
    
    # Compute geodesic track angles between consecutive flight segments
    geod = Geod(ellps="WGS84")
    lons = mission_data["lons"]
    lats = mission_data["lats"]
    gs = mission_data["GS"]
    
    track_angles = []
    headings = []
    drifts = []
    
    # Loop over all points along the arc
    for i in range(len(lons) - 1):
        # Compute track angle for the two lat-lon pairs
        lon1, lat1 = lons[i], lats[i]
        lon2, lat2 = lons[i + 1], lats[i + 1]
        az_fwd, _, _ = geod.inv(lon1, lat1, lon2, lat2)
        track = az_fwd % 360
        track_angles.append(track)
        
    # Use wind triangle to compute heading and drift
        track_rad = np.deg2rad(track)
        vgx = gs[i] * np.cos(track_rad)  # ground speed-x along track
        vgy = gs[i] * np.sin(track_rad)  # ground speed-y along track
        
        
        # Substract wind vector (souce ERA-5 dummy weather)
        vax = vgx - u_vals[i]   # TAS x-component
        vay = vgy - v_vals[i]   # TAS y-component
        
        tas = np.sqrt(vax**2 + vay**2)
        heading_rad = np.arctan2(vay, vax)
        heading_deg = np.rad2deg(heading_rad) % 360
        
        # Compute path drift
        drift = (heading_deg - track + 360) % 360
        if drift > 180:
            drift -= 360  # Convert to signed drift
        headings.append(heading_deg)
        drifts.append(drift)
        
    return (
        np.array(track_angles),
        np.array(headings),
        np.array(drifts),
        u_vals,
        v_vals,
        wind_speed
    )