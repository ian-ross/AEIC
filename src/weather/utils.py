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