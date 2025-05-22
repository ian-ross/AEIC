from pyproj import Geod
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xarray as xr

def get_flight_track(trajectory):
    
    # Instantiate WGS84 ellipsoid
    geod = Geod(ellps ="WGS84")

    lons = trajectory["lons"]
    lats = trajectory["lats"]
    
    track_angles = []
    for lon1, lat1, lon2, lat2 in zip(lons[:-1], lats[:-1], lons[1:], lats[1:]):
        azimuth_fwd, _, _ = geod.inv(lon1, lat1, lon2, lat2)
        track_angles.append(azimuth_fwd % 360)


    

def get_wind_at_points(era5_path):
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
    
    print(v_interp)
        