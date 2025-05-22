import json
import draw as dr
import utils as util
import trajectory as traj

mission_path = "../missions/sample_missions_10.json"

# Load JSON data
with open(mission_path, 'r') as file:
    missions = json.load(file)
    

# Select a dummy mission for npw
first_mission = missions[0]


# Get mission points based on mission def
trajectory = traj.get_mission_points(first_mission)

# Print the first few points
for lon, lat, gs, alt in zip(trajectory["lons"][:5], trajectory["lats"][:5], trajectory["GS"][:5], trajectory["H"][:5]):
    print(f"Lon: {lon:.2f}, Lat: {lat:.2f}, GS: {gs} kt, Alt: {alt} ft")


#util.get_flight_track(trajectory)

u, v, windmag = util.get_wind_at_points(trajectory, "ERA5/sample.grib")



#dr.plot_flight_arc(first_mission)

#util.get_flight_track(first_mission)







