import json
import draw as dr
import utils as util
import trajectory as traj

mission_path = "../missions/sample_missions_10.json"
weather_data = "ERA5/sample.grib"

# Load JSON data
with open(mission_path, 'r') as file:
    missions = json.load(file)
    

# Select a dummy mission for now
first_mission = missions[0]


# Get mission points based on mission def
trajectory = traj.create_dummy_traj(first_mission)

# Print the first few points
print("\n")
print("#--------------------------------SAMPLE TRAJECTORY (SNAPSHOT) -----------------------------#\n")
for lon, lat, tas, alt in zip(trajectory["lons"][:5], trajectory["lats"][:5], trajectory["TAS"][:5], trajectory["H"][:5]):
    print(f"Lon: {lon:.2f}, Lat: {lat:.2f}, TAS: {tas:.2f} kt, Alt: {alt:.2f} ft")
print("#------------------------------------------------------------------------------------------#\n")





#track, heading, drift, tas, u, v, wind_mag = util.get_tas(trajectory, "era5.grib")

#for i in range(5):
#    print(f"Pt {i}: Track={track[i]:.1f}°, Heading={heading[i]:.1f}°, Drift={drift[i]:+.1f}°, "
#          f"TAS={tas[i]:.1f} kt, U={u[i]:.1f}, V={v[i]:.1f}, Wind={wind_mag[i]:.1f} m/s")



#dr.plot_flight_arc(first_mission)

#util.get_flight_track(first_mission)







