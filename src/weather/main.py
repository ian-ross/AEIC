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



#dr.plot_flight_arc(first_mission)

#util.get_flight_track(first_mission)







