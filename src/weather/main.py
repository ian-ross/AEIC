import json
import draw as dr
import utils as util


mission_path = "../missions/sample_missions_10.json"

# Load JSON data
with open(mission_path, 'r') as file:
    missions = json.load(file)
    

first_mission = missions[0]

#dr.plot_flight_arc(first_mission)

util.get_flight_track(first_mission)







