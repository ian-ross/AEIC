import json
import draw as dr


mission_path = "../missions/sample_missions_10.json"

# Load JSON data
with open(mission_path, 'r') as file:
    missions = json.load(file)
    

first_mission = missions[0]

dr.plot_flight_arc(first_mission)



