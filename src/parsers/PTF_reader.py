# import re
# import json

# def parse_climb_data(input_data):
#     """
#     Reads a PTF file and extracts Flight Level (FL) data
#     for climb performance into a structured dict.
#     store rocd_lo, rocd_hi, and fuel (just nominal).
#     """
#     climb_data = {
#         "flight_levels_ft": [],
#         "tas": [],
#         "rocd_lo": [],
#         "rocd_hi": [],
#         "fuel_flow_lo": [],  #store the single nominal climb fuel in both
#         "fuel_flow_hi": []   # "lo" and "hi" for lack of separate values
#     }
#     capture = False

#     for line in input_data:
#         # Start capturing after detecting the FL table header
#         if "FL |" in line:
#             capture = True
#             continue
        
#         if capture:
#             # Extract FL values and corresponding data
#             parts = line.split("|")
#             if len(parts) < 4:
#                 # We expect 4 columns:
#                 #   [0]: FL
#                 #   [1]: CRUISE columns
#                 #   [2]: CLIMB columns
#                 #   [3]: DESCENT columns
#                 continue
            
#             # Try to parse the flight level from column 0
#             try:
#                 fl = int(parts[0].strip())
#             except ValueError:
#                 continue  # Skip lines that don't have numeric FL

#             # Climb data is in parts[2]
#             climb_match = re.findall(r"\d+\.?\d*", parts[2])
#             # Based on the sample file, we expect:
#             #   climb_match[0] = TAS
#             #   climb_match[1] = ROCD low
#             #   climb_match[2] = ROCD nominal
#             #   climb_match[3] = ROCD high
#             #   climb_match[4] = fuel nominal
#             if len(climb_match) >= 5:
#                 climb_data["flight_levels_ft"].append(fl)
#                 climb_data["tas"].append(float(climb_match[0]))
#                 climb_data["rocd_lo"].append(float(climb_match[1]))
#                 climb_data["rocd_hi"].append(float(climb_match[3]))
#                 # The file only provides nominal fuel for climb => replicate it in both lo/hi
#                 nominal_fuel = float(climb_match[4])
#                 climb_data["fuel_flow_lo"].append(nominal_fuel)
#                 climb_data["fuel_flow_hi"].append(nominal_fuel)

#     return climb_data


# def parse_cruise_data(input_data):
#     """
#     Reads a PTF file and extracts Flight Level (FL) data
#     for cruise performance into a structured dict.
#     We'll store the TAS, and the fuel_flow lo/high, ignoring nominal.
#     """
#     cruise_data = {
#         "flight_levels_ft": [],
#         "tas": [],
#         "rocd_lo": [],    # always 0 for cruise
#         "rocd_hi": [],    # always 0 for cruise
#         "fuel_flow_lo": [],
#         "fuel_flow_hi": []
#     }
#     capture = False

#     for line in input_data:
#         if "FL |" in line:
#             capture = True
#             continue
        
#         if capture:
#             parts = line.split("|")
#             if len(parts) < 4:
#                 continue
            
#             # Flight level
#             try:
#                 fl = int(parts[0].strip())
#             except ValueError:
#                 continue

#             # Cruise data is in parts[1]
#             cruise_match = re.findall(r"\d+\.?\d*", parts[1])
#             # Based on the sample file for the cruise columns, we expect:
#             #   cruise_match[0] = TAS
#             #   cruise_match[1] = fuel low
#             #   cruise_match[2] = fuel nominal
#             #   cruise_match[3] = fuel high
#             if len(cruise_match) >= 4:
#                 cruise_data["flight_levels_ft"].append(fl)
#                 cruise_data["tas"].append(float(cruise_match[0]))
#                 cruise_data["rocd_lo"].append(0.0)  # no climb/descent in cruise
#                 cruise_data["rocd_hi"].append(0.0)
#                 # ignoring nominal => keep low in [0], high in [1]
#                 cruise_data["fuel_flow_lo"].append(float(cruise_match[1]))
#                 cruise_data["fuel_flow_hi"].append(float(cruise_match[3]))

#     return cruise_data


# def parse_descent_data(input_data):
#     """
#     Reads a PTF file and extracts Flight Level (FL) data
#     for descent performance into a structured dict.
#     The sample file shows only nominal values for ROCD & fuel. We'll store
#     them in both "lo" and "hi" to match the two-column approach.
#     """
#     descent_data = {
#         "flight_levels_ft": [],
#         "tas": [],
#         "rocd_lo": [],
#         "rocd_hi": [],
#         "fuel_flow_lo": [],
#         "fuel_flow_hi": []
#     }
#     capture = False

#     for line in input_data:
#         if "FL |" in line:
#             capture = True
#             continue
        
#         if capture:
#             parts = line.split("|")
#             if len(parts) < 4:
#                 continue
            
#             # Flight level
#             try:
#                 fl = int(parts[0].strip())
#             except ValueError:
#                 continue

#             # Descent data is in parts[3]
#             descent_match = re.findall(r"\d+\.?\d*", parts[3])
#             # Based on the sample file for the descent columns, we expect:
#             #   descent_match[0] = TAS nominal
#             #   descent_match[1] = ROCD nominal
#             #   descent_match[2] = fuel nominal
#             if len(descent_match) >= 3:
#                 descent_data["flight_levels_ft"].append(fl)
#                 descent_data["tas"].append(float(descent_match[0]))
#                 rocd_nom = float(descent_match[1])
#                 fuel_nom = float(descent_match[2])
#                 # replicate nominal => lo & hi
#                 descent_data["rocd_lo"].append(rocd_nom)
#                 descent_data["rocd_hi"].append(rocd_nom)
#                 descent_data["fuel_flow_lo"].append(fuel_nom)
#                 descent_data["fuel_flow_hi"].append(fuel_nom)

#     return descent_data


# def parse_max_alt_payload(input_data):
#     """
#     Reads an PTF and extracts Max Altitude [ft] and Max Payload [kg].
#     """
#     max_alt = None
#     max_payload = None
    
#     for line in input_data:
#         # Search for Max Altitude
#         match_alt = re.search(r"Max Alt\.\s*\[ft\]:\s*([\d,]+)", line)
#         if match_alt:
#             max_alt = int(match_alt.group(1).replace(",", ""))  # Remove commas if present
        
#         # Search for Max Payload
#         match_payload = re.search(r"Max Payload\s*\[kg\]:\s*([\d,]+)", line)
#         if match_payload:
#             max_payload = int(match_payload.group(1).replace(",", ""))  # Remove commas if present

#     return {"max_alt_ft": max_alt, "max_payload_kg": max_payload}


# def parse_mass_levels(input_data):
#     """
#     Reads an PTF and extracts the low, nominal, and high mass levels.
#     """
#     low_mass = None
#     nominal_mass = None
#     high_mass = None

#     for line in input_data:
#         # Low mass
#         if 'low' in line and 'climb' in line:
#             match = re.search(r'low\s*-\s*(\d+)', line)
#             if match:
#                 low_mass = int(match.group(1))
#         # Nominal mass
#         elif 'nominal' in line and 'cruise' in line:
#             match = re.search(r'nominal\s*-\s*(\d+)', line)
#             if match:
#                 nominal_mass = int(match.group(1))
#         # High mass
#         elif 'high' in line and 'descent' in line:
#             match = re.search(r'high\s*-\s*(\d+)', line)
#             if match:
#                 high_mass = int(match.group(1))

#     return {
#         'low_mass_kg': low_mass,
#         'nominal_mass_kg': nominal_mass,
#         'high_mass_kg': high_mass
#     }


# def parse_PTF(file_path):
#     """
#     Main parser that reads a PTF and returns
#     a dict of the form:
#     {
#       "phases": {
#         "climb": {
#           "flight_levels_ft": [...],
#           "rocd_lo": [...],
#           "rocd_hi": [...],
#           "tas": [...],
#           "fuel_flow_lo": [...],
#           "fuel_flow_hi": [...]
#         },
#         "cruise": { ... },
#         "descent": { ... }
#       },
#       "max_alt_ft": ...,
#       "max_payload_kg": ...,
#       "low_mass_kg": ...,
#       "nominal_mass_kg": ...,
#       "high_mass_kg": ...
#     }
#     """
#     with open(file_path, 'r', errors='ignore') as file:
#         input_data = file.read()
#     # Parse the separate pieces
#     climb = parse_climb_data(input_data)
#     cruise = parse_cruise_data(input_data)
#     descent = parse_descent_data(input_data)
#     alt_payload = parse_max_alt_payload(input_data)
#     masses = parse_mass_levels(input_data)
    
#     data = {
#         "phases": {
#             "climb": climb,
#             "cruise": cruise,
#             "descent": descent
#         }
#     }
#     # Add in the top-level info
#     data.update(alt_payload)
#     data.update(masses)

#     return data

import re
import json

def parse_PTF(file_path):
    """
    Reads the TASOPT performance file in a single pass and returns a dict:
      {
        "phases": {
          "climb": {
            "flight_levels_ft": [...],
            "rocd_lo": [...],
            "rocd_hi": [...],
            "tas": [...],
            "fuel_flow_lo": [...],
            "fuel_flow_hi": [...]
          },
          "cruise": { ... },
          "descent": { ... }
        },
        "max_alt_ft": ...,
        "max_payload_kg": ...,
        "low_mass_kg": ...,
        "nominal_mass_kg": ...,
        "high_mass_kg": ...
      }
    """
    
    # Prepare data structures for phases
    climb_data = {
        "flight_levels_ft": [],
        "tas": [],
        "rocd_lo": [],
        "rocd_hi": [],
        "fuel_flow_lo": [],
        "fuel_flow_hi": [],
    }
    cruise_data = {
        "flight_levels_ft": [],
        "tas": [],
        "rocd_lo": [],
        "rocd_hi": [],
        "fuel_flow_lo": [],
        "fuel_flow_hi": [],
    }
    descent_data = {
        "flight_levels_ft": [],
        "tas": [],
        "rocd_lo": [],
        "rocd_hi": [],
        "fuel_flow_lo": [],
        "fuel_flow_hi": [],
    }
    
    # Prepare storage for top-level info
    max_alt = None
    max_payload = None
    low_mass = None
    nominal_mass = None
    high_mass = None

    capture = False

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 1) Parse top-level lines before/after the table for alt/payload/mass:
            if not capture:
                # Max altitude
                match_alt = re.search(r"Max Alt\.\s*\[ft\]:\s*([\d,]+)", line)
                if match_alt:
                    max_alt = int(match_alt.group(1).replace(",", ""))
                
                # Max payload
                match_payload = re.search(r"Max Payload\s*\[kg\]:\s*([\d,]+)", line)
                if match_payload:
                    max_payload = int(match_payload.group(1).replace(",", ""))

                # Mass levels
                # low mass
                if 'low' in line and 'climb' in line:
                    match = re.search(r'low\s*-\s*(\d+)', line)
                    if match:
                        low_mass = int(match.group(1))
                # nominal mass
                elif 'nominal' in line and 'cruise' in line:
                    match = re.search(r'nominal\s*-\s*(\d+)', line)
                    if match:
                        nominal_mass = int(match.group(1))
                # high mass
                elif 'high' in line and 'descent' in line:
                    match = re.search(r'high\s*-\s*(\d+)', line)
                    if match:
                        high_mass = int(match.group(1))

            # 2) Detect the start of the flight-level table
            if "FL |" in line:
                capture = True
                continue

            # 3) Once capturing, parse columns for FL, climb, cruise, descent
            if capture:
                parts = line.split("|")
                # We expect 4 columns: FL, CRUISE, CLIMB, DESCENT
                if len(parts) < 4:
                    continue
                
                # Flight level in parts[0]
                try:
                    fl = int(parts[0].strip())
                except ValueError:
                    # If we can't parse a valid FL, skip
                    continue

                # ============ CRUISE ============ 
                cruise_str = parts[1]
                # Typically we expect [TAS, fuel_low, fuel_nom, fuel_high]
                # (or sometimes fewer if blank line, so check carefully)
                c_vals = re.findall(r"\d+\.?\d*", cruise_str)
                if len(c_vals) >= 4:
                    # TAS
                    c_tas = float(c_vals[0])
                    # Fuel flows ignoring nominal => Low and High
                    c_fuel_lo = float(c_vals[1])
                    c_fuel_hi = float(c_vals[3])
                else:
                    # If line is blank for cruise, skip
                    c_tas = None

                # ============ CLIMB ============ 
                climb_str = parts[2]
                # Typically we expect [TAS, rocd_lo, rocd_nom, rocd_hi, fuel_nom]
                cl_vals = re.findall(r"\d+\.?\d*", climb_str)
                if len(cl_vals) >= 5:
                    # TAS
                    cl_tas = float(cl_vals[0])
                    # ROCD
                    cl_rocd_lo = float(cl_vals[1])
                    cl_rocd_hi = float(cl_vals[3])    # ignoring nominal
                    # Fuel: replicate nominal in lo & hi
                    cl_fuel_nom = float(cl_vals[4])
                else:
                    # If line is blank for climb, skip
                    cl_tas = None

                # ============ DESCENT ============ 
                descent_str = parts[3]
                # Typically we expect [TAS_nom, rocd_nom, fuel_nom]
                d_vals = re.findall(r"\d+\.?\d*", descent_str)
                if len(d_vals) >= 3:
                    d_tas_nom = float(d_vals[0])
                    # TODO: For now just putting rocd for descent as negative
                    d_rocd_nom = -float(d_vals[1])
                    d_fuel_nom = float(d_vals[2])
                else:
                    d_tas_nom = None

                # Now store them if they exist
                # -- Cruise
                if c_tas is not None:
                    cruise_data["flight_levels_ft"].append(fl)
                    cruise_data["tas"].append(c_tas)
                    # Cruise rocd is always 0
                    cruise_data["rocd_lo"].append(0.0)
                    cruise_data["rocd_hi"].append(0.0)
                    cruise_data["fuel_flow_lo"].append(c_fuel_lo)
                    cruise_data["fuel_flow_hi"].append(c_fuel_hi)

                # -- Climb
                if cl_tas is not None:
                    climb_data["flight_levels_ft"].append(fl)
                    climb_data["tas"].append(cl_tas)
                    climb_data["rocd_lo"].append(cl_rocd_lo)
                    climb_data["rocd_hi"].append(cl_rocd_hi)
                    climb_data["fuel_flow_lo"].append(cl_fuel_nom)
                    climb_data["fuel_flow_hi"].append(cl_fuel_nom)

                # -- Descent
                if d_tas_nom is not None:
                    descent_data["flight_levels_ft"].append(fl)
                    descent_data["tas"].append(d_tas_nom)
                    # replicate nominal rocd in both lo & hi
                    descent_data["rocd_lo"].append(d_rocd_nom)
                    descent_data["rocd_hi"].append(d_rocd_nom)
                    descent_data["fuel_flow_lo"].append(d_fuel_nom)
                    descent_data["fuel_flow_hi"].append(d_fuel_nom)

    # Build final data structure
    data = {
        "phases": {
            "climb": climb_data,
            "cruise": cruise_data,
            "descent": descent_data
        },
        # Top-level info
        "max_alt_ft": max_alt,
        "max_payload_kg": max_payload,
        "low_mass_kg": low_mass,
        "nominal_mass_kg": nominal_mass,
        "high_mass_kg": high_mass
    }
    return data
