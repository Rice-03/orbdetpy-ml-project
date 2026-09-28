#satellite simulation v1.4 (one Nominal pass simulation only)
#this code only simulates the change distance between the satellite and ground station change over time
#provides a range measurement and calculates the signal delay based on the range and speed of light
#plots the range and signal delay over time since the start of the pass, and saves the data to a CSV file
#saves simulation metadata to a JSON file and outputs both files into an output folder
#the satellite is a realistic LEO orbit (ISS-like) built from Keplerian elements, visible for ~6 minutes

#creates the simulation settings, adds a ground station, selects the measurement type
from orbdetpy import (configure, add_station, MeasurementType, Frame, PositionAngle)
#handles time conversions and orbital-element-to-state-vector conversion
from orbdetpy.conversion import get_J2000_epoch_offset, get_UTC_string, elem_to_pv
#runs the forwards orbit propagation and generates simulated measurements
from orbdetpy.propagation import propagate_orbits
import matplotlib.pyplot as plt
import math
import csv
import json
import os

# Real ISS-like LEO orbital elements (used to build a realistic starting position/velocity)
sma  = 6798000.0           # semi-major axis [m]  -> ~416 km altitude
ecc  = 0.0006              # eccentricity         -> near-circular orbit
inc  = math.radians(51.6)  # inclination [rad]    -> real ISS value
raan = math.radians(60.0)  # right ascension of ascending node [rad]
argp = 0.0                 # argument of perigee [rad]
anom = 0.0                 # mean anomaly [rad]   -> starting point on the orbit

# Reference epoch for the orbit (the orbital elements are defined at this instant)
epoch = get_J2000_epoch_offset("2019-05-01T00:00:00")

# Convert the orbital elements into a Cartesian state vector [x, y, z, vx, vy, vz]
# in the EME2000 (J2000) inertial frame, in metres and m/s
initial_state = elem_to_pv(Frame.EME2000, epoch, sma, ecc, inc, raan, argp, anom, PositionAngle.MEAN)


# Station sampling cadence and search window.
# orbdetpy only returns a measurement when the satellite is above the station horizon,
# so propagating over a window wide enough to contain the pass, at a fixed station
# sampling cadence, yields exactly the visible points of that pass. The number of
# records therefore varies naturally with the pass geometry (a higher pass lasts longer
# and produces more points) -- which is physically realistic.
SAMPLE_CADENCE_S = 30.0     # ground-station sampling rate: one measurement every 30 s
SEARCH_WINDOW_S  = 20 * 60  # 20-minute window (comfortably contains a single LEO pass)

# Setting up simulation settings
cfg = configure(rso_mass=2500.0, prop_start=epoch,
                prop_initial_state=initial_state,
                prop_end=epoch + SEARCH_WINDOW_S, prop_step=SAMPLE_CADENCE_S, sim_measurements=True)

# Define ground stations
add_station(cfg, "Maui", 0.3614, -2.7272, 3059.0)

# Use range as the simulated measurement type (10 m measurement noise)
cfg.measurements[MeasurementType.RANGE].error[:] = [10.0]

# Propagate and collect every visible measurement of the pass
meas = list(m for m in propagate_orbits([cfg])[0].array)

if not meas:
    raise SystemExit("No visible pass found for this orbit/station geometry.")

# Pass summary -- AOS/LOS and record count fall out naturally from the geometry
aos = meas[0].time               # acquisition of signal (first visible point)
los = meas[-1].time              # loss of signal (last visible point)
pass_duration_s = los - aos
num_records = len(meas)

# UTC strings for the detected pass boundaries (used in the table and metadata)
prop_start_utc = get_UTC_string(aos)
prop_end_utc   = get_UTC_string(los)

#speed of light constant
c = 299792458.0
t_start = meas[0].time
#placeholders for arrays:
times = []      #UTC time strings
times_S = []    #time since start of pass in seconds
ranges = []     #range values in meters
delays = []     #signal delay values in seconds, calculated as range / speed of light

for m in meas:
    times.append(get_UTC_string(m.time))     #convert measurement time to UTC string format
    times_S.append(m.time - t_start)         #calculate time since start of pass in seconds
    ranges.append(m.values[0])              
    delays.append(m.values[0] / c)           #calculate signal delay as range divided by speed of light

#print the results of simulation
print("\nSatellite: RBH-SAT1")
print("Ground station: Maui")

# Print the range and delay values in a formatted table
print("\nSimulated Range and Signal Delay Data")
print("-" * 75)
print(f"{'UTC Time':<25} {'Range (m)':>20} {'Delay (s)':>20}")
print("-" * 75)
for time, range_value, delay in zip(times, ranges, delays):
    print(f"{time:<25} {range_value:>20.3f} {delay:>20.9f}")
print("-" * 75)

# Create metadata dictionary
metadata = {
    "sample_id": "nominal_pass_001",
    "satellite_name": "RBH-SAT1",
    "station_name": "Maui",
    "measurement_type": "RANGE",
    "prop_start_utc": prop_start_utc,
    "prop_end_utc": prop_end_utc,
    "sample_cadence_s": SAMPLE_CADENCE_S,
    "pass_duration_s": pass_duration_s,
    "num_records": num_records,
    "range_error_m": 10.0,
    "rso_mass_kg": 2500.0,
    "prop_initial_state": list(initial_state),
    "orbital_elements": {
        "sma_m": sma,
        "ecc": ecc,
        "inc_deg": math.degrees(inc),
        "raan_deg": math.degrees(raan),
        "argp_deg": math.degrees(argp),
        "anom_deg": math.degrees(anom),
        "anom_type": "MEAN",
        "frame": "EME2000"
    }
}

# Create output folder
os.makedirs("output", exist_ok=True)

# Saving to a CSV file
Orbital_Data = "output/nominal_pass_001.csv"
with open(Orbital_Data, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["UTC Time", "Range (m)", "Delay (s)", "time_since_start_s"])
    for time, range_value, delay, t_s in zip(times, ranges, delays, times_S):
        writer.writerow([time, range_value, delay, t_s])
print(f"\nData saved to {Orbital_Data}")

# Save JSON metadata file
json_file = "output/nominal_pass_001.json"
with open(json_file, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"Metadata saved to {json_file}")

# Plotting Graphs
fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True ,figsize=(10, 8))
fig.suptitle("Satellite Range and Signal Delay Over Time")

ax1.plot(times_S,ranges,marker ="o")
ax1.set_ylabel("Range (m)")
ax1.grid(True)

ax2.plot(times_S,delays,marker ="o",color="orange")
ax2.set_ylabel("Delays (s)")
ax2.set_xlabel("Time since pass started (s)")
ax2.grid(True)

plt.tight_layout()
plot_file = "output/nominal_pass_001.png"
plt.savefig(plot_file, dpi=150)
print(f"Plot saved to {plot_file}")
plt.show()