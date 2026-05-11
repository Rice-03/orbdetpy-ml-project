#satellite simulation v1.2 (one Nominal pass simulation only)
#this code only simulates the change distance between the satellite and ground station change over time
#provides a range measurement and calculates the signal delay based on the range and speed of light
#plots the range and signal delay over time since the start of the pass, and saves the data to a CSV file

#creates the simulation settings, adds a ground station, selects the measurement type
from orbdetpy import (configure, add_station, MeasurementType)
#handles time conversions
from orbdetpy.conversion import get_J2000_epoch_offset, get_UTC_string
#runs the forwards orbit propagation and generates simulated measurements
from orbdetpy.propagation import propagate_orbits
import matplotlib.pyplot as plt
import csv

# Setting up simulatiodn settings
cfg = configure(rso_mass=2500.0, prop_start=get_J2000_epoch_offset("2019-05-01T00:00:00"),
                prop_initial_state=[-23183898.259, 35170229.755, 43425.075, -2566.938, -1692.19, 138.948],
                prop_end=get_J2000_epoch_offset("2019-05-01T01:00:00"), prop_step=300.0, sim_measurements=True)

# Define ground stations
add_station(cfg, "Maui", 0.3614, -2.7272, 3059.0)

# Use range as the simulated measurement type
cfg.measurements[MeasurementType.RANGE].error[:] = [10.0]

# Propagate orbits and generate measurements
meas = list(m for m in propagate_orbits([cfg])[0].array)

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

#saving to a CSV file
Orbital_Data = "orbital_data.csv"
with open(Orbital_Data, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["UTC Time", "Range (m)", "Delay (s)"])
    for time, range_value, delay in zip(times, ranges, delays):
        writer.writerow([time, range_value, delay])
print(f"\nData saved to {Orbital_Data}")

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
plt.show()