#satellite simulation v1.0 (one Nominal pass simulation only)
#this code only simulates the change distance between the satellite and ground station change over time
#provides a range vs time sequence

#creates the simulation settings, adds a ground station, selects the measurement type
from orbdetpy import (configure, add_station, MeasurementType)
#handles time conversions
from orbdetpy.conversion import get_J2000_epoch_offset
#runs the forwards orbit propagation and generates simulated measurements
from orbdetpy.propagation import propagate_orbits

# Setting up simulation settings
cfg = configure(rso_mass=2500.0, prop_start=get_J2000_epoch_offset("2019-05-01T00:00:00"),
                prop_initial_state=[-23183898.259, 35170229.755, 43425.075, -2566.938, -1692.19, 138.948],
                prop_end=get_J2000_epoch_offset("2019-05-01T01:00:00"), prop_step=300.0, sim_measurements=True)

# Define ground stations
add_station(cfg, "Maui", 0.3614, -2.7272, 3059.0)

# Use range as the simulated measurement type
cfg.measurements[MeasurementType.RANGE].error[:] = [10.0]

# Propagate orbits and generate measurements
meas = list(m for m in propagate_orbits([cfg])[0].array)

# Inspect the output
print("Number of measurements:", len(meas))
print("\nFirst measurement object:")
print("Satellite: RBH-SAT1 ")
print(meas[-1])
