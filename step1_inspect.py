import orbdetpy

print("Top-level names:\n")
for name in sorted(dir(orbdetpy)):
    if not name.startswith("_"):
        print(name)

print("\n--- key objects exist? ---")
for name in ["configure", "add_station", "build_measurement", "add_maneuver", "MeasurementType"]:
    print(name, hasattr(orbdetpy, name))