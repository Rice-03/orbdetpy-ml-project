import orbdetpy
import inspect

targets = [
    orbdetpy.configure,
    orbdetpy.add_station,
    orbdetpy.build_measurement,
    orbdetpy.add_maneuver,
]

for obj in targets:
    print("\n" + "=" * 80)
    print(obj.__name__)
    print("=" * 80)
    print(inspect.signature(obj))
    print()
    print(inspect.getdoc(obj))

print("\n" + "=" * 80)
print("MeasurementType members")
print("=" * 80)
for name in dir(orbdetpy.MeasurementType):
    if not name.startswith("_"):
        print(name)