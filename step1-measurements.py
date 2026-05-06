import orbdetpy

for name in dir(orbdetpy.MeasurementType):
    if not name.startswith("_"):
        print(name, getattr(orbdetpy.MeasurementType, name))