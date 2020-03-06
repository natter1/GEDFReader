import pickle
from pathlib import Path
from typing import List

from gdef_reader.gdef_measurement import GDEFMeasurement


def load_pygdf_measurements(path: Path) -> List[GDEFMeasurement]:
    result = []
    files = path.rglob("*.pygdf")

    for filename in files:
        print(filename)
        with open(filename, 'rb') as file:
            measurement = pickle.load(file)
            measurement.filename = filename
            result.append(measurement)
    return result