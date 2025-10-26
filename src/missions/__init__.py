from dataclasses import dataclass
from datetime import datetime

from utils.units import NAUTICAL_MILES_TO_METERS

from .database import Database  # noqa
from .filter import BoundingBox, Filter  # noqa
from .query import CountQuery, FrequentFlightQuery, Query  # noqa


@dataclass
class Location:
    longitude: float
    latitude: float
    altitude: float


@dataclass
class Mission:
    dep_airport: str
    arr_airport: str
    dep_location: Location
    arr_location: Location
    dep_datetime: datetime
    arr_datetime: datetime
    gc_distance: float
    load_factor: float
    ac_code: str

    @classmethod
    def from_toml(cls, data: dict) -> list['Mission']:
        result = []
        for f in data['flight']:
            dep_loc = Location(
                longitude=f['dep_location'][0],
                latitude=f['dep_location'][1],
                altitude=f['dep_location'][2],
            )
            arr_loc = Location(
                longitude=f['arr_location'][0],
                latitude=f['arr_location'][1],
                altitude=f['arr_location'][2],
            )
            result.append(
                cls(
                    dep_airport=f['dep_airport'],
                    arr_airport=f['arr_airport'],
                    dep_location=dep_loc,
                    arr_location=arr_loc,
                    gc_distance=f['distance_nm'] * NAUTICAL_MILES_TO_METERS,
                    dep_datetime=datetime.fromisoformat(f['dep_datetime']),
                    arr_datetime=datetime.fromisoformat(f['arr_datetime']),
                    load_factor=f['load_factor'],
                    ac_code=f['ac_code'],
                )
            )
        return result
