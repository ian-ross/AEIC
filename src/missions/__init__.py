from dataclasses import dataclass
from datetime import datetime

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
    distance_nm: float
    dep_datetime: datetime
    arr_datetime: datetime
    load_factor: float

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
                    distance_nm=f['distance_nm'],
                    dep_datetime=datetime.fromisoformat(f['dep_datetime']),
                    arr_datetime=datetime.fromisoformat(f['arr_datetime']),
                    load_factor=f['load_factor'],
                )
            )
        return result
