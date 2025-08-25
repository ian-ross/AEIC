from typing import Any

from missions.custom_types import DayOfWeek
from missions.OAG_db import BoundingBox, OAGDatabase, OAGFilter


def test_dow_mask():
    assert (
        OAGDatabase._make_dow_mask(
            {
                DayOfWeek.MONDAY,
                DayOfWeek.TUESDAY,
                DayOfWeek.WEDNESDAY,
                DayOfWeek.THURSDAY,
                DayOfWeek.FRIDAY,
                DayOfWeek.SATURDAY,
                DayOfWeek.SUNDAY,
            }
        )
        == 0b01111111
    )
    assert (
        OAGDatabase._make_dow_mask(
            {DayOfWeek.MONDAY, DayOfWeek.WEDNESDAY, DayOfWeek.FRIDAY}
        )
        == 0b00010101
    )
    assert (
        OAGDatabase._make_dow_mask({DayOfWeek.TUESDAY, DayOfWeek.THURSDAY})
        == 0b00001010
    )
    assert OAGDatabase._make_dow_mask({DayOfWeek.SUNDAY}) == 0b01000000
    assert OAGDatabase._make_dow_mask(set()) == 0b0000000


def check_simple_filter(filter: OAGFilter, *checks: tuple[str, Any]):
    conds, params = filter.to_sql()
    cps = set(zip(conds, params))
    for check in checks:
        assert check in cps


def test_oag_filter():
    check_simple_filter(
        OAGFilter(min_distance=1000, max_distance=5000),
        ('distance >= ?', 1000),
        ('distance <= ?', 5000)
    )

    conds, params = OAGFilter(country='US').to_sql()
    assert len(conds) == 1
    assert len(params) == 2
    assert conds[0] == (
        '(origin IN (SELECT id FROM airports WHERE country IN (?)) OR '
        'destination IN (SELECT id FROM airports WHERE country IN (?)))'
    )
    assert params == ['US', 'US']

    conds, params = OAGFilter(continent='SA').to_sql()
    assert len(conds) == 1
    assert len(params) == 2
    assert conds[0] == (
        '(origin IN (SELECT id FROM airports WHERE country IN '
        '(SELECT code FROM countries WHERE continent IN (?))) OR '
        'destination IN (SELECT id FROM airports WHERE country IN '
        '(SELECT code FROM countries WHERE continent IN (?))))'
    )
    assert params == ['SA', 'SA']

    # Bounding box for Austria.
    bbox = BoundingBox(
        min_longitude=11.343, max_longitude=16.570,
        min_latitude=46.642, max_latitude=48.234
    )
    conds, params = OAGFilter(bounding_box=bbox).to_sql()
    assert len(conds) == 1
    assert len(params) == 8
    assert conds[0] == (
        '(origin IN (SELECT id FROM airport_location_idx '
        'WHERE min_latitude >= ? AND max_latitude <= ? AND '
        'min_longitude >= ? AND max_longitude <= ?) OR '
        'destination IN (SELECT id FROM airport_location_idx '
        'WHERE min_latitude >= ? AND max_latitude <= ? AND '
        'min_longitude >= ? AND max_longitude <= ?))'
    )
    assert params == [46.642, 48.234, 11.343, 16.57, 46.642, 48.234, 11.343, 16.57]

    conds, params = OAGFilter(aircraft_type=['B737', '777']).to_sql()
    assert len(conds) == 1
    assert len(params) == 2
    assert conds[0] == 'aircraft_type IN (?, ?)'
    assert params == ['B737', '777']

    conds, params = OAGFilter(
        min_distance=2000,
        min_seat_capacity=200,
        country=['US', 'CA']
    ).to_sql()
    assert len(conds) == 3
    assert len(params) == 6
    assert conds[0] == 'distance >= ?'
    assert conds[1] == 'seat_capacity >= ?'
    assert conds[2] == (
        '(origin IN (SELECT id FROM airports WHERE country IN (?, ?)) OR '
        'destination IN (SELECT id FROM airports WHERE country IN (?, ?)))'
    )
    assert params == [2000, 200, 'US', 'CA', 'US', 'CA']
