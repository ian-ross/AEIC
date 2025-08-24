from missions.custom_types import DayOfWeek
from missions.OAG_db import OAGDatabase


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
