import json

import pytest
from pydantic import ValidationError

from trip_agent.env.loader import DataError, load_catalog, load_scenarios, validate_dataset, write_dataset
from trip_agent.env.models import Catalog, OpeningHours, format_clock, parse_clock


def test_clock_round_trip():
    assert parse_clock("09:30") == 570
    assert parse_clock("24:00") == 1440
    assert format_clock(570) == "09:30"
    for bad in ("9:75", "25:00", "noon", True):
        with pytest.raises(ValueError):
            parse_clock(bad)


def test_opening_hours_must_be_ordered():
    with pytest.raises(ValidationError):
        OpeningHours(open="18:00", close="09:00")


def test_catalog_rejects_duplicate_ids_and_dangling_legs(catalog):
    data = catalog.model_dump(mode="json")
    dup = json.loads(json.dumps(data))
    dup["hotels"].append(dup["hotels"][0])
    with pytest.raises(ValidationError, match="duplicate place id"):
        Catalog.model_validate(dup)
    dangling = json.loads(json.dumps(data))
    dangling["travel"].append({"from_id": "a_aquarium", "to_id": "nowhere", "minutes": 5})
    with pytest.raises(ValidationError, match="unknown place"):
        Catalog.model_validate(dangling)


def test_symmetric_travel_lookup(catalog):
    leg = catalog.leg("h_copley", "a_art_museum")
    assert leg is not None
    assert catalog.leg("a_art_museum", "h_copley") is leg or catalog.leg("a_art_museum", "h_copley") == leg


def test_example_dataset_is_valid(catalog, scenarios):
    report = validate_dataset(catalog, scenarios)
    assert report.ok, report.errors
    assert not report.warnings


def test_round_trip_through_disk(tmp_path, catalog, scenarios):
    write_dataset(tmp_path, catalog, scenarios)
    assert load_catalog(tmp_path) == catalog
    assert [s.id for s in load_scenarios(tmp_path)] == [s.id for s in scenarios]


def test_validation_catches_infeasible_reference(catalog, scenarios):
    broken = scenarios[0].model_copy(deep=True)
    broken.reference_itinerary.items[3].start = parse_clock("16:00")  # art museum would end 18:00, closes 17:00
    report = validate_dataset(catalog, [broken])
    assert not report.ok
    assert any("opening_hours" in e for e in report.errors)


def test_missing_files_raise_data_error(tmp_path):
    with pytest.raises(DataError):
        load_catalog(tmp_path)
    with pytest.raises(DataError):
        load_scenarios(tmp_path)


def test_dataset_hash_ignores_json_key_order(catalog, scenarios):
    """Postgres jsonb returns object keys in its own order; the hash must not change."""
    from trip_agent.env.loader import dataset_hash
    from trip_agent.env.models import Scenario

    shuffled = []
    for s in scenarios:
        data = s.model_dump(mode="json")
        windows = data["constraints"]["meal_windows"]
        data["constraints"]["meal_windows"] = dict(reversed(list(windows.items())))
        shuffled.append(Scenario.model_validate(data))
    assert dataset_hash(catalog, shuffled) == dataset_hash(catalog, scenarios)
