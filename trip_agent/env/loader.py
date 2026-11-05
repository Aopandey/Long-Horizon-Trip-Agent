"""Load and validate the synthetic dataset from disk.

Layout of a data directory:
    catalog.json                 one Catalog
    scenarios/*.json             one Scenario per file   (or scenarios.json: a list)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from trip_agent.checker.consistency import TruthView, check_itinerary
from trip_agent.env.models import Attraction, Catalog, ItemKind, Scenario


class DataError(ValueError):
    pass


def load_catalog(data_dir: Path) -> Catalog:
    path = Path(data_dir) / "catalog.json"
    if not path.exists():
        raise DataError(f"missing {path}")
    try:
        return Catalog.model_validate_json(path.read_text())
    except ValidationError as exc:
        raise DataError(f"{path}: {exc}") from exc


def load_scenarios(data_dir: Path) -> list[Scenario]:
    data_dir = Path(data_dir)
    raw: list[tuple[str, object]] = []
    if (data_dir / "scenarios.json").exists():
        items = json.loads((data_dir / "scenarios.json").read_text())
        raw += [(f"scenarios.json[{i}]", s) for i, s in enumerate(items)]
    if (data_dir / "scenarios").is_dir():
        raw += [(p.name, json.loads(p.read_text())) for p in sorted((data_dir / "scenarios").glob("*.json"))]
    if not raw:
        raise DataError(f"no scenarios in {data_dir} (expected scenarios.json or scenarios/*.json)")
    scenarios, errors = [], []
    for where, obj in raw:
        try:
            scenarios.append(Scenario.model_validate(obj))
        except ValidationError as exc:
            errors.append(f"{where}: {exc}")
    if errors:
        raise DataError("\n".join(errors))
    ids = [s.id for s in scenarios]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise DataError(f"duplicate scenario ids: {dupes}")
    return scenarios


def write_dataset(data_dir: Path, catalog: Catalog, scenarios: list[Scenario]) -> None:
    data_dir = Path(data_dir)
    (data_dir / "scenarios").mkdir(parents=True, exist_ok=True)
    (data_dir / "catalog.json").write_text(catalog.model_dump_json(indent=2))
    for s in scenarios:
        (data_dir / "scenarios" / f"{s.id}.json").write_text(s.model_dump_json(indent=2))


def dataset_hash(catalog: Catalog, scenarios: list[Scenario]) -> str:
    """Content hash used to freeze a dataset (sorted, canonical JSON)."""
    h = hashlib.sha256(catalog.model_dump_json().encode())
    for s in sorted(scenarios, key=lambda s: s.id):
        h.update(s.model_dump_json().encode())
    return h.hexdigest()[:16]


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dataset(catalog: Catalog, scenarios: list[Scenario]) -> ValidationReport:
    """Cross-checks that JSON Schema cannot express: references, feasibility, horizon counts."""
    report = ValidationReport()
    for s in scenarios:
        c = s.constraints
        for aid in c.required_attractions:
            if not isinstance(catalog.place(aid), Attraction):
                report.errors.append(f"{s.id}: required attraction {aid!r} is not in the catalog")
        if c.hotel_neighborhoods:
            hoods = {h.neighborhood for h in catalog.hotels}
            if not hoods & set(c.hotel_neighborhoods):
                report.errors.append(f"{s.id}: no hotel in allowed neighborhoods {c.hotel_neighborhoods}")
        truth = check_itinerary(s.reference_itinerary, s, catalog, TruthView(catalog))
        for v in truth.violations:
            report.errors.append(f"{s.id}: reference itinerary violates {v.name}: {v.message}")
        n_ref = len(s.reference_itinerary.items)
        if n_ref != s.horizon:
            report.warnings.append(f"{s.id}: horizon={s.horizon} but reference itinerary has {n_ref} items")
        if s.required_commitments() > s.horizon:
            report.warnings.append(
                f"{s.id}: {s.required_commitments()} required commitments exceed horizon={s.horizon}"
            )
        visits = [i for i in s.reference_itinerary.items if i.kind == ItemKind.visit]
        if truth.total_cost is not None:
            slack = (c.budget - truth.total_cost) / c.budget
            report.info.append(
                f"{s.id}: reference cost {truth.total_cost:.2f}, budget slack {slack:.0%}, {len(visits)} visits"
            )
    ids = sorted(catalog.places)
    missing = [
        (a, b)
        for i, a in enumerate(ids)
        for b in ids[i + 1 :]
        if catalog.leg(a, b) is None or catalog.leg(b, a) is None
    ]
    if missing:
        sample = ", ".join(f"{a}<->{b}" for a, b in missing[:5])
        report.warnings.append(
            f"{len(missing)} place pairs lack travel data in at least one direction (e.g. {sample}); "
            "an itinerary that needs one of these legs fails the travel check"
        )
    splits = {s.split for s in scenarios}
    if "eval" in splits and "dev" not in splits:
        report.warnings.append("no dev scenarios: tuning needs a separate dev split")
    return report
