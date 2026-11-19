"""Registry, metadata catalog, and verified downloaders for benchmark datasets.

Supports the 4D Copper paper workbook, all nine standard TSLib multivariate
forecasting benchmarks (``ETTh1``, ``ETTh2``, ``ETTm1``, ``ETTm2``, ``Weather``,
``Electricity``, ``Traffic``, ``Exchange``, ``ILI``), and aggregated hourly
multivariate **NYC Taxi Trip Record Data** from the official NYC TLC archive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import urllib.request

import numpy as np
import pandas as pd


TSLIB_HF_COMMIT = "2b66e59ee19dac8f6f19fb5d4997f289fdfea357"
TSLIB_HF_BASE = (
    f"https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/{TSLIB_HF_COMMIT}"
)
NYC_TLC_TRIP_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
NYC_TLC_ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
NYC_TLC_ZONE_SHA256 = "1a99e105092230f8620f301edcca7f80d3080642ff404d28ed957d3fa222c8ed"
NYC_TLC_PARQUET_SHA256: dict[str, str] = {
    "2024-01": "c4d59da7bbc8abaeeeb1727947ee93d9891a71acb42854bd80db1571b2030510",
    "2024-02": "c76c43c18c6c6664080dd920baab4928988d5786a6b65980792ca7cd796f9f20",
    "2024-03": "2d4cdc8fb96726cdd3803b13b02d2e61e71d45720aff0ebc693a8bdd1f249823",
    "2024-04": "ce37f736c7cf9e92164c850ef2b502a8f7bb590fc1804b6edbcb80293da01b89",
    "2024-05": "1974230be2b5a3ea92e47b5eb77f32dbf1de896d81ad8df4ab6faba52749ad23",
    "2024-06": "677cf14c8347f745b583f012fbdba072334c6c4efa17bfd6a64369f2ba30329c",
}

ETT_COLUMNS: tuple[str, ...] = ("HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT")
WEATHER_COLUMNS: tuple[str, ...] = (
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
    "rain (mm)",
    "raining (s)",
    "SWDR (W/m\ufffd)",
    "PAR (\ufffdmol/m\ufffd/s)",
    "max. PAR (\ufffdmol/m\ufffd/s)",
    "Tlog (degC)",
    "OT",
)
EXCHANGE_COLUMNS: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6", "OT")
ILI_COLUMNS: tuple[str, ...] = (
    "% WEIGHTED ILI",
    "%UNWEIGHTED ILI",
    "AGE 0-4",
    "AGE 5-24",
    "ILITOTAL",
    "NUM. OF PROVIDERS",
    "OT",
)
NYC_TAXI_COLUMNS: tuple[str, ...] = (
    "manhattan_pickups",
    "queens_pickups",
    "brooklyn_pickups",
    "other_pickups",
    "total_distance",
    "mean_duration_min",
    "mean_fare",
    "trip_count",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata and provenance specification for a registered forecasting dataset.

    Parameters
    ----------
    id : str
        Canonical lowercase dataset identifier (for example ``"etth2"`` or
        ``"nyc_taxi"``).
    label : str
        Human-readable dataset display name.
    group : str
        Source family (``"Paper"``, ``"TSLib"``, or ``"NYC TLC"``).
    data_path : str
        Canonical project-relative path under ``data/``.
    target_column : str
        Default target series name.
    features : int
        Number of numeric channels in the dataset.
    rows : int
        Expected observation row count.
    frequency : str
        Sampling cadence label (for example ``"hourly"`` or ``"15-min"``).
    columns : tuple[str, ...]
        Ordered numeric column names in the stored file.
    source_url : str
        Upstream download URL or base archive URL.
    sha256 : str or None, default=None
        Pinned hexadecimal SHA-256 digest of the canonical file when static.
    fallback_paths : tuple[str, ...], default=()
        Alternative legacy project-relative paths accepted if ``data_path`` is absent.
    description : str, default=""
        Concise domain and variable summary.
    """

    id: str
    label: str
    group: str
    data_path: str
    target_column: str
    features: int
    rows: int
    frequency: str
    columns: tuple[str, ...]
    source_url: str
    sha256: str | None = None
    fallback_paths: tuple[str, ...] = ()
    description: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "copper": DatasetSpec(
        id="copper",
        label="Copper (4D paper)",
        group="Paper",
        data_path="data/raw/paper_data.xlsx",
        target_column="Copper",
        features=4,
        rows=2008,
        frequency="daily",
        columns=("Copper", "FCX", "CLP", "SCCO"),
        source_url="https://doi.org/10.1038/s41598-025-08957-5",
        sha256=None,
        description="Four-series commodity benchmark (Copper, FCX, CLP exchange rate, SCCO).",
    ),
    "etth1": DatasetSpec(
        id="etth1",
        label="ETTh1",
        group="TSLib",
        data_path="data/external/tslib/ETTh1.csv",
        target_column="OT",
        features=7,
        rows=17420,
        frequency="hourly",
        columns=ETT_COLUMNS,
        source_url=f"{TSLIB_HF_BASE}/ETT-small/ETTh1.csv",
        sha256="f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
        fallback_paths=("data/external/etth1-1d16c8f/ETTh1.csv",),
        description="Electricity Transformer Temperature H1 (hourly, 6 load features + oil temperature OT).",
    ),
    "etth2": DatasetSpec(
        id="etth2",
        label="ETTh2",
        group="TSLib",
        data_path="data/external/tslib/ETTh2.csv",
        target_column="OT",
        features=7,
        rows=17420,
        frequency="hourly",
        columns=ETT_COLUMNS,
        source_url=f"{TSLIB_HF_BASE}/ETT-small/ETTh2.csv",
        sha256="a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
        description="Electricity Transformer Temperature H2 (hourly, second county transformer).",
    ),
    "ettm1": DatasetSpec(
        id="ettm1",
        label="ETTm1",
        group="TSLib",
        data_path="data/external/tslib/ETTm1.csv",
        target_column="OT",
        features=7,
        rows=69680,
        frequency="15-min",
        columns=ETT_COLUMNS,
        source_url=f"{TSLIB_HF_BASE}/ETT-small/ETTm1.csv",
        sha256="6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
        description="Electricity Transformer Temperature M1 (15-minute granularity, 7 channels).",
    ),
    "ettm2": DatasetSpec(
        id="ettm2",
        label="ETTm2",
        group="TSLib",
        data_path="data/external/tslib/ETTm2.csv",
        target_column="OT",
        features=7,
        rows=69680,
        frequency="15-min",
        columns=ETT_COLUMNS,
        source_url=f"{TSLIB_HF_BASE}/ETT-small/ETTm2.csv",
        sha256="db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
        description="Electricity Transformer Temperature M2 (15-minute granularity, second transformer).",
    ),
    "weather": DatasetSpec(
        id="weather",
        label="Weather",
        group="TSLib",
        data_path="data/external/tslib/weather.csv",
        target_column="OT",
        features=21,
        rows=52696,
        frequency="10-min",
        columns=WEATHER_COLUMNS,
        source_url=f"{TSLIB_HF_BASE}/weather/weather.csv",
        sha256="34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63",
        description="Max Planck Biogeochemistry Institute meteorological station (21 indicators, 10-min).",
    ),
    "electricity": DatasetSpec(
        id="electricity",
        label="Electricity (ECL)",
        group="TSLib",
        data_path="data/external/tslib/electricity.csv",
        target_column="OT",
        features=321,
        rows=26304,
        frequency="hourly",
        columns=(*(str(i) for i in range(320)), "OT"),
        source_url=f"{TSLIB_HF_BASE}/electricity/electricity.csv",
        sha256="7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad",
        description="Hourly electricity consumption (kW) across 321 clients from 2016 to 2019.",
    ),
    "traffic": DatasetSpec(
        id="traffic",
        label="Traffic",
        group="TSLib",
        data_path="data/external/tslib/traffic.csv",
        target_column="OT",
        features=862,
        rows=17544,
        frequency="hourly",
        columns=(*(str(i) for i in range(861)), "OT"),
        source_url=f"{TSLIB_HF_BASE}/traffic/traffic.csv",
        sha256="cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16",
        description="Caltrans PeMS hourly freeway lane occupancy rates across 862 Bay Area sensors.",
    ),
    "exchange": DatasetSpec(
        id="exchange",
        label="Exchange Rate",
        group="TSLib",
        data_path="data/external/tslib/exchange_rate.csv",
        target_column="OT",
        features=8,
        rows=7588,
        frequency="daily",
        columns=EXCHANGE_COLUMNS,
        source_url=f"{TSLIB_HF_BASE}/exchange_rate/exchange_rate.csv",
        sha256="48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842",
        description="Daily foreign-exchange rates for 8 currencies from 1990 to 2016.",
    ),
    "ili": DatasetSpec(
        id="ili",
        label="ILI (Illness)",
        group="TSLib",
        data_path="data/external/tslib/national_illness.csv",
        target_column="OT",
        features=7,
        rows=966,
        frequency="weekly",
        columns=ILI_COLUMNS,
        source_url=f"{TSLIB_HF_BASE}/illness/national_illness.csv",
        sha256="93601f64d2566dc796ca4305adad8b8560c2db1a1ff04543c3bd813a7263570a",
        description="Weekly CDC Influenza-Like Illness patient ratios and age-bracket counts.",
    ),
    "nyc_taxi": DatasetSpec(
        id="nyc_taxi",
        label="NYC Taxi Trips",
        group="NYC TLC",
        data_path="data/external/nyc_taxi/nyc_taxi.csv",
        target_column="trip_count",
        features=8,
        rows=4368,
        frequency="hourly",
        columns=NYC_TAXI_COLUMNS,
        source_url=NYC_TLC_TRIP_BASE,
        sha256=None,
        description=(
            "Hourly NYC TLC Yellow Taxi trip records (2024 H1): borough pickup counts, "
            "total distance, mean duration, mean fare, and total trip_count."
        ),
    ),
}

_DATASET_ALIASES: dict[str, str] = {
    "paper": "copper",
    "ecl": "electricity",
    "exchange_rate": "exchange",
    "exchange-rate": "exchange",
    "illness": "ili",
    "national_illness": "ili",
    "national-illness": "ili",
    "nyc-taxi": "nyc_taxi",
    "nyctaxi": "nyc_taxi",
    "taxi": "nyc_taxi",
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Resolve a [`DatasetSpec`][hypercast4d.datasets.DatasetSpec] by ID or alias.

    Parameters
    ----------
    name : str
        Case-insensitive dataset identifier (such as ``"etth2"``, ``"weather"``,
        ``"ecl"``, ``"exchange_rate"``, ``"ili"``, or ``"nyc-taxi"``).

    Returns
    -------
    DatasetSpec
        Matching dataset specification.

    Raises
    ------
    ValueError
        If ``name`` does not match any registered dataset or alias.
    """
    key = _DATASET_ALIASES.get(name.strip().lower(), name.strip().lower())
    if key not in DATASETS:
        choices = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {choices}")
    return DATASETS[key]


def find_dataset_by_path(data_path: str | Path) -> DatasetSpec | None:
    """Match a project-relative ``data_path`` to a registered :class:`DatasetSpec`."""
    normalized = Path(data_path).as_posix()
    for spec in DATASETS.values():
        if normalized == spec.data_path or normalized in spec.fallback_paths:
            return spec
    return None


def resolve_dataset_path(data_path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve ``data_path`` (supporting registered fallback locations when needed).

    Parameters
    ----------
    data_path : str or Path
        Project-relative or absolute dataset path.
    project_root : Path or None, default=None
        Base directory for relative paths (defaults to ``Path.cwd()``).

    Returns
    -------
    Path
        Resolved path on disk (or canonical target path if not yet downloaded).
    """
    path = Path(data_path)
    if path.is_absolute():
        return path
    root = (project_root or Path.cwd()).resolve()
    candidate = root / path
    if candidate.is_file():
        return candidate
    spec = find_dataset_by_path(path)
    if spec is not None:
        for alt in (spec.data_path, *spec.fallback_paths):
            alt_path = root / alt
            if alt_path.is_file():
                return alt_path
    return candidate


def dataset_feature_count(
    data_path: str | Path | None,
    project_root: Path | None = None,
    fallback: int = 4,
) -> int:
    """Determine the numeric channel count of ``data_path`` from registry or file header.

    Parameters
    ----------
    data_path : str, Path, or None
        Dataset path to inspect; returns ``fallback`` when ``None``.
    project_root : Path or None, default=None
        Repository root resolving relative ``data/`` paths.
    fallback : int, default=4
        Default feature count returned when ``data_path`` is ``None`` or missing.

    Returns
    -------
    int
        Number of numeric time-series channels in the dataset.
    """
    if not data_path:
        return fallback
    spec = find_dataset_by_path(data_path)
    if spec is not None:
        return spec.features
    resolved = resolve_dataset_path(data_path, project_root)
    if not resolved.is_file():
        return fallback
    from .data import load_dataset

    return int(load_dataset(resolved).shape[1])


def dataset_catalog(
    project_root: Path | None = None, *, check_disk: bool = True
) -> list[dict[str, Any]]:
    """Return serializable metadata and local availability for all 11 datasets.

    Parameters
    ----------
    project_root : Path or None, default=None
        Repository root checked for downloaded files (defaults to ``Path.cwd()``).
    check_disk : bool, default=True
        Inspect local files on disk to set ``available`` and resolve fallback
        paths; set ``False`` when generating deterministic static manifests.

    Returns
    -------
    list[dict[str, Any]]
        Ordered dataset descriptors consumed by the Playground UI and CLI.
    """
    root = (project_root or Path.cwd()).resolve()
    entries: list[dict[str, Any]] = []
    for spec in DATASETS.values():
        resolved = resolve_dataset_path(spec.data_path, root) if check_disk else root / spec.data_path
        available = bool(check_disk and resolved.is_file())
        effective_path = (
            resolved.relative_to(root).as_posix()
            if available and resolved.is_relative_to(root)
            else spec.data_path
        )
        entries.append(
            {
                "id": spec.id,
                "label": spec.label,
                "group": spec.group,
                "data_path": effective_path,
                "target_column": spec.target_column,
                "features": spec.features,
                "rows": spec.rows,
                "frequency": spec.frequency,
                "columns": list(spec.columns),
                "available": available,
                "download_command": f"hypercast4d-download --dataset {spec.id}",
                "description": spec.description,
            }
        )
    return entries


def adapt_spec_features(spec: dict[str, Any], features: int) -> dict[str, Any]:
    """Adapt an unlocked architecture's default 4-channel ``feature_order`` to ``features``.

    Locked presets (such as the 4D Figure 3 paper models) and explicit
    single-channel target-only specs (``feature_order == [0]``) are preserved
    unchanged. Unlocked specs using the default ``[0, 1, 2, 3]`` placeholder
    switch to ``"all"`` when ``features != 4`` (and back to ``[0, 1, 2, 3]`` on
    4-channel datasets).
    """
    import copy

    if spec.get("locked"):
        return spec
    updated = copy.deepcopy(spec)
    desired: Any = [0, 1, 2, 3] if features == 4 else "all"
    if updated.get("schema_version") == 2:
        for source in updated.get("sources", {}).values():
            inp = source.get("input", {})
            if inp.get("feature_order") in ([0, 1, 2, 3], "all"):
                inp["feature_order"] = copy.deepcopy(desired)
    else:
        inp = updated.setdefault("input", {})
        if inp.get("feature_order", [0, 1, 2, 3]) in ([0, 1, 2, 3], "all"):
            inp["feature_order"] = copy.deepcopy(desired)
    return updated


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
    allow_updated_source: bool = False,
) -> str:
    """Download ``url`` atomically to ``destination`` and verify optional SHA-256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.is_file():
        temporary = destination.with_name(f".{destination.name}.tmp")
        request = urllib.request.Request(url, headers={"User-Agent": "Hypercast"})
        try:
            with (
                urllib.request.urlopen(request, timeout=300) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    observed = _sha256(destination)
    if (
        expected_sha256 is not None
        and observed != expected_sha256
        and not allow_updated_source
    ):
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: expected {expected_sha256}, "
            f"received {observed}. Pass --allow-updated-source only after inspecting."
        )
    return observed


def build_nyc_taxi_dataset(
    project_root: Path,
    *,
    months: tuple[str, ...] = tuple(NYC_TLC_PARQUET_SHA256),
    force: bool = False,
    allow_updated_source: bool = False,
) -> dict[str, Any]:
    """Download official NYC TLC Yellow Taxi parquet files and aggregate hourly channels.

    Constructs a leak-free, regular hourly 8-channel time series across ``months``
    (defaulting to 2024-01 through 2024-06, 4,368 hours) containing four spatial
    borough pickup series (``manhattan_pickups``, ``queens_pickups``,
    ``brooklyn_pickups``, ``other_pickups``), three trip-economics series
    (``total_distance``, ``mean_duration_min``, ``mean_fare``), and total hourly
    ``trip_count``.

    Parameters
    ----------
    project_root : Path
        Repository root directory under which ``data/external/nyc_taxi/`` is written.
    months : tuple[str, ...], default=("2024-01", ..., "2024-06")
        Ordered ``YYYY-MM`` months to download from NYC TLC CloudFront.
    force : bool, default=False
        Re-download raw parquet files and rebuild ``nyc_taxi.csv`` even if present.
    allow_updated_source : bool, default=False
        Accept upstream parquet files whose SHA-256 differs from pinned hashes.

    Returns
    -------
    dict[str, Any]
        Provenance dictionary recorded in ``data/external/nyc_taxi/checksums.json``.
    """
    if not months:
        raise ValueError("At least one YYYY-MM month is required for nyc_taxi")
    out_dir = project_root / "data" / "external" / "nyc_taxi"
    raw_dir = out_dir / "raw"
    out_csv = out_dir / "nyc_taxi.csv"
    manifest_path = out_dir / "checksums.json"
    if not force and out_csv.is_file() and manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    lookup_path = raw_dir / "taxi_zone_lookup.csv"
    lookup_sha = _download_file(
        NYC_TLC_ZONE_URL,
        lookup_path,
        expected_sha256=NYC_TLC_ZONE_SHA256,
        force=force,
        allow_updated_source=allow_updated_source,
    )
    zones = pd.read_csv(lookup_path)
    borough_map = dict(zip(zones["LocationID"], zones["Borough"].fillna("Other")))

    ordered_months = sorted(months)
    start_ts = pd.Timestamp(f"{ordered_months[0]}-01 00:00:00")
    last_period = pd.Period(ordered_months[-1], freq="M")
    end_ts = last_period.end_time.floor("1h")
    end_bound = last_period.end_time
    full_index = pd.date_range(start_ts, end_ts, freq="1h", name="date")

    parquet_hashes: dict[str, str] = {}
    parts: list[pd.DataFrame] = []
    for month in ordered_months:
        filename = f"yellow_tripdata_{month}.parquet"
        parquet_path = raw_dir / filename
        parquet_hashes[filename] = _download_file(
            f"{NYC_TLC_TRIP_BASE}/{filename}",
            parquet_path,
            expected_sha256=NYC_TLC_PARQUET_SHA256.get(month),
            force=force,
            allow_updated_source=allow_updated_source,
        )
        frame = pd.read_parquet(
            parquet_path,
            columns=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "PULocationID",
                "trip_distance",
                "fare_amount",
            ],
        )
        pickup = pd.to_datetime(frame["tpep_pickup_datetime"])
        dropoff = pd.to_datetime(frame["tpep_dropoff_datetime"])
        # NYC TLC raw files occasionally include out-of-range meter timestamps.
        valid = (pickup >= start_ts) & (pickup <= end_bound)
        frame = frame.loc[valid]
        pickup = pickup.loc[valid]
        dropoff = dropoff.loc[valid]

        hour = pickup.dt.floor("1h")
        borough = frame["PULocationID"].map(borough_map).fillna("Other")
        duration_min = (
            ((dropoff - pickup).dt.total_seconds() / 60.0)
            .clip(lower=0.0, upper=180.0)
            .fillna(0.0)
        )
        distance = frame["trip_distance"].clip(lower=0.0, upper=200.0).fillna(0.0)
        fare = frame["fare_amount"].clip(lower=0.0, upper=1000.0).fillna(0.0)

        grouped = (
            pd.DataFrame(
                {
                    "date": hour,
                    "manhattan_pickups": (borough == "Manhattan").astype(np.int64),
                    "queens_pickups": (borough == "Queens").astype(np.int64),
                    "brooklyn_pickups": (borough == "Brooklyn").astype(np.int64),
                    "other_pickups": (
                        ~borough.isin(["Manhattan", "Queens", "Brooklyn"])
                    ).astype(np.int64),
                    "total_distance": distance,
                    "duration_sum": duration_min,
                    "fare_sum": fare,
                    "trip_count": 1,
                }
            )
            .groupby("date", as_index=True)
            .sum()
        )
        parts.append(grouped)

    combined = pd.concat(parts).groupby(level=0).sum().reindex(full_index, fill_value=0.0)
    denom = np.maximum(combined["trip_count"].to_numpy(dtype=np.float64), 1.0)
    hourly = pd.DataFrame(
        {
            "manhattan_pickups": combined["manhattan_pickups"].astype(np.float64),
            "queens_pickups": combined["queens_pickups"].astype(np.float64),
            "brooklyn_pickups": combined["brooklyn_pickups"].astype(np.float64),
            "other_pickups": combined["other_pickups"].astype(np.float64),
            "total_distance": combined["total_distance"].round(4),
            "mean_duration_min": (combined["duration_sum"] / denom).round(4),
            "mean_fare": (combined["fare_sum"] / denom).round(4),
            "trip_count": combined["trip_count"].astype(np.float64),
        },
        index=full_index,
    ).reset_index()

    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_csv.with_name(".nyc_taxi.csv.tmp")
    hourly.to_csv(temporary, index=False)
    temporary.replace(out_csv)

    manifest = {
        "dataset": "nyc_taxi",
        "months": ordered_months,
        "rows": len(hourly),
        "features": int(hourly.shape[1] - 1),
        "target_column": "trip_count",
        "taxi_zone_lookup.csv": lookup_sha,
        "parquet_sha256": parquet_hashes,
        "nyc_taxi.csv": _sha256(out_csv),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def download_dataset(
    name: str,
    project_root: Path | None = None,
    *,
    force: bool = False,
    allow_updated_source: bool = False,
    nyc_taxi_months: tuple[str, ...] = tuple(NYC_TLC_PARQUET_SHA256),
) -> dict[str, Any]:
    """Download and verify one registered dataset into ``project_root / data/``.

    Parameters
    ----------
    name : str
        Registered dataset ID or alias (for example ``"etth2"``, ``"weather"``,
        ``"nyc_taxi"``, or ``"copper"``).
    project_root : Path or None, default=None
        Target repository root (defaults to ``Path.cwd()``).
    force : bool, default=False
        Re-download upstream files even if already cached locally.
    allow_updated_source : bool, default=False
        Permit changed upstream checksums.
    nyc_taxi_months : tuple[str, ...], default=("2024-01", ..., "2024-06")
        Months aggregated when ``name == "nyc_taxi"``.

    Returns
    -------
    dict[str, Any]
        Verification manifest including ``id``, ``data_path``, and ``sha256``.
    """
    root = (project_root or Path.cwd()).resolve()
    spec = get_dataset_spec(name)
    if spec.id == "copper":
        from .download import download as download_paper

        checksums = download_paper(
            root / "data" / "raw",
            force=force,
            allow_updated_source=allow_updated_source,
        )
        return {
            "id": "copper",
            "data_path": spec.data_path,
            "sha256": checksums["paper_data.xlsx"],
        }
    if spec.id == "nyc_taxi":
        manifest = build_nyc_taxi_dataset(
            root,
            months=nyc_taxi_months,
            force=force,
            allow_updated_source=allow_updated_source,
        )
        return {
            "id": "nyc_taxi",
            "data_path": spec.data_path,
            "sha256": manifest["nyc_taxi.csv"],
            "rows": manifest["rows"],
        }

    destination = root / spec.data_path
    if not force and not destination.is_file():
        # Reuse existing verified fallback copy if present (e.g., ETTh1).
        for alt in spec.fallback_paths:
            alt_path = root / alt
            if alt_path.is_file() and (
                spec.sha256 is None or _sha256(alt_path) == spec.sha256
            ):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(alt_path, destination)
                break

    observed = _download_file(
        spec.source_url,
        destination,
        expected_sha256=spec.sha256,
        force=force,
        allow_updated_source=allow_updated_source,
    )
    checksums_path = destination.parent / "checksums.json"
    existing = (
        json.loads(checksums_path.read_text(encoding="utf-8"))
        if checksums_path.is_file()
        else {"commit": TSLIB_HF_COMMIT, "files": {}}
    )
    existing.setdefault("files", {})[destination.name] = observed
    checksums_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return {"id": spec.id, "data_path": spec.data_path, "sha256": observed}
