"""Turns each monthly ChileCompra ZIP into a compact Parquet file of order lines.

The CSVs are not in a single encoding. Almost everything is Windows-1252, but some fields
arrive as UTF-8 (1,828 sequences in January 2025 alone, e.g. "ISOFÁNICA"), so decoding the
whole file as either one breaks the other. Here every byte sequence that is valid UTF-8 is
read as UTF-8 and every other byte as Windows-1252.

Buyers and suppliers are kept as codes only, never names: the analysis does not need them
and everything published from it is aggregated.
"""

from __future__ import annotations

import codecs
import io
import zipfile
from pathlib import Path

import polars as pl

from src.sources import MONTHS, RAW_DIR, oc_path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

COLUMNS = {
    "Codigo": "order_code",
    "Tipo": "order_type",
    "EsTratoDirecto": "is_direct",
    "Estado": "status",
    "FechaCreacion": "created_date",
    "FechaEnvio": "sent_date",
    "TipoMonedaOC": "currency",
    "MontoTotalOC_PesosChilenos": "total_clp",
    "TotalNetoOC": "net_total",
    "CodigoUnidadCompra": "buyer_unit",
    "CodigoOrganismoPublico": "buyer_org",
    "sector": "sector",
    "RegionUnidadCompra": "buyer_region",
    "CodigoProveedor": "supplier",
    "IDItem": "line_id",
    "codigoCategoria": "category_code",
    "codigoProductoONU": "product_code",
    "RubroN1": "rubro",
    "cantidad": "quantity",
    "UnidadMedida": "unit",
    "monedaItem": "line_currency",
    "precioNeto": "unit_price",
    "totalLineaNeto": "line_total",
}
NUMERIC = ["total_clp", "net_total", "quantity", "unit_price", "line_total"]


def _cp1252_fallback(error: UnicodeDecodeError) -> tuple[str, int]:
    bad = error.object[error.start : error.end]
    text = bad.decode("cp1252", errors="ignore")
    return (text or bad.decode("latin-1")), error.end


codecs.register_error("cp1252_fallback", _cp1252_fallback)


def decode_mixed(raw: bytes) -> str:
    """UTF-8 where the bytes are valid UTF-8, Windows-1252 everywhere else."""
    return raw.decode("utf-8", errors="cp1252_fallback")


def _to_number(column: str) -> pl.Expr:
    # Decimals come with a comma in some fields and a point in others; thousands are never grouped.
    return pl.col(column).str.replace(",", ".", literal=True).cast(pl.Float64, strict=False)


def parse_month(raw_csv: bytes) -> pl.DataFrame:
    text = decode_mixed(raw_csv)
    df = pl.read_csv(
        io.BytesIO(text.encode("utf-8")),
        separator=";",
        quote_char='"',
        columns=list(COLUMNS),
        infer_schema=False,
        null_values=["NA", ""],
    ).rename(COLUMNS)
    return df.with_columns(
        [_to_number(c) for c in NUMERIC]
        + [
            pl.col("created_date").str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
            pl.col("sent_date").str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False),
            (pl.col("is_direct") == "Si").alias("is_direct"),
        ]
    )


def processed_path(year: int, month: int, processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / f"lines_{year}-{month:02d}.parquet"


def ingest_month(year: int, month: int, raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> Path:
    out = processed_path(year, month, processed_dir)
    if out.exists():
        return out
    with zipfile.ZipFile(oc_path(year, month, raw_dir)) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"{year}-{month}: expected one CSV in the ZIP, found {members}")
        raw = archive.read(members[0])
    df = parse_month(raw)
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out


def ingest_all(months: list[tuple[int, int]] = MONTHS) -> list[Path]:
    paths = []
    for year, month in months:
        path = ingest_month(year, month)
        print(f"ingested {path.name}", flush=True)
        paths.append(path)
    return paths
