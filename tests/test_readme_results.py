"""The numbers in the READMEs' tables must come from `results/results.json`, written by
`main.py`. If the pipeline changes and a README is not updated, this fails."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "results.json"
READMES = [("en", "README.md"), ("es", "README.es.md")]
TYPES = ["overpricing", "inflated_quantity", "digit_error", "new_supplier", "wrong_product"]
DETECTORS = {
    "en": {"Isolation Forest": "Isolation Forest", "Autoencoder (GELU)": "Autoencoder (GELU)",
           "Autoencoder (ReLU)": "Autoencoder (ReLU)", "Autoencoder (Swish (SiLU))": "Autoencoder (Swish (SiLU))",
           "Rule (z-scores)": "Rule (z-scores)"},
    "es": {"Isolation Forest": "Isolation Forest", "Autoencoder (GELU)": "Autoencoder (GELU)",
           "Autoencoder (ReLU)": "Autoencoder (ReLU)", "Autoencoder (Swish (SiLU))": "Autoencoder (Swish (SiLU))",
           "Rule (z-scores)": "Regla (z-scores)"},
}
BUNCHING = {
    "en": {"30 UTM, 2024 (cap)": ("cap30", True), "100 UTM, 2025-26 (cap)": ("cap100", True),
           "30 UTM, 2025-26 (no longer a cap)": (None, False)},
    "es": {"30 UTM, 2024 (tope)": ("cap30", True), "100 UTM, 2025-26 (tope)": ("cap100", True),
           "30 UTM, 2025-26 (ya no es tope)": (None, False)},
}


def _pct(x: float, lang: str) -> str:
    s = f"{x * 100:.1f}%"
    return s.replace(".", ",") if lang == "es" else s


def _int(n: float, lang: str) -> str:
    s = f"{round(n):,}"
    return s.replace(",", ".") if lang == "es" else s


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("results/results.json not found: run `python main.py` first")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("lang, readme", READMES)
def test_detector_table_matches_results(results, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    for key, label in DETECTORS[lang].items():
        r = results["detectors"]["recall_at_5pct"][key]
        row = f"| {label} | " + " | ".join(_pct(r[k], lang) for k in ["overall"] + TYPES) + " |"
        assert row in text, f"{readme}: expected {row}"


@pytest.mark.parametrize("lang, readme", READMES)
def test_bunching_table_matches_results(results, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    to = "to" if lang == "en" else "a"
    for label, (regime, extra) in BUNCHING[lang].items():
        b = results["compra_agil"][regime]["bunching"] if regime else results["old_cap_30_after_change"]
        row = f"| {label} | {_pct(b['excess_mass'], lang)} | {_pct(b['ci95'][0], lang)} {to} {_pct(b['ci95'][1], lang)} |"
        if extra:
            row += f" {_int(b['observed_in_window'] - b['expected_in_window'], lang)} |"
        assert row in text, f"{readme}: expected {row}"


@pytest.mark.parametrize("lang, readme", READMES)
def test_data_table_matches_results(results, lang, readme):
    text = (ROOT / readme).read_text(encoding="utf-8")
    for key in ("order_lines", "orders", "buyer_units", "suppliers"):
        assert f"| {_int(results['data'][key], lang)} |" in text, f"{readme}: {key}"
    for regime in ("cap30", "cap100"):
        assert f"| {_int(results['compra_agil'][regime]['orders'], lang)} |" in text, f"{readme}: {regime}"
