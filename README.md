**[English](README.md) | [Español](README.es.md)**

# Public Procurement Anomaly Engine (Chile)

[![CI](https://github.com/Rxyxs/chile-public-procurement-anomalies/actions/workflows/ci.yml/badge.svg)](https://github.com/Rxyxs/chile-public-procurement-anomalies/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue) ![Data](https://img.shields.io/badge/data-real%20(ChileCompra)-2ea44f) ![License](https://img.shields.io/badge/license-MIT-green)

On 13.7 million real purchase-order lines from Mercado Público, Compra Ágil orders pile up just under the legal cap, twice the trend under the old 30 UTM cap and 71% above it under the new 100 UTM one, and the pile moved when Ley 21.634 moved the cap; but within the same buyer, orders at the cap come with a same-supplier order within a week only 2 points more often, so most of the pile looks like purchases sized to the limit rather than split under it.

## What I found

| Finding | Evidence |
|---|---|
| **The pile follows the cap** | Under the 30 UTM cap (January-November 2024) the last 10% below it holds 103% more Compra Ágil orders than the trend predicts (95% interval 100% to 107%), about 34,200 extra orders. Under the 100 UTM cap (January 2025-September 2026) it holds 71% more (68% to 75%), about 18,300. At 30 UTM in 2025-26, once it was no longer a cap, the excess drops to 18%, in line with six amounts that were never caps (-6% to +30%). |
| **Splitting is a weak signal, not the main story** | Orders at 95-100% of the cap have another order of the same buyer and supplier within 7 days more often than orders at 60-90% (43% against 32% in 2024). Comparing orders of the same buyer, the gap shrinks to 2.6 points (1.7 to 3.6) in 2024 and 1.7 points (0.6 to 2.8) in 2025-26; same-day pairs are not significant after the change (p = 0.53). |
| **There is still a worklist for an auditor** | Since 2025, 20,020 chains of orders from one buyer to one supplier, each under the cap but adding up to more than 100 UTM within a week of each other, hold 15.7% of all the Compra Ágil amount. Not proof of anything, but where a review should start. |
| **On real lines, Isolation Forest is the detector to use** | Reviewing the top 5% of 200,000 real 2026 lines with 10,000 planted anomalies, Isolation Forest finds 38.8% of them (7.8 times a random review), the autoencoder 22.4% to 22.9% depending on the activation, and the z-score rule 17.4%. On the simulated invoices of the first version the rule won; real data reversed the ranking. |
| **Overpricing hides in the noise** | A price 3 to 8 times too high is found in at most 10.6% of the cases by any detector: the ONU product codes are broad ("medical exams", "stationery"), so real prices already vary that much. A thousandfold typo is found 70.2% of the time, and so is a high bill from a supplier with no history. |

## The data

Every purchase order sent through [Mercado Público](https://www.mercadopublico.cl) is published as open data by ChileCompra, one ZIP per month with one row per order line. The pipeline downloads 32 months (`python main.py --download`): January to November 2024, when the Compra Ágil cap was 30 UTM, and January 2025 to September 2026, after [Ley 21.634](https://www.chilecompra.cl/ley-de-compras-publicas/) raised it to 100 UTM on 12 December 2024. December 2024 mixes both rules and is left out.

| | |
|---|---:|
| Order lines | 13,742,385 |
| Purchase orders | 5,006,455 |
| Buying units | 6,217 |
| Suppliers | 122,356 |
| Compra Ágil orders, 2024 (cap 30 UTM) | 686,395 |
| Compra Ágil orders, 2025-26 (cap 100 UTM) | 1,278,781 |

Things in the files that had to be handled before any analysis:

- **Two encodings in the same file.** The CSVs are Windows-1252, except for fields that arrive in UTF-8 (1,828 sequences in January 2025 alone, such as "ISOFÁNICA"). Reading the whole file as either one corrupts the other; every byte sequence that is valid UTF-8 is read as UTF-8 and the rest as Windows-1252.
- **The cap that applies is the one in force when the order was created**, not when it was sent: orders created under the old rule keep appearing in 2025 files. Amounts are converted with the UTM of the creation month.
- **The cap binds on the total including taxes.** No Compra Ágil order goes above 100 UTM gross, while net amounts stop near 84 UTM (100 / 1.19). The 26 and 12 orders above the cap in each period, and five orders above one trillion pesos, are data-entry errors.
- **Totals are computed by the platform**: only 0.06% of lines have a total that does not match quantity times price, so the "total that does not reconcile" anomaly of the first, simulated version does not exist in real data.
- **Buyers and suppliers are kept as codes.** No name of any buyer or supplier appears in the results: everything published here is aggregated.

## 1. The pile under the cap

![Compra Ágil orders by amount](results/figures/bunching_compra_agil.png)

Compra Ágil orders by total amount: under either rule, the density is flat between 60% and 90% of the cap and climbs sharply in the last 10% below it, which is where the excess is measured against a straight line fitted on the flat part.

![Excess mass at real and fake caps](results/figures/placebo_caps.png)

The same measure at amounts that never were a cap stays between -6% and +30%; under the real caps it is 103% and 71%, and at 30 UTM it fell to 18% as soon as the cap moved away from it.

| Where the excess is measured | Excess mass | 95% interval | Extra orders |
|---|---:|---:|---:|
| 30 UTM, 2024 (cap) | 103.0% | 99.5% to 107.1% | 34,223 |
| 100 UTM, 2025-26 (cap) | 71.5% | 68.0% to 75.2% | 18,319 |
| 30 UTM, 2025-26 (no longer a cap) | 18.1% | 15.5% to 20.6% | |

How sure is the size: the excess depends on the counterfactual. With a constant or a straight line, three fit ranges and three window widths (18 specifications), it goes from 52% to 139% under the 30 UTM cap and from 26% to 138% under the 100 UTM one. The direction is never in doubt; the exact percentage is. A fifth-degree polynomial, the textbook choice, swung from +65% to -747% with the degree and is not used.

## 2. Split purchases or purchases sized to the cap?

A pile under a cap fits two behaviours the histogram cannot separate: a buyer sizing a purchase to the maximum allowed, which is legal, and one purchase split in two to stay under it, which the law forbids. Splitting leaves a trace: another order of the same buyer to the same supplier a few days apart.

![Orders with a sibling by size](results/figures/sibling_rates.png)

The share of orders with a same-buyer, same-supplier order within 7 days rises near the cap under both rules, from about 21% to 34% after the change.

But buyers that buy near the cap are also buyers that repeat purchases often. Comparing orders *of the same buying unit* (a linear probability model with buyer fixed effects, errors clustered by buyer), most of the gap disappears:

| Period | With a sibling: at 95-100% of the cap | At 60-90% | Within the same buyer | 95% interval | Same-day sibling, within buyer |
|---|---:|---:|---:|---:|---:|
| 2024 (cap 30 UTM) | 42.8% | 31.9% | +2.6 points | +1.7 to +3.6 | +1.1 points (p = 0.025) |
| 2025-26 (cap 100 UTM) | 34.8% | 28.7% | +1.7 points | +0.6 to +2.8 | +0.3 points (p = 0.53) |

So the pile is mostly purchases sized to the limit. Splitting does appear in the data, as a gap of one or two points, not as the explanation of the pile. What an auditor can still use: since 2025, **20,020 chains** of orders from one buyer to one supplier, each under the cap and each within 7 days of the previous one, add up to more than 100 UTM. They are 7.0% of the Compra Ágil orders and 15.7% of the amount; in 2024 they were 16.0% of the orders and 24.5% of the amount. Recurring legitimate purchases (food, supplies) look the same, so these are a ranking for review, not findings.

## 3. Detectors on real order lines

Real orders have no labels, so the three detectors of the first version are compared on a real background. They are trained on 400,000 lines from 2025; then 10,000 anomalies of five kinds (2,000 each) are planted in 200,000 real lines from 2026, and each detector ranks all of them. The score is the share of planted lines in its top 5%, what a team reviewing one line in twenty would catch. Real anomalies already in the data count against the detectors, as they would in a real review.

The features compare each line with what its product usually costs. Product means ONU code and unit of measure, with median and MAD from 2025 (12,453 groups covering 92% of the 2026 lines): price, quantity and total as robust z-scores, the distance to what *the same buyer* paid before for the same product, the supplier's age in the data and the buyer-supplier history.

![Planted anomalies found by each detector](results/figures/detector_recall.png)

| Detector | All | Overpricing ×3-8 | Quantity ×5-10 | Thousandfold typo | New supplier, high bill | Price of another product |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 38.8% | 7.0% | 12.0% | 70.2% | 70.0% | 34.9% |
| Autoencoder (GELU) | 22.9% | 9.9% | 11.6% | 51.6% | 15.2% | 26.2% |
| Autoencoder (ReLU) | 22.4% | 10.1% | 14.5% | 46.2% | 16.7% | 24.6% |
| Autoencoder (Swish (SiLU)) | 22.4% | 10.6% | 12.3% | 45.2% | 16.0% | 27.8% |
| Rule (z-scores) | 17.4% | 5.9% | 23.5% | 38.6% | 4.5% | 14.5% |

A random review of 5% finds 5%. Isolation Forest wins because it uses the supplier's history, where planted new suppliers stand out, and because extreme combinations isolate quickly in a tree. The rule only looks at prices and quantities and is best at inflated quantities. The autoencoder sits in between with any activation; its best epoch was 148 of 150, so it was still improving slowly.

![Autoencoder training curves](results/figures/training_curves.png)

Reconstruction error on 2025 lines for the three activations; the validation curve tracks the training one, without overfitting.

The two changes that made the detectors usable on real data were robust statistics (median and MAD, because real prices have heavy tails and a mean and standard deviation are dragged by the outliers the detectors look for) and the comparison with the same buyer's past prices: without it, during development, Isolation Forest found about a third as many planted anomalies and missed most typos.

## What changed from the first version

The first version detected anomalies in 15,000 simulated invoices of a mining company, with anomalies injected by the same code that generated the data. It now runs on every public purchase order in Chile. The detectors are still there, scored the same way but on a real background, and the two new analyses (the cap and split purchases) answer questions that only real data can raise. The repository was renamed from `mining-procurement-anomaly-engine` to match; old links to it on GitHub redirect here.

## Technology stack

| Layer | Technology | Role |
|---|---|---|
| Data | **urllib**, **Polars**, **Parquet** | Download, mixed-encoding parsing, 13.7 M lines in 313 MB |
| Statistics | **NumPy**, **statsmodels** | Bunching estimator, fixed-effects models with clustered errors |
| Detection | **scikit-learn** (Isolation Forest), **PyTorch** (autoencoder) | Unsupervised detectors and the activation comparison |

## Getting started

```powershell
py -m venv venv
./venv/Scripts/pip install -r requirements.txt
./venv/Scripts/python main.py --download   # once: 32 monthly ZIPs (~3 GB, about 10 minutes) and the UTM
./venv/Scripts/python main.py              # about 20 minutes on a laptop CPU
```

It writes `results/results.json`, the source of every number in this README, and the figures in `results/figures/`.

### Tests

```powershell
./venv/Scripts/pytest -v
```

The tests run without network on small fixtures: the mixed-encoding decoder, parsing (comma decimals, line breaks inside fields, missing values), the cap in force by creation date, the bunching estimator on a flat density and on a planted pile, sibling flags and chains, the within-buyer model on data with a known effect, history features that only look backwards, planted anomalies, the autoencoder, and a check that every number in both READMEs' tables matches `results/results.json`.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
