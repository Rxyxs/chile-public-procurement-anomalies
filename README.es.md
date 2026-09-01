<h1 align="center">Mining Procurement Anomaly Engine</h1>

<p align="center">
  <b>Español</b> · <a href="README.md">English</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.13%2Bcpu-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Polars" src="https://img.shields.io/badge/Polars-1.44-CD792C?logo=polars&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
</p>

Sistema no supervisado de detección de anomalías para facturas de
proveedores mineros. Un autoencoder en PyTorch se entrena sobre features
tabulares de las facturas, y el 5% con mayor error de reconstrucción se
aísla como candidato a auditoría manual — sin necesidad de datos etiquetados
de fraude.

## Por qué este proyecto

Las operaciones mineras manejan gasto de compras recurrente y masivo a
través de decenas de categorías de proveedores (explosivos, neumáticos
CAEX, repuestos de chancado, combustible, servicios de mantención) con
escalas de precio radicalmente distintas. La auditoría manual de facturas no
escala, y las reglas fijas solo detectan los patrones de fraude que alguien
ya pensó en codificar. Un modelo no supervisado que aprende "cómo se ve una
factura normal" y marca lo que no logra reconstruir bien le da a un equipo
de auditoría una lista priorizada sin necesitar historial de fraude
etiquetado — algo que, en general, los departamentos de compras mineras en
Chile no tienen.

## Impacto de Negocio e Indicadores Clave (KPIs)

| Métrica | Resultado | Qué significa |
|---|---|---|
| Recall total a presupuesto fijo de auditoría del 5% | 37,3% (280/750 anomalías inyectadas) | **~7,5x** mejor que el ~5% de recall que obtendría una muestra aleatoria del 5% por azar |
| Tipo de fraude mejor detectado | Sobreprecio (3-8x), 0,54 de recall | La feature de reconciliación (`monto_ratio_log`) es lo que hace detectable este y otros patrones |
| Tipo de fraude más difícil, reportado honestamente | Inflación de cantidad, 0,06 de recall | Causa raíz: la desviación estándar por categoría del propio set de entrenamiento está contaminada por el fraude que se supone debe detectar -- un trade-off documentado, no escondido |
| Bug real corregido: escalado ciego a la categoría | ~16% de recall → 37,3% tras z-scores relativos a categoría | Un `StandardScaler` global dejaba que la varianza de precio entre categorías ahogara las anomalías dentro de cada categoría |

## Cómo funciona

```mermaid
flowchart LR
    A[Facturas de compra sinteticas<br/>Polars, 15.000 filas] --> B["Ingenieria de features<br/>z-score relativo a categoria + monto_ratio_log"]
    B --> C["Autoencoder PyTorch<br/>6 -> 16 -> 8 -> 4 -> 8 -> 16 -> 6"]
    C --> D[Top 5% por error de reconstruccion<br/>marcado para auditoria manual]
```

1. **Datos sintéticos** (`generate_procurement_data`, Polars): 15.000
   facturas repartidas en 8 categorías de compra, 6 regiones mineras y 180
   proveedores, con distribuciones log-normales de precio/cantidad
   específicas por categoría. No existe un dataset público de facturación de
   proveedores mineros chilenos, así que el generador modela escalas de
   precio realistas (combustible ~$850 CLP/litro vs. neumáticos CAEX ~$8,5M
   CLP/unidad) en vez de inventar números arbitrarios.
2. **Inyección de anomalías** (5% de las filas): cinco patrones de fraude
   distintos — sobreprecio (3-8x), cantidad inflada (5-10x), un monto total
   que no reconcilia con cantidad × precio unitario (1,4-2,5x), un precio
   unitario tomado de la distribución de otra categoría, y un proveedor
   recién registrado que factura un monto inusualmente alto. La etiqueta se
   guarda **solo** para validar el modelo después — nunca se usa durante el
   entrenamiento.
3. **Ingeniería de features**: precio y cantidad crudos son log-normales y
   varían en varios órdenes de magnitud *entre* categorías, así que se
   expresan como z-score *relativo a su propia categoría* (ajustado solo
   sobre el split de train, para evitar leakage) en vez de usar el valor
   crudo. Una feature `monto_ratio_log` expone directamente si el monto
   declarado reconcilia con cantidad × precio unitario — esta única feature
   diseñada es lo que hace detectable el fraude de reconciliación y de
   proveedor nuevo (ver Resultados).
4. **Autoencoder** (PyTorch, CPU): 6 → 16 → 8 → 4 → 8 → 16 → 6, entrenado con
   Adam + pérdida MSE y early stopping sobre un split de validación.
5. **Detección**: se calcula el error de reconstrucción para cada factura;
   el umbral del percentil 95 aísla el 5% superior como anómalo.

## Resultados

De una corrida real (semilla 42, 15.000 facturas, 750 anomalías
inyectadas):

La versión animada de abajo traza las mismas curvas de pérdida train/val epoch a epoch, con una etiqueta que se actualiza en la punta de cada línea.

<p align="center">
  <img src="results/training_curve_animated.gif" width="48%" alt="Curva de entrenamiento animada">
  <img src="results/training_curve.png" width="48%" alt="Curva de entrenamiento">
</p>

<p align="center">
  <img src="results/reconstruction_error_hist.png" width="48%" alt="Distribucion del error de reconstruccion">
</p>

- El entrenamiento convergió sin problemas en 150 epochs (mejor epoch 146),
  con pérdida de train/val muy cercana y sin overfitting.
- **Total: 280/750 anomalías inyectadas capturadas en el 5% superior por
  error de reconstrucción (37,3% de recall / 37,3% de precisión** —
  precisión y recall coinciden porque el conjunto marcado tiene exactamente
  el mismo tamaño que la tasa real de anomalías, 5%).
- Eso es ~7,5x mejor que el ~5% de recall que obtendría una muestra
  aleatoria del 5% por puro azar.

**Recall por tipo de anomalía inyectada** (este desglose es la parte honesta
del resultado — no todos los patrones de fraude son igual de separables con
un presupuesto fijo del 5%):

| Tipo de anomalía | Recall | Detectadas / inyectadas |
|---|---|---|
| Sobreprecio (3-8x) | 0,54 | 83/155 |
| Proveedor nuevo + monto inflado | 0,51 | 77/152 |
| Monto no reconcilia con el detalle | 0,45 | 69/153 |
| Mismatch categoría/precio | 0,30 | 43/145 |
| Cantidad inflada (5-10x) | 0,06 | 8/145 |

**Hallazgo honesto**: la cantidad inflada es, estructuralmente, el patrón
más difícil de capturar aquí. La desviación estándar por categoría de
`cantidad_zscore_categoria` se estima sobre un set de entrenamiento no
supervisado que ya contiene ~1% de este mismo tipo de fraude — la
estimación queda contaminada por los mismos outliers que se quiere
detectar, lo que ensancha el rango "normal" y atenúa la señal. Cambiar a un
estimador robusto de mediana/MAD arregla la cantidad inflada (recall 0,06 →
0,17) pero **baja** el recall total (37,3% → 33,6%), porque redistribuye
qué tipo de anomalía se queda con los cupos fijos del top-5% — es un
trade-off real, no un bug, documentado en el docstring de
`compute_category_stats()` en [autoencoder.py](autoencoder.py). Se mantiene
la versión media/std como default porque tiene mayor recall total.

Se encontraron y corrigieron dos bugs reales durante la construcción,
ambos corriendo el pipeline e inspeccionando números reales en vez de
confiar en el diseño:
1. Alimentar al autoencoder con precio/cantidad/monto crudos daba solo ~16%
   de recall — un `StandardScaler` global dejaba que la varianza *entre*
   categorías (diferencias de precio de varios órdenes de magnitud) ahogara
   las anomalías *dentro* de una categoría. Se corrigió con z-scores
   relativos a la categoría.
2. Una anomalía de mismatch categoría/precio puede producir un z-score
   crudo de decenas de desviaciones estándar (un precio de neumático
   evaluado contra la distribución de combustible), y una categoría de
   proveedor (`Servicios Mantención`) tiene más del 50% de sus facturas con
   cantidad=1, dejando su desviación absoluta mediana en exactamente cero —
   ambos casos producían inestabilidad/`NaN` y se corrigieron con
   winsorizing y un piso mínimo de MAD, respectivamente.

## Comparación de modelos: 3 enfoques complementarios

`models_comparison.py` evalúa tres enfoques de detección complementarios
sobre exactamente las mismas features/split que el autoencoder de arriba,
más una comparación de funciones de activación del propio autoencoder
(ReLU vs. GELU vs. Swish/SiLU, misma arquitectura y semilla para las tres).
Los tres enfoques se evalúan igual: aislar el 5% con mayor score de
anomalía y medir precision/recall contra las etiquetas inyectadas.

| Enfoque | Precision @5% | Recall @5% | TP / inyectadas |
|---|---|---|---|
| Baseline (regla de z-score combinado, sin entrenamiento) | 0,539 | 0,539 | 404/750 |
| Isolation Forest (300 árboles) | 0,531 | 0,531 | 398/750 |
| Autoencoder — ReLU | 0,373 | 0,373 | 280/750 |
| Autoencoder — GELU | 0,305 | 0,305 | 229/750 |
| Autoencoder — Swish (SiLU) | 0,228 | 0,228 | 171/750 |

La versión animada de abajo corre la curva de val loss de cada función de activación epoch a epoch, con una etiqueta que se actualiza por línea.

<p align="center">
  <img src="results/model_comparison.png" width="48%" alt="Comparacion de modelos">
  <img src="results/activation_comparison_animated.gif" width="48%" alt="Comparacion de funciones de activacion animada">
</p>

<p align="center">
  <img src="results/activation_comparison.png" width="48%" alt="Comparacion de funciones de activacion">
</p>

**Hallazgo honesto**: en este dataset sintético en particular, tanto el
baseline interpretable como Isolation Forest **superan** al autoencoder en
recall. La regla de z-score suma directamente las tres señales
manufacturadas (`precio_zscore_categoria`, `cantidad_zscore_categoria`,
`monto_ratio_log`) que la inyección de fraude manipula, así que no tiene
que aprender ninguna representación — es la señal misma. El autoencoder
tiene que aprender esa representación solo a partir del error de
reconstrucción, y lo paga en recall en un dataset de este tamaño (15.000
filas, 6 features). Es una ilustración realista de por qué los baselines
interpretables van en la comparación y no solo como formalidad: la
complejidad extra del modelo no es gratis, y acá no se está pagando sola.
Entre las tres activaciones, ReLU converge al mejor recall de detección
pese a que GELU/Swish alcanzan un MSE de validación *más bajo* —
activaciones más suaves reconstruyen mejor el grueso de facturas normales
pero también reconstruyen parcialmente las anomalías, justo lo contrario
de lo que necesita un umbral de top-5%-por-error.

Las métricas y predicciones por factura de los cinco enfoques se persisten
en `results/metrics.duckdb` (tablas `approach_metrics`, `predictions`) para
poder consultarlas directamente con SQL, por ejemplo:

```sql
SELECT approach, precision, recall FROM approach_metrics ORDER BY recall DESC;
```

## Estructura del proyecto

```
mining-procurement-anomaly-engine/
├── autoencoder.py          # generacion de datos, features, entrenamiento/evaluacion del autoencoder
├── models_comparison.py    # regla baseline z-score + Isolation Forest + comparacion de activaciones, persistencia DuckDB
├── tests/
│   └── test_models_comparison.py  # tests unitarios pytest para los 3 enfoques
├── requirements.txt
├── data/                  # dataset generado (gitignored, se regenera al correr el script)
├── models/                # checkpoint del modelo entrenado (gitignored)
└── results/               # CSV de anomalias + metrics.duckdb (gitignored) + graficos versionados
```

## Cómo correrlo

```powershell
py -3.10 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe autoencoder.py
.\venv\Scripts\python.exe models_comparison.py
```

`autoencoder.py` genera `data/procurement_invoices.csv` (dataset sintético
completo), `models/autoencoder.pt` (pesos entrenados), y tres archivos en
`results/`: `anomalias_detectadas.csv` (las facturas marcadas, ordenadas),
más la curva de entrenamiento y el histograma de error de reconstrucción
mostrados arriba. `models_comparison.py` reutiliza ese mismo dataset (o lo
regenera si no existe), y agrega `results/metrics.duckdb`,
`results/model_comparison.png` y `results/activation_comparison.png`.

### Tests

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

## Licencia

MIT — ver [LICENSE](LICENSE).

## Autor

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
