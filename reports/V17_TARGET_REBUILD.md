# v17 strict-target rebuild

Conflict threshold: 1.0 t/ha (max-min across non-null sources).
Target priority: physical_t_ha > prod_fact_t_ha > productivity_t_ha (per-crop cap normalized).

| base | asof | rows_in | rows_v17_target | rows_conflict | rows_clean |
|------|------|--------:|----------------:|--------------:|-----------:|
| v12 | 07_01 | 183 | 183 | 3 | 180 |
| v12 | 08_01 | 183 | 183 | 3 | 180 |
| v12 | 09_01 | 183 | 183 | 3 | 180 |
| v14 | 07_01 | 183 | 183 | 3 | 180 |
| v14 | 08_01 | 183 | 183 | 3 | 180 |
| v14 | 09_01 | 183 | 183 | 3 | 180 |
| v15_rates | 07_01 | 183 | 183 | 3 | 180 |
| v15_rates | 08_01 | 183 | 183 | 3 | 180 |
| v15_rates | 09_01 | 183 | 183 | 3 | 180 |
