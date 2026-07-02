# H17g — Portfolio rep risk-bounding SL verdict

| rep | pair | tf | TP-only oos_pd | TP-only oos_dd | best-SL | SL oos_pd | SL oos_dd | edge % | r_sl | verdict |
|-----|------|----|---------------:|---------------:|---------|----------:|----------:|-------:|-----:|---------|
| 30bb3bff76bf | USD_JPY | S5/M10/M30 | +10.78 | +0.0 | sl_200p | +12.74 | +0.0 | 118% | 0.04 | **KEEP** (SL=sl_200p) |
| a237cb171d85 | USD_JPY | S5/S30/M5 | +12.08 | -106.9 | — | — | — | — | — | **CULL** (no SL keeps edge) |
| ba2988590380 | EUR_USD | S5/M1/M15 | +9.04 | -59.2 | — | — | — | — | — | **CULL** (no SL keeps edge) |
| f5ea18b318f2 | GBP_USD | S5/S30/M5 | +6.64 | -71.4 | — | — | — | — | — | **CULL** (no SL keeps edge) |
| 268d48dadd94 | EUR_JPY | S5/S30/M5/H1 | +7.18 | +0.0 | — | — | — | — | — | **CULL** (no SL keeps edge) |
| 9eda87cef382 | EUR_JPY | S5/M1/M5/H1 | +7.05 | +0.0 | — | — | — | — | — | **CULL** (no SL keeps edge) |

Survive with bounding SL: 1/6
