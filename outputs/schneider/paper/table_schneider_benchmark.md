# Schneider EVRPTW external benchmark

Hanif results are not included. Published columns come from the repository BKS file
(`src/schneider/evaluation/schneider_bks.py`, Erdelić & Carić 2022 compilation).
Our columns are the archived Schneider PR run in `outputs/schneider/pr/`.

Instances: 56. Fleet wins/ties/losses: 0/5/51.
Equal-K mean distance gap: 76.05%
Near-BKS matches (|gap|<0.05% at equal K): 0. New best (ΔK<0): 0.

| instance | source | pub K | pub D | our K | our D | ΔK | equal-K gap % | runtime s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c101_21 | Schiffer and Walther | 12 | 1043.38 | 13 | 1323.22 | 1 |  | 360.0 |
| c102_21 | Erdelic and Caric (2022) | 10 | 1058.67 | 13 | 1391.38 | 3 |  | 320.2 |
| c103_21 | Erdelic and Caric (2022) | 10 | 971.19 | 12 | 1614.10 | 2 |  | 405.8 |
| c104_21 | Schiffer and Walther | 10 | 884.38 | 11 | 1311.93 | 1 |  | 367.8 |
| c105_21 | Erdelic and Caric (2022) | 10 | 1064.66 | 13 | 1350.20 | 3 |  | 320.0 |
| c106_21 | Erdelic and Caric (2022) | 10 | 1061.61 | 14 | 1638.27 | 4 |  | 325.6 |
| c107_21 | Schiffer and Walther | 10 | 1010.91 | 13 | 1331.15 | 3 |  | 320.0 |
| c108_21 | Erdelic and Caric (2022) | 10 | 1022.93 | 13 | 1528.64 | 3 |  | 320.0 |
| c109_21 | Schiffer and Walther | 10 | 940.38 | 14 | 1587.80 | 4 |  | 320.0 |
| c201_21 | Keskin and Catay (2016) | 4 | 629.95 | 5 | 880.80 | 1 |  | 376.0 |
| c202_21 | Keskin and Catay (2016) | 4 | 629.95 | 5 | 1070.14 | 1 |  | 360.0 |
| c203_21 | Keskin and Catay (2016) | 4 | 629.95 | 4 | 1145.81 | 0 | 81.89 | 359.1 |
| c204_21 | Schiffer and Walther | 4 | 628.91 | 4 | 1368.52 | 0 | 117.60 | 366.6 |
| c205_21 | Keskin and Catay (2016) | 4 | 629.95 | 6 | 1121.54 | 2 |  | 323.1 |
| c206_21 | Keskin and Catay (2016) | 4 | 629.95 | 5 | 1110.86 | 1 |  | 371.9 |
| c207_21 | Keskin and Catay (2016) | 4 | 629.95 | 6 | 1604.79 | 2 |  | 320.0 |
| c208_21 | Keskin and Catay (2016) | 4 | 629.95 | 5 | 982.15 | 1 |  | 360.0 |
| r101_21 | Erdelic and Caric (2022) | 17 | 1624.89 | 23 | 2098.95 | 6 |  | 405.3 |
| r102_21 | Erdelic and Caric (2022) | 15 | 1454.53 | 19 | 1941.09 | 4 |  | 342.0 |
| r103_21 | Erdelic and Caric (2022) | 12 | 1304.24 | 17 | 1625.38 | 5 |  | 351.2 |
| r104_21 | Erdelic and Caric (2022) | 11 | 1051.41 | 14 | 1434.98 | 3 |  | 320.0 |
| r105_21 | Schiffer and Walther | 14 | 1347.80 | 20 | 1994.93 | 6 |  | 364.3 |
| r106_21 | Erdelic and Caric (2022) | 13 | 1263.13 | 18 | 2040.98 | 5 |  | 346.4 |
| r107_21 | Erdelic and Caric (2022) | 11 | 1104.51 | 15 | 1599.72 | 4 |  | 320.0 |
| r108_21 | Erdelic and Caric (2022) | 10 | 1030.44 | 14 | 1414.22 | 4 |  | 370.1 |
| r109_21 | Erdelic and Caric (2022) | 12 | 1176.69 | 16 | 1569.63 | 4 |  | 320.0 |
| r110_21 | Erdelic and Caric (2022) | 11 | 1067.11 | 16 | 1692.52 | 5 |  | 320.0 |
| r111_21 | Erdelic and Caric (2022) | 11 | 1076.15 | 16 | 1683.02 | 5 |  | 332.1 |
| r112_21 | Schiffer and Walther | 11 | 1001.79 | 14 | 1468.55 | 3 |  | 347.1 |
| r201_21 | Schiffer and Walther | 3 | 1255.81 | 4 | 1876.62 | 1 |  | 360.0 |
| r202_21 | Hiermann et al. | 3 | 1051.46 | 4 | 1357.36 | 1 |  | 360.0 |
| r203_21 | Keskin and Catay (2016) | 3 | 895.54 | 3 | 1517.50 | 0 | 69.45 | 365.0 |
| r204_21 | Schiffer and Walther | 2 | 779.49 | 3 | 1159.01 | 1 |  | 357.3 |
| r205_21 | Erdelic and Caric (2022) | 3 | 987.22 | 3 | 1554.13 | 0 | 57.43 | 370.3 |
| r206_21 | Erdelic and Caric (2022) | 3 | 922.08 | 3 | 1418.86 | 0 | 53.88 | 361.9 |
| r207_21 | Schiffer and Walther | 2 | 843.20 | 3 | 1319.67 | 1 |  | 363.7 |
| r208_21 | Keskin and Catay (2016) | 2 | 736.12 | 3 | 1170.58 | 1 |  | 360.0 |
| r209_21 | Erdelic and Caric (2022) | 3 | 863.17 | 4 | 1537.73 | 1 |  | 363.9 |
| r210_21 | Keskin and Catay (2016) | 3 | 843.36 | 4 | 1773.10 | 1 |  | 360.0 |
| r211_21 | Erdelic and Caric (2022) | 2 | 825.25 | 3 | 1262.78 | 1 |  | 359.9 |
| rc101_21 | Erdelic and Caric (2022) | 15 | 1661.53 | 21 | 2292.63 | 6 |  | 410.8 |
| rc102_21 | Keskin and Catay (2016) | 14 | 1155.50 | 19 | 2063.59 | 5 |  | 353.0 |
| rc103_21 | Erdelic and Caric (2022) | 12 | 1359.34 | 18 | 1973.18 | 6 |  | 320.0 |
| rc104_21 | Erdelic and Caric (2022) | 11 | 1174.32 | 14 | 1517.04 | 3 |  | 320.0 |
| rc105_21 | Erdelic and Caric (2022) | 13 | 1471.80 | 19 | 2045.75 | 6 |  | 344.4 |
| rc106_21 | Schiffer and Walther | 13 | 1385.96 | 19 | 2034.19 | 6 |  | 370.1 |
| rc107_21 | Erdelic and Caric (2022) | 11 | 1244.37 | 16 | 1989.23 | 5 |  | 336.2 |
| rc108_21 | Schiffer and Walther | 11 | 1154.14 | 17 | 2062.66 | 6 |  | 346.6 |
| rc201_21 | Erdelic and Caric (2022) | 4 | 1433.57 | 5 | 2162.56 | 1 |  | 360.0 |
| rc202_21 | Erdelic and Caric (2022) | 3 | 1403.67 | 5 | 2045.54 | 2 |  | 324.8 |
| rc203_21 | Erdelic and Caric (2022) | 3 | 1054.91 | 5 | 1791.27 | 2 |  | 321.7 |
| rc204_21 | Schiffer and Walther | 3 | 884.75 | 4 | 1619.52 | 1 |  | 364.3 |
| rc205_21 | Erdelic and Caric (2022) | 3 | 1238.46 | 5 | 1925.05 | 2 |  | 320.0 |
| rc206_21 | Schiffer and Walther | 3 | 1189.11 | 4 | 2019.27 | 1 |  | 360.0 |
| rc207_21 | Erdelic and Caric (2022) | 3 | 978.30 | 4 | 1940.39 | 1 |  | 360.0 |
| rc208_21 | Schiffer and Walther | 3 | 833.12 | 4 | 1554.21 | 1 |  | 360.0 |
