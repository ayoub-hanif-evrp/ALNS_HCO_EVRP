# ALNS-HCO vs published Schneider EVRPTW-PR BKS

Hanif results are not included. Ranking is feasible, then min $K$, then min distance.
BKS: Erdelić, T.; Carić, T. Goods Delivery with Electric Vehicles: Electric Vehicle Routing Optimization with Time Windows and Partial or Full Recharge. Energies 2022, 15, 285. https://doi.org/10.3390/en15010285

Instances: 56. Feasible: 44.
Fleet wins/ties/losses vs BKS: **0/15/29**.
Equal-K distance wins: 0. Near-BKS (|gap|<0.05% at equal K): 0.
Mean ΔK: 0.80.
Equal-K mean distance gap: 21.87%.

| instance | family | BKS source | BKS K | BKS D | our K | our D | ΔK | equal-K gap % | verdict | seed | s |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| c101_21 | C1 | Schiffer and Walther | 12 | 1043.38 | 12 | 1113.34 | 0 | 6.71 | distance_loss | 1 | 180.1 |
| c102_21 | C1 | Erdelic and Caric (2022) | 10 | 1058.67 | 12 | 1071.77 | 2 |  | fleet_loss | 0 | 180.1 |
| c103_21 | C1 | Erdelic and Caric (2022) | 10 | 971.19 | 11 | 1040.15 | 1 |  | fleet_loss | 1 | 180.1 |
| c104_21 | C1 | Schiffer and Walther | 10 | 884.38 | 10 | 988.23 | 0 | 11.74 | distance_loss | 4 | 180.1 |
| c105_21 | C1 | Erdelic and Caric (2022) | 10 | 1064.66 | 11 | 1178.79 | 1 |  | fleet_loss | 4 | 180.0 |
| c106_21 | C1 | Erdelic and Caric (2022) | 10 | 1061.61 | 12 | 1036.21 | 2 |  | fleet_loss | 1 | 180.1 |
| c107_21 | C1 | Schiffer and Walther | 10 | 1010.91 | 11 | 1111.82 | 1 |  | fleet_loss | 0 | 180.1 |
| c108_21 | C1 | Erdelic and Caric (2022) | 10 | 1022.93 | 11 | 1153.61 | 1 |  | fleet_loss | 1 | 180.1 |
| c109_21 | C1 | Schiffer and Walther | 10 | 940.38 | 11 | 1075.31 | 1 |  | fleet_loss | 4 | 180.1 |
| c201_21 | C2 | Keskin and Catay (2016) | 4 | 629.95 | 5 | 742.37 | 1 |  | fleet_loss | 0 | 180.1 |
| c202_21 | C2 | Keskin and Catay (2016) | 4 | 629.95 | 4 | 716.11 | 0 | 13.68 | distance_loss | 1 | 180.1 |
| c203_21 | C2 | Keskin and Catay (2016) | 4 | 629.95 | 4 | 892.26 | 0 | 41.64 | distance_loss | 3 | 180.1 |
| c204_21 | C2 | Schiffer and Walther | 4 | 628.91 | 4 | 862.47 | 0 | 37.14 | distance_loss | 1 | 180.1 |
| c205_21 | C2 | Keskin and Catay (2016) | 4 | 629.95 | 4 | 662.61 | 0 | 5.18 | distance_loss | 0 | 180.1 |
| c206_21 | C2 | Keskin and Catay (2016) | 4 | 629.95 | 5 | 735.63 | 1 |  | fleet_loss | 2 | 180.1 |
| c207_21 | C2 | Keskin and Catay (2016) | 4 | 629.95 | 5 | 861.54 | 1 |  | fleet_loss | 2 | 180.1 |
| c208_21 | C2 | Keskin and Catay (2016) | 4 | 629.95 | 4 | 841.98 | 0 | 33.66 | distance_loss | 2 | 180.1 |
| r101_21 | R1 | Erdelic and Caric (2022) | 17 | 1624.89 | 25 | 2262.13 | 8 |  | fleet_loss | 0 | 180.1 |
| r102_21 | R1 | Erdelic and Caric (2022) | 15 | 1454.53 | 21 | 1974.93 | 6 |  | fleet_loss | 0 | 180.1 |
| r103_21 | R1 | Erdelic and Caric (2022) | 12 | 1304.24 | 16 | 1704.01 | 4 |  | fleet_loss | 0 | 180.1 |
| r104_21 | R1 | Erdelic and Caric (2022) | 11 | 1051.41 | 12 | 1148.92 | 1 |  | fleet_loss | 1 | 180.1 |
| r105_21 | R1 | Schiffer and Walther | 14 | 1347.80 | 20 | 1991.82 | 6 |  | fleet_loss | 0 | 180.1 |
| r106_21 | R1 | Erdelic and Caric (2022) | 13 | 1263.13 | 18 | 1962.95 | 5 |  | fleet_loss | 0 | 180.1 |
| r107_21 | R1 | Erdelic and Caric (2022) | 11 | 1104.51 | 13 | 1255.14 | 2 |  | fleet_loss | 4 | 180.1 |
| r108_21 | R1 | Erdelic and Caric (2022) | 10 | 1030.44 | 17 | 1972.49 | 7 |  | fleet_loss | 0 | 180.1 |
| r109_21 | R1 | Erdelic and Caric (2022) | 12 | 1176.69 | 14 | 1398.49 | 2 |  | fleet_loss | 2 | 180.1 |
| r110_21 | R1 | Erdelic and Caric (2022) | 11 | 1067.11 | 13 | 1204.95 | 2 |  | fleet_loss | 2 | 180.1 |
| r111_21 | R1 | Erdelic and Caric (2022) | 11 | 1076.15 | 13 | 1221.12 | 2 |  | fleet_loss | 1 | 180.1 |
| r112_21 | R1 | Schiffer and Walther | 11 | 1001.79 | 12 | 1179.27 | 1 |  | fleet_loss | 0 | 180.1 |
| r201_21 | R2 | Schiffer and Walther | 3 | 1255.81 | 4 | 1476.60 | 1 |  | fleet_loss | 3 | 180.1 |
| r202_21 | R2 | Hiermann et al. | 3 | 1051.46 | 4 | 1102.02 | 1 |  | fleet_loss | 4 | 180.1 |
| r203_21 | R2 | Keskin and Catay (2016) | 3 | 895.54 | 4 | 1016.79 | 1 |  | fleet_loss | 2 | 180.1 |
| r204_21 | R2 | Schiffer and Walther | 2 | 779.49 | 3 | 770.83 | 1 |  | fleet_loss | 2 | 180.0 |
| r205_21 | R2 | Erdelic and Caric (2022) | 3 | 987.22 | 3 | 1123.11 | 0 | 13.76 | distance_loss | 3 | 180.1 |
| r206_21 | R2 | Erdelic and Caric (2022) | 3 | 922.08 | 3 | 1064.45 | 0 | 15.44 | distance_loss | 0 | 180.1 |
| r207_21 | R2 | Schiffer and Walther | 2 | 843.20 | 3 | 850.61 | 1 |  | fleet_loss | 3 | 180.1 |
| r208_21 | R2 | Keskin and Catay (2016) | 2 | 736.12 | 2 | 807.11 | 0 | 9.64 | distance_loss | 3 | 180.1 |
| r209_21 | R2 | Erdelic and Caric (2022) | 3 | 863.17 | 3 | 1055.08 | 0 | 22.23 | distance_loss | 4 | 180.0 |
| r210_21 | R2 | Keskin and Catay (2016) | 3 | 843.36 | 3 | 1057.87 | 0 | 25.43 | distance_loss | 3 | 180.1 |
| r211_21 | R2 | Erdelic and Caric (2022) | 2 | 825.25 | 3 | 905.97 | 1 |  | fleet_loss | 1 | 180.1 |
| rc101_21 | RC1 | Erdelic and Caric (2022) | 15 | 1661.53 | 24 | 2702.06 | 9 |  | fleet_loss | 0 | 180.1 |
| rc102_21 | RC1 | Keskin and Catay (2016) | 14 | 1155.50 | 15 | 1678.23 | 1 |  | fleet_loss | 3 | 180.1 |
| rc103_21 | RC1 | Erdelic and Caric (2022) | 12 | 1359.34 | 18 | 2181.90 | 6 |  | fleet_loss | 0 | 180.1 |
| rc104_21 | RC1 | Erdelic and Caric (2022) | 11 | 1174.32 | 16 | 2189.48 | 5 |  | fleet_loss | 0 | 180.1 |
| rc105_21 | RC1 | Erdelic and Caric (2022) | 13 | 1471.80 | 22 | 2522.03 | 9 |  | fleet_loss | 0 | 180.1 |
| rc106_21 | RC1 | Schiffer and Walther | 13 | 1385.96 | 21 | 2438.38 | 8 |  | fleet_loss | 0 | 180.1 |
| rc107_21 | RC1 | Erdelic and Caric (2022) | 11 | 1244.37 | 20 | 2457.92 | 9 |  | fleet_loss | 0 | 180.1 |
| rc108_21 | RC1 | Schiffer and Walther | 11 | 1154.14 | 12 | 1294.04 | 1 |  | fleet_loss | 4 | 180.1 |
| rc201_21 | RC2 | Erdelic and Caric (2022) | 4 | 1433.57 | 5 | 1658.82 | 1 |  | fleet_loss | 0 | 180.1 |
| rc202_21 | RC2 | Erdelic and Caric (2022) | 3 | 1403.67 | 4 | 1454.02 | 1 |  | fleet_loss | 4 | 180.1 |
| rc203_21 | RC2 | Erdelic and Caric (2022) | 3 | 1054.91 | 4 | 1164.86 | 1 |  | fleet_loss | 2 | 180.1 |
| rc204_21 | RC2 | Schiffer and Walther | 3 | 884.75 | 3 | 1105.01 | 0 | 24.90 | distance_loss | 1 | 180.1 |
| rc205_21 | RC2 | Erdelic and Caric (2022) | 3 | 1238.46 | 4 | 1425.44 | 1 |  | fleet_loss | 4 | 180.0 |
| rc206_21 | RC2 | Schiffer and Walther | 3 | 1189.11 | 4 | 1409.34 | 1 |  | fleet_loss | 0 | 180.1 |
| rc207_21 | RC2 | Erdelic and Caric (2022) | 3 | 978.30 | 3 | 1323.35 | 0 | 35.27 | distance_loss | 2 | 180.1 |
| rc208_21 | RC2 | Schiffer and Walther | 3 | 833.12 | 3 | 1096.15 | 0 | 31.57 | distance_loss | 4 | 180.1 |

## By family

| family | n | wins | ties | losses | mean ΔK |
|---|---:|---:|---:|---:|---:|
| C1 | 9 | 0 | 2 | 7 | 1.00 |
| C2 | 8 | 0 | 5 | 3 | 0.38 |
| R1 | 6 | 0 | 0 | 6 | 1.67 |
| R2 | 11 | 0 | 5 | 6 | 0.55 |
| RC1 | 2 | 0 | 0 | 2 | 1.00 |
| RC2 | 8 | 0 | 3 | 5 | 0.62 |

