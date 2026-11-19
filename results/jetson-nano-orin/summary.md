# Benchmark summary

GPU: Orin (8 SMs, 7.4 GB, CC 8.7)

## Accuracy (%), stratified 10-fold cross-validation

| dataset | records | tree: gain | tree: gain, pruned | tree: gain ratio | tree: gain ratio, pruned | tree: Gini | tree: Gini, pruned | naive Bayes | kNN (k=5) | logistic regression |
|---|---|---|---|---|---|---|---|---|---|---|
| votes | 435 | 93.1 | 94.2 | 94.3 | 95.2 | 94.2 | 94.9 | 90.1 | 92.4 | **96.3** |
| tictactoe | 958 | 86.2 | 85.7 | 86.9 | 85.7 | 94.3 | 94.6 | 69.5 | **98.5** | 98.3 |
| splice | 3,190 | 91.2 | 93.7 | 91.8 | 94.1 | 92.8 | 94.0 | **95.4** | 79.8 | 94.6 |
| chess | 3,196 | **99.7** | 99.6 | **99.7** | 99.6 | 99.7 | 99.4 | 87.7 | 96.3 | 96.9 |
| mushroom | 8,124 | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | 95.5 | **100.0** | **100.0** |
| nursery | 12,960 | 98.9 | 97.0 | 98.8 | 97.1 | **99.9** | 99.5 | 90.2 | 98.1 | 92.5 |
| letter | 20,000 | 88.2 | 88.3 | 87.5 | 87.7 | 87.7 | 87.7 | 64.2 | **95.4** | 77.3 |
| krkopt | 28,056 | 72.8 | 71.4 | 72.7 | 71.2 | **82.0** | 80.6 | 24.7 | 57.7 | 32.7 |
| adult | 48,842 | 81.7 | 85.1 | 82.3 | **85.8** | 82.2 | **85.8** | 83.2 | 82.8 | 85.2 |
| connect4 | 67,557 | 75.8 | 78.0 | 79.1 | **80.9** | 77.1 | 78.7 | 72.2 | **80.9** | 75.8 |
| census | 299,285 | 93.3 | 94.8 | 93.7 | **95.3** | 93.4 | 94.8 | 85.6 | 94.3 | **95.3** |
| covertype | 581,012 | **95.3** | **95.3** | 94.9 | 94.9 | 94.3 | 94.4 | 65.7 | 93.9 | 72.4 |
| poker | 1,025,010 | 51.8 | 56.7 | 61.8 | **65.5** | 60.7 | 63.3 | 50.1 | 53.4 | 50.1 |

## Time per fold (ms): training + prediction, CPU vs GPU

Mean over the folds. *: one stratified holdout split instead of 10 folds.

| dataset | decision tree (gain) CPU | GPU | speedup | naive Bayes CPU | GPU | speedup | kNN (k=5) CPU | GPU | speedup | logistic regression CPU | GPU | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| votes | 0.17 | 4.99 | 0.03x | 0.03 | 1.67 | 0.02x | 0.21 | 2.41 | 0.09x | 2.67 | 5.85 | 0.46x |
| tictactoe | 0.43 | 4.45 | 0.10x | 0.05 | 1.68 | 0.03x | 0.57 | 1.82 | 0.31x | 4.40 | 6.57 | 0.67x |
| splice | 5.90 | 7.04 | 0.84x | 0.56 | 2.00 | 0.28x | 24.0 | 14.1 | 1.7x | 122 | 34.1 | 3.6x |
| chess | 2.84 | 6.44 | 0.44x | 0.64 | 1.79 | 0.36x | 15.2 | 8.16 | 1.9x | 76.7 | 21.7 | 3.5x |
| mushroom | 1.66 | 3.39 | 0.49x | 0.76 | 1.94 | 0.39x | 62.4 | 17.1 | 3.6x | 57.4 | 14.5 | 3.9x |
| nursery | 2.72 | 5.95 | 0.46x | 0.38 | 1.75 | 0.22x | 80.0 | 14.8 | 5.4x | 315 | 68.7 | 4.6x |
| letter | 58.7 | 31.9 | 1.8x | 5.08 | 4.85 | 1.0x | 305 | 108 | 2.8x | 1,309 | 356 | 3.7x |
| krkopt | 25.6 | 22.7 | 1.1x | 2.26 | 3.55 | 0.64x | 326 | 29.1 | 11x | 1,582 | 347 | 4.6x |
| adult | 73.9 | 36.9 | 2.0x | 3.92 | 4.15 | 0.94x | 1,621 | 178 | 9.1x | 877 | 116 | 7.5x |
| connect4 | 227 | 56.6 | 4.0x | 12.2 | 9.43 | 1.3x | 8,721 | 1,832 | 4.8x | 4,266 | 532 | 8.0x |
| census | 1,661 | 176 | 9.4x | 68.1 | 32.0 | 2.1x | 185,296* | 38,320* | 4.8x | 15,378 | 1,797 | 8.6x |
| covertype | 1,770 | 494 | 3.6x | 84.6 | 44.0 | 1.9x | 264,344* | 23,442* | 11x | 21,622 | 3,860 | 5.6x |
| poker | 3,978 | 720 | 5.5x | 67.4 | 66.8 | 1.0x | 410,371* | 26,506* | 15x | 30,811 | 7,393 | 4.2x |

## Training and prediction time per fold (ms)

| dataset | classifier | split | CPU train | CPU predict | GPU train | GPU predict |
|---|---|---|---|---|---|---|
| votes | tree: gain | 10-fold | 0.17 | 0.00 | 3.96 | 1.02 |
| votes | tree: gain, pruned | 10-fold | 0.20 | 0.00 | 4.29 | 0.62 |
| votes | tree: gain ratio | 10-fold | 0.20 | 0.00 | 3.75 | 0.57 |
| votes | tree: gain ratio, pruned | 10-fold | 0.18 | 0.00 | 5.31 | 0.60 |
| votes | tree: Gini | 10-fold | 0.29 | 0.00 | 4.79 | 0.67 |
| votes | tree: Gini, pruned | 10-fold | 0.26 | 0.00 | 8.32 | 1.15 |
| votes | naive Bayes | 10-fold | 0.02 | 0.01 | 0.98 | 0.69 |
| votes | kNN (k=5) | 10-fold | 0.01 | 0.19 | 0.43 | 1.98 |
| votes | logistic regression | 10-fold | 2.66 | 0.01 | 5.25 | 0.60 |
| tictactoe | tree: gain | 10-fold | 0.43 | 0.01 | 3.91 | 0.53 |
| tictactoe | tree: gain, pruned | 10-fold | 0.43 | 0.01 | 3.98 | 0.54 |
| tictactoe | tree: gain ratio | 10-fold | 0.43 | 0.01 | 4.65 | 0.56 |
| tictactoe | tree: gain ratio, pruned | 10-fold | 0.45 | 0.01 | 4.02 | 0.54 |
| tictactoe | tree: Gini | 10-fold | 0.47 | 0.01 | 7.78 | 0.63 |
| tictactoe | tree: Gini, pruned | 10-fold | 0.51 | 0.01 | 6.55 | 0.55 |
| tictactoe | naive Bayes | 10-fold | 0.04 | 0.01 | 0.98 | 0.70 |
| tictactoe | kNN (k=5) | 10-fold | 0.02 | 0.55 | 0.43 | 1.39 |
| tictactoe | logistic regression | 10-fold | 4.39 | 0.01 | 6.00 | 0.58 |
| splice | tree: gain | 10-fold | 5.88 | 0.02 | 6.44 | 0.61 |
| splice | tree: gain, pruned | 10-fold | 5.94 | 0.01 | 5.79 | 0.57 |
| splice | tree: gain ratio | 10-fold | 6.09 | 0.03 | 5.62 | 0.59 |
| splice | tree: gain ratio, pruned | 10-fold | 6.11 | 0.02 | 5.96 | 0.57 |
| splice | tree: Gini | 10-fold | 9.82 | 0.04 | 10.2 | 0.67 |
| splice | tree: Gini, pruned | 10-fold | 9.85 | 0.03 | 9.17 | 0.65 |
| splice | naive Bayes | 10-fold | 0.43 | 0.12 | 1.21 | 0.79 |
| splice | kNN (k=5) | 10-fold | 0.88 | 23.2 | 5.77 | 8.37 |
| splice | logistic regression | 10-fold | 122 | 0.21 | 33.0 | 1.12 |
| chess | tree: gain | 10-fold | 2.82 | 0.02 | 5.86 | 0.57 |
| chess | tree: gain, pruned | 10-fold | 2.83 | 0.02 | 5.88 | 0.57 |
| chess | tree: gain ratio | 10-fold | 3.01 | 0.02 | 5.83 | 0.56 |
| chess | tree: gain ratio, pruned | 10-fold | 3.02 | 0.02 | 5.81 | 0.57 |
| chess | tree: Gini | 10-fold | 3.30 | 0.03 | 7.26 | 0.59 |
| chess | tree: Gini, pruned | 10-fold | 3.34 | 0.03 | 7.34 | 0.59 |
| chess | naive Bayes | 10-fold | 0.57 | 0.07 | 1.07 | 0.72 |
| chess | kNN (k=5) | 10-fold | 0.55 | 14.7 | 3.23 | 4.92 |
| chess | logistic regression | 10-fold | 76.6 | 0.12 | 20.8 | 0.91 |
| mushroom | tree: gain | 10-fold | 1.65 | 0.02 | 2.83 | 0.55 |
| mushroom | tree: gain, pruned | 10-fold | 1.65 | 0.02 | 2.89 | 0.56 |
| mushroom | tree: gain ratio | 10-fold | 1.67 | 0.02 | 2.78 | 0.56 |
| mushroom | tree: gain ratio, pruned | 10-fold | 1.67 | 0.02 | 2.88 | 0.55 |
| mushroom | tree: Gini | 10-fold | 4.08 | 0.05 | 4.75 | 0.57 |
| mushroom | tree: Gini, pruned | 10-fold | 4.08 | 0.04 | 4.66 | 0.58 |
| mushroom | naive Bayes | 10-fold | 0.64 | 0.12 | 1.23 | 0.71 |
| mushroom | kNN (k=5) | 10-fold | 0.83 | 61.6 | 4.88 | 12.2 |
| mushroom | logistic regression | 10-fold | 57.2 | 0.21 | 13.2 | 1.30 |
| nursery | tree: gain | 10-fold | 2.64 | 0.08 | 5.36 | 0.58 |
| nursery | tree: gain, pruned | 10-fold | 2.70 | 0.07 | 5.47 | 0.57 |
| nursery | tree: gain ratio | 10-fold | 2.66 | 0.07 | 5.39 | 0.56 |
| nursery | tree: gain ratio, pruned | 10-fold | 2.73 | 0.07 | 5.44 | 0.58 |
| nursery | tree: Gini | 10-fold | 3.68 | 0.15 | 8.46 | 0.61 |
| nursery | tree: Gini, pruned | 10-fold | 3.71 | 0.14 | 8.54 | 0.61 |
| nursery | naive Bayes | 10-fold | 0.28 | 0.10 | 1.04 | 0.71 |
| nursery | kNN (k=5) | 10-fold | 0.23 | 79.8 | 2.42 | 12.4 |
| nursery | logistic regression | 10-fold | 315 | 0.14 | 67.6 | 1.08 |
| letter | tree: gain | 10-fold | 58.3 | 0.42 | 30.9 | 0.92 |
| letter | tree: gain, pruned | 10-fold | 58.6 | 0.37 | 29.9 | 0.79 |
| letter | tree: gain ratio | 10-fold | 63.5 | 0.48 | 32.7 | 0.80 |
| letter | tree: gain ratio, pruned | 10-fold | 63.7 | 0.45 | 35.5 | 1.21 |
| letter | tree: Gini | 10-fold | 65.8 | 0.49 | 40.3 | 1.25 |
| letter | tree: Gini, pruned | 10-fold | 66.2 | 0.46 | 41.0 | 0.97 |
| letter | naive Bayes | 10-fold | 1.81 | 3.27 | 3.21 | 1.64 |
| letter | kNN (k=5) | 10-fold | 3.53 | 302 | 9.88 | 97.6 |
| letter | logistic regression | 10-fold | 1,308 | 0.95 | 355 | 1.71 |
| krkopt | tree: gain | 10-fold | 25.1 | 0.55 | 21.8 | 0.86 |
| krkopt | tree: gain, pruned | 10-fold | 26.9 | 0.51 | 22.3 | 1.32 |
| krkopt | tree: gain ratio | 10-fold | 25.0 | 0.55 | 23.6 | 1.29 |
| krkopt | tree: gain ratio, pruned | 10-fold | 27.0 | 0.51 | 25.0 | 1.35 |
| krkopt | tree: Gini | 10-fold | 34.0 | 0.84 | 58.9 | 1.05 |
| krkopt | tree: Gini, pruned | 10-fold | 35.1 | 0.78 | 60.3 | 1.27 |
| krkopt | naive Bayes | 10-fold | 1.50 | 0.76 | 2.62 | 0.92 |
| krkopt | kNN (k=5) | 10-fold | 1.40 | 325 | 4.50 | 24.6 |
| krkopt | logistic regression | 10-fold | 1,581 | 0.41 | 345 | 1.40 |
| adult | tree: gain | 10-fold | 72.7 | 1.26 | 35.8 | 1.10 |
| adult | tree: gain, pruned | 10-fold | 74.1 | 0.36 | 36.0 | 1.33 |
| adult | tree: gain ratio | 10-fold | 92.7 | 1.66 | 43.9 | 1.23 |
| adult | tree: gain ratio, pruned | 10-fold | 94.3 | 0.65 | 44.5 | 1.40 |
| adult | tree: Gini | 10-fold | 121 | 2.25 | 59.1 | 1.33 |
| adult | tree: Gini, pruned | 10-fold | 122 | 0.83 | 59.0 | 1.10 |
| adult | naive Bayes | 10-fold | 3.43 | 0.49 | 3.33 | 0.83 |
| adult | kNN (k=5) | 10-fold | 2.81 | 1,618 | 17.6 | 160 |
| adult | logistic regression | 10-fold | 876 | 0.81 | 115 | 1.88 |
| connect4 | tree: gain | 10-fold | 225 | 2.45 | 54.2 | 2.36 |
| connect4 | tree: gain, pruned | 10-fold | 227 | 1.16 | 58.5 | 1.84 |
| connect4 | tree: gain ratio | 10-fold | 237 | 2.26 | 59.1 | 2.50 |
| connect4 | tree: gain ratio, pruned | 10-fold | 238 | 1.32 | 58.9 | 1.91 |
| connect4 | tree: Gini | 10-fold | 371 | 4.21 | 78.7 | 2.56 |
| connect4 | tree: Gini, pruned | 10-fold | 371 | 3.28 | 81.5 | 2.02 |
| connect4 | naive Bayes | 10-fold | 10.2 | 2.06 | 7.58 | 1.85 |
| connect4 | kNN (k=5) | 10-fold | 10.2 | 8,711 | 105 | 1,727 |
| connect4 | logistic regression | 10-fold | 4,261 | 5.26 | 528 | 4.17 |
| census | tree: gain | 10-fold | 1,652 | 8.99 | 171 | 5.30 |
| census | tree: gain, pruned | 10-fold | 1,652 | 1.67 | 173 | 3.82 |
| census | tree: gain ratio | 10-fold | 2,036 | 11.5 | 221 | 5.29 |
| census | tree: gain ratio, pruned | 10-fold | 1,997 | 2.56 | 231 | 3.74 |
| census | tree: Gini | 10-fold | 3,469 | 18.2 | 364 | 3.64 |
| census | tree: Gini, pruned | 10-fold | 3,457 | 6.15 | 363 | 3.42 |
| census | naive Bayes | 10-fold | 58.6 | 9.48 | 28.2 | 3.81 |
| census | kNN (k=5) | 10-fold | - | - | 410 | 37,900 |
| census | kNN (k=5) | holdout | 56.6 | 185,240 | 426 | 37,894 |
| census | logistic regression | 10-fold | 15,350 | 28.7 | 1,786 | 10.7 |
| covertype | tree: gain | 10-fold | 1,757 | 13.3 | 491 | 2.94 |
| covertype | tree: gain, pruned | 10-fold | 1,698 | 14.2 | 523 | 2.85 |
| covertype | tree: gain ratio | 10-fold | 1,894 | 16.8 | 609 | 2.88 |
| covertype | tree: gain ratio, pruned | 10-fold | 1,956 | 18.9 | 635 | 2.94 |
| covertype | tree: Gini | 10-fold | 2,399 | 21.2 | 743 | 3.19 |
| covertype | tree: Gini, pruned | 10-fold | 2,379 | 21.4 | 781 | 2.61 |
| covertype | naive Bayes | 10-fold | 66.5 | 18.1 | 36.7 | 7.30 |
| covertype | kNN (k=5) | 10-fold | - | - | 198 | 23,181 |
| covertype | kNN (k=5) | holdout | 56.9 | 264,287 | 192 | 23,250 |
| covertype | logistic regression | 10-fold | 21,608 | 13.8 | 3,846 | 13.8 |
| poker | tree: gain | 10-fold | 3,879 | 98.9 | 708 | 11.9 |
| poker | tree: gain, pruned | 10-fold | 3,950 | 64.7 | 769 | 6.03 |
| poker | tree: gain ratio | 10-fold | 4,131 | 95.2 | 742 | 17.8 |
| poker | tree: gain ratio, pruned | 10-fold | 4,176 | 76.3 | 800 | 9.47 |
| poker | tree: Gini | 10-fold | 4,703 | 137 | 880 | 11.4 |
| poker | tree: Gini, pruned | 10-fold | 4,770 | 141 | 1,024 | 11.7 |
| poker | naive Bayes | 10-fold | 45.1 | 22.3 | 56.6 | 10.2 |
| poker | kNN (k=5) | 10-fold | - | - | 241 | 49,042 |
| poker | kNN (k=5) | holdout | 59.0 | 410,312 | 254 | 26,251 |
| poker | logistic regression | 10-fold | 30,788 | 22.3 | 7,379 | 13.5 |

## Peak memory (MB)

Host: peak resident memory of the CPU program (includes the loaded table). Device: peak GPU memory allocated by the CUDA program.

| dataset | decision tree (gain) host | device | naive Bayes host | device | kNN (k=5) host | device | logistic regression host | device |
|---|---|---|---|---|---|---|---|---|
| votes | 15 | 0 | 15 | 0 | 15 | 0 | 15 | 0 |
| tictactoe | 15 | 0 | 15 | 0 | 15 | 0 | 15 | 0 |
| splice | 15 | 1 | 15 | 1 | 15 | 1 | 15 | 2 |
| chess | 15 | 1 | 15 | 0 | 15 | 0 | 15 | 1 |
| mushroom | 15 | 1 | 15 | 1 | 15 | 1 | 15 | 2 |
| nursery | 15 | 1 | 15 | 0 | 15 | 1 | 15 | 2 |
| letter | 15 | 12 | 15 | 2 | 15 | 3 | 15 | 7 |
| krkopt | 15 | 9 | 15 | 1 | 15 | 4 | 15 | 6 |
| adult | 15 | 9 | 15 | 2 | 16 | 11 | 16 | 8 |
| connect4 | 37 | 21 | 35 | 10 | 54 | 27 | 54 | 32 |
| census | 243 | 73 | 243 | 42 | 243 | 65 | 243 | 131 |
| covertype | 116 | 278 | 91 | 28 | 130 | 91 | 139 | 107 |
| poker | 280 | 380 | 145 | 42 | 196 | 118 | 226 | 190 |

## Decision trees: size and training time

Nodes and depth are means over the folds; time is CPU training per fold.

| dataset | measure | accuracy % | nodes | depth | pruned accuracy % | pruned nodes | CPU train ms | GPU train ms |
|---|---|---|---|---|---|---|---|---|
| votes | gain | 93.1 | 66 | 7 | 94.2 | 18 | 0.17 | 3.96 |
| votes | gain ratio | 94.3 | 71 | 8 | 95.2 | 18 | 0.20 | 3.75 |
| votes | Gini | 94.2 | 48 | 9 | 94.9 | 15 | 0.29 | 4.79 |
| tictactoe | gain | 86.2 | 335 | 7 | 85.7 | 170 | 0.43 | 3.91 |
| tictactoe | gain ratio | 86.9 | 328 | 7 | 85.7 | 178 | 0.43 | 4.65 |
| tictactoe | Gini | 94.3 | 134 | 12 | 94.6 | 90 | 0.47 | 7.78 |
| splice | gain | 91.2 | 658 | 9 | 93.7 | 223 | 5.88 | 6.44 |
| splice | gain ratio | 91.8 | 698 | 10 | 94.1 | 285 | 6.09 | 5.62 |
| splice | Gini | 92.8 | 251 | 12 | 94.0 | 148 | 9.82 | 10.2 |
| chess | gain | 99.7 | 93 | 15 | 99.6 | 63 | 2.82 | 5.86 |
| chess | gain ratio | 99.7 | 91 | 16 | 99.6 | 59 | 3.01 | 5.83 |
| chess | Gini | 99.7 | 101 | 15 | 99.4 | 59 | 3.30 | 7.26 |
| mushroom | gain | 100.0 | 38 | 4 | 100.0 | 38 | 1.65 | 2.83 |
| mushroom | gain ratio | 100.0 | 29 | 5 | 100.0 | 29 | 1.67 | 2.78 |
| mushroom | Gini | 100.0 | 19 | 8 | 100.0 | 18 | 4.08 | 4.75 |
| nursery | gain | 98.9 | 1,140 | 8 | 97.0 | 508 | 2.64 | 5.36 |
| nursery | gain ratio | 98.8 | 1,139 | 8 | 97.1 | 506 | 2.66 | 5.39 |
| nursery | Gini | 99.9 | 397 | 13 | 99.5 | 325 | 3.68 | 8.46 |
| letter | gain | 88.2 | 3,919 | 20 | 88.3 | 3,550 | 58.3 | 30.9 |
| letter | gain ratio | 87.5 | 3,977 | 27 | 87.7 | 3,441 | 63.5 | 32.7 |
| letter | Gini | 87.7 | 4,190 | 29 | 87.7 | 3,690 | 65.8 | 40.3 |
| krkopt | gain | 72.8 | 19,068 | 16 | 71.4 | 11,987 | 25.1 | 21.8 |
| krkopt | gain ratio | 72.7 | 19,117 | 17 | 71.2 | 11,880 | 25.0 | 23.6 |
| krkopt | Gini | 82.0 | 13,749 | 22 | 80.6 | 8,592 | 34.0 | 58.9 |
| adult | gain | 81.7 | 24,628 | 33 | 85.1 | 3,819 | 72.7 | 35.8 |
| adult | gain ratio | 82.3 | 26,438 | 42 | 85.8 | 3,079 | 92.7 | 43.9 |
| adult | Gini | 82.2 | 14,923 | 44 | 85.8 | 2,371 | 121 | 59.1 |
| connect4 | gain | 75.8 | 37,725 | 20 | 78.0 | 11,112 | 225 | 54.2 |
| connect4 | gain ratio | 79.1 | 33,978 | 24 | 80.9 | 11,147 | 237 | 59.1 |
| connect4 | Gini | 77.1 | 27,630 | 36 | 78.7 | 13,534 | 371 | 78.7 |
| census | gain | 93.3 | 102,415 | 32 | 94.8 | 9,718 | 1,652 | 171 |
| census | gain ratio | 93.7 | 109,146 | 40 | 95.3 | 8,803 | 2,036 | 221 |
| census | Gini | 93.4 | 27,518 | 55 | 94.8 | 8,198 | 3,469 | 364 |
| covertype | gain | 95.3 | 44,186 | 32 | 95.3 | 34,956 | 1,757 | 491 |
| covertype | gain ratio | 94.9 | 50,778 | 47 | 94.9 | 39,401 | 1,894 | 609 |
| covertype | Gini | 94.3 | 54,270 | 37 | 94.4 | 38,824 | 2,399 | 743 |
| poker | gain | 51.8 | 787,157 | 38 | 56.7 | 249,048 | 3,879 | 708 |
| poker | gain ratio | 61.8 | 672,892 | 45 | 65.5 | 238,691 | 4,131 | 742 |
| poker | Gini | 60.7 | 587,318 | 43 | 63.3 | 307,288 | 4,703 | 880 |

## Logistic regression: convergence

L-BFGS iterations (limit 100) and the share of folds that reached the gradient tolerance.

| dataset | iterations | converged folds |
|---|---|---|
| votes | 22 | 100% |
| tictactoe | 24 | 100% |
| splice | 46 | 100% |
| chess | 45 | 100% |
| mushroom | 22 | 100% |
| nursery | 95 | 70% |
| letter | 58 | 100% |
| krkopt | 100 | 0% |
| adult | 81 | 100% |
| connect4 | 100 | 0% |
| census | 93 | 60% |
| covertype | 100 | 0% |
| poker | 71 | 100% |
