# Benchmark summary

GPU: NVIDIA GeForce RTX 3060 Ti (38 SMs, 8.0 GB, CC 8.6)

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
| connect4 | 67,557 | 75.8 | 78.0 | 79.1 | **80.9** | 77.1 | 78.7 | 72.2 | **80.9** | 75.7 |
| census | 299,285 | 93.3 | 94.8 | 93.7 | **95.3** | 93.4 | 94.8 | 85.6 | 94.3 | **95.3** |
| covertype | 581,012 | **95.3** | **95.3** | 94.9 | 94.9 | 94.3 | 94.4 | 65.7 | 93.9 | 72.4 |
| poker | 1,025,010 | 51.8 | 56.7 | 61.8 | **65.5** | 60.7 | 63.3 | 50.1 | 53.4 | 50.1 |

## Time per fold (ms): training + prediction, CPU vs GPU

Mean over the folds. *: one stratified holdout split instead of 10 folds.

| dataset | decision tree (gain) CPU | GPU | speedup | naive Bayes CPU | GPU | speedup | kNN (k=5) CPU | GPU | speedup | logistic regression CPU | GPU | speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| votes | 0.07 | 2.92 | 0.02x | 0.01 | 1.09 | 0.01x | 0.08 | 1.21 | 0.07x | 1.05 | 3.77 | 0.28x |
| tictactoe | 0.19 | 3.09 | 0.06x | 0.01 | 1.33 | 0.01x | 0.22 | 1.34 | 0.16x | 2.14 | 4.14 | 0.52x |
| splice | 1.98 | 3.90 | 0.51x | 0.21 | 1.46 | 0.15x | 6.27 | 6.54 | 0.96x | 42.5 | 12.8 | 3.3x |
| chess | 1.06 | 4.13 | 0.26x | 0.20 | 1.21 | 0.17x | 4.34 | 3.78 | 1.1x | 28.4 | 9.73 | 2.9x |
| mushroom | 0.69 | 2.29 | 0.30x | 0.27 | 1.49 | 0.18x | 18.6 | 3.91 | 4.8x | 22.3 | 6.48 | 3.4x |
| nursery | 1.12 | 3.45 | 0.32x | 0.13 | 1.19 | 0.11x | 23.3 | 2.82 | 8.3x | 122 | 21.5 | 5.7x |
| letter | 21.3 | 15.5 | 1.4x | 0.99 | 2.62 | 0.38x | 88.8 | 14.5 | 6.1x | 479 | 45.9 | 10x |
| krkopt | 11.1 | 11.7 | 0.94x | 0.61 | 1.99 | 0.31x | 102 | 6.09 | 17x | 648 | 46.5 | 14x |
| adult | 26.7 | 17.4 | 1.5x | 1.53 | 2.47 | 0.62x | 507 | 23.8 | 21x | 383 | 28.4 | 13x |
| connect4 | 68.3 | 20.7 | 3.3x | 4.25 | 4.62 | 0.92x | 2,610 | 498 | 5.2x | 1,509 | 90.2 | 17x |
| census | 375 | 58.4 | 6.4x | 22.7 | 14.8 | 1.5x | 80,899* | 4,553* | 18x | 5,413 | 394 | 14x |
| covertype | 673 | 112 | 6.0x | 21.2 | 15.1 | 1.4x | 109,239* | 2,595* | 42x | 7,685 | 874 | 8.8x |
| poker | 1,166 | 191 | 6.1x | 19.3 | 19.2 | 1.0x | 159,720* | 3,408* | 47x | 13,213 | 1,700 | 7.8x |

## Training and prediction time per fold (ms)

| dataset | classifier | split | CPU train | CPU predict | GPU train | GPU predict |
|---|---|---|---|---|---|---|
| votes | tree: gain | 10-fold | 0.07 | 0.00 | 2.42 | 0.50 |
| votes | tree: gain, pruned | 10-fold | 0.06 | 0.00 | 2.47 | 0.51 |
| votes | tree: gain ratio | 10-fold | 0.07 | 0.00 | 2.40 | 0.46 |
| votes | tree: gain ratio, pruned | 10-fold | 0.07 | 0.00 | 2.48 | 0.42 |
| votes | tree: Gini | 10-fold | 0.09 | 0.00 | 3.36 | 0.54 |
| votes | tree: Gini, pruned | 10-fold | 0.10 | 0.00 | 3.13 | 0.48 |
| votes | naive Bayes | 10-fold | 0.01 | 0.00 | 0.62 | 0.47 |
| votes | kNN (k=5) | 10-fold | 0.02 | 0.06 | 0.31 | 0.90 |
| votes | logistic regression | 10-fold | 1.04 | 0.00 | 3.30 | 0.47 |
| tictactoe | tree: gain | 10-fold | 0.19 | 0.00 | 2.60 | 0.49 |
| tictactoe | tree: gain, pruned | 10-fold | 0.17 | 0.00 | 2.57 | 0.50 |
| tictactoe | tree: gain ratio | 10-fold | 0.17 | 0.00 | 2.73 | 0.48 |
| tictactoe | tree: gain ratio, pruned | 10-fold | 0.17 | 0.00 | 2.48 | 0.49 |
| tictactoe | tree: Gini | 10-fold | 0.17 | 0.01 | 4.00 | 0.54 |
| tictactoe | tree: Gini, pruned | 10-fold | 0.20 | 0.00 | 4.00 | 0.50 |
| tictactoe | naive Bayes | 10-fold | 0.01 | 0.00 | 0.77 | 0.56 |
| tictactoe | kNN (k=5) | 10-fold | 0.02 | 0.19 | 0.47 | 0.87 |
| tictactoe | logistic regression | 10-fold | 2.13 | 0.01 | 3.66 | 0.47 |
| splice | tree: gain | 10-fold | 1.97 | 0.01 | 3.43 | 0.46 |
| splice | tree: gain, pruned | 10-fold | 2.00 | 0.01 | 3.29 | 0.49 |
| splice | tree: gain ratio | 10-fold | 2.14 | 0.01 | 3.61 | 0.48 |
| splice | tree: gain ratio, pruned | 10-fold | 2.09 | 0.01 | 3.54 | 0.46 |
| splice | tree: Gini | 10-fold | 3.23 | 0.02 | 4.93 | 0.50 |
| splice | tree: Gini, pruned | 10-fold | 3.27 | 0.01 | 5.29 | 0.54 |
| splice | naive Bayes | 10-fold | 0.17 | 0.04 | 0.86 | 0.60 |
| splice | kNN (k=5) | 10-fold | 0.36 | 5.91 | 1.60 | 4.93 |
| splice | logistic regression | 10-fold | 42.3 | 0.13 | 12.1 | 0.70 |
| chess | tree: gain | 10-fold | 1.05 | 0.01 | 3.66 | 0.47 |
| chess | tree: gain, pruned | 10-fold | 1.01 | 0.01 | 3.81 | 0.49 |
| chess | tree: gain ratio | 10-fold | 1.07 | 0.01 | 3.71 | 0.45 |
| chess | tree: gain ratio, pruned | 10-fold | 1.09 | 0.01 | 3.70 | 0.50 |
| chess | tree: Gini | 10-fold | 1.25 | 0.01 | 4.70 | 0.52 |
| chess | tree: Gini, pruned | 10-fold | 1.23 | 0.01 | 4.22 | 0.51 |
| chess | naive Bayes | 10-fold | 0.18 | 0.02 | 0.72 | 0.49 |
| chess | kNN (k=5) | 10-fold | 0.27 | 4.08 | 1.04 | 2.74 |
| chess | logistic regression | 10-fold | 28.3 | 0.07 | 9.10 | 0.63 |
| mushroom | tree: gain | 10-fold | 0.68 | 0.01 | 1.83 | 0.46 |
| mushroom | tree: gain, pruned | 10-fold | 0.68 | 0.01 | 1.72 | 0.43 |
| mushroom | tree: gain ratio | 10-fold | 0.64 | 0.01 | 1.88 | 0.45 |
| mushroom | tree: gain ratio, pruned | 10-fold | 0.63 | 0.01 | 2.07 | 0.46 |
| mushroom | tree: Gini | 10-fold | 1.42 | 0.02 | 3.20 | 0.52 |
| mushroom | tree: Gini, pruned | 10-fold | 1.49 | 0.02 | 2.97 | 0.49 |
| mushroom | naive Bayes | 10-fold | 0.23 | 0.04 | 0.87 | 0.62 |
| mushroom | kNN (k=5) | 10-fold | 0.37 | 18.2 | 1.48 | 2.43 |
| mushroom | logistic regression | 10-fold | 22.2 | 0.13 | 5.92 | 0.56 |
| nursery | tree: gain | 10-fold | 1.08 | 0.04 | 2.97 | 0.48 |
| nursery | tree: gain, pruned | 10-fold | 1.08 | 0.03 | 3.21 | 0.49 |
| nursery | tree: gain ratio | 10-fold | 1.09 | 0.04 | 2.94 | 0.50 |
| nursery | tree: gain ratio, pruned | 10-fold | 1.08 | 0.03 | 3.05 | 0.46 |
| nursery | tree: Gini | 10-fold | 1.33 | 0.07 | 5.04 | 0.55 |
| nursery | tree: Gini, pruned | 10-fold | 1.37 | 0.07 | 4.89 | 0.60 |
| nursery | naive Bayes | 10-fold | 0.10 | 0.03 | 0.71 | 0.48 |
| nursery | kNN (k=5) | 10-fold | 0.23 | 23.1 | 0.84 | 1.97 |
| nursery | logistic regression | 10-fold | 122 | 0.06 | 20.8 | 0.66 |
| letter | tree: gain | 10-fold | 21.2 | 0.16 | 15.0 | 0.55 |
| letter | tree: gain, pruned | 10-fold | 21.3 | 0.16 | 14.8 | 0.48 |
| letter | tree: gain ratio | 10-fold | 23.0 | 0.20 | 17.3 | 0.58 |
| letter | tree: gain ratio, pruned | 10-fold | 23.1 | 0.18 | 16.7 | 0.54 |
| letter | tree: Gini | 10-fold | 24.0 | 0.21 | 19.9 | 0.62 |
| letter | tree: Gini, pruned | 10-fold | 24.3 | 0.19 | 19.4 | 0.58 |
| letter | naive Bayes | 10-fold | 0.55 | 0.44 | 1.66 | 0.97 |
| letter | kNN (k=5) | 10-fold | 1.02 | 87.8 | 3.14 | 11.4 |
| letter | logistic regression | 10-fold | 479 | 0.37 | 45.0 | 0.84 |
| krkopt | tree: gain | 10-fold | 10.8 | 0.21 | 11.2 | 0.52 |
| krkopt | tree: gain, pruned | 10-fold | 11.5 | 0.20 | 12.5 | 0.60 |
| krkopt | tree: gain ratio | 10-fold | 10.2 | 0.21 | 12.0 | 0.55 |
| krkopt | tree: gain ratio, pruned | 10-fold | 11.4 | 0.20 | 13.1 | 0.59 |
| krkopt | tree: Gini | 10-fold | 13.1 | 0.37 | 28.0 | 0.61 |
| krkopt | tree: Gini, pruned | 10-fold | 14.0 | 0.34 | 27.5 | 0.62 |
| krkopt | naive Bayes | 10-fold | 0.47 | 0.14 | 1.46 | 0.53 |
| krkopt | kNN (k=5) | 10-fold | 0.46 | 101 | 1.39 | 4.71 |
| krkopt | logistic regression | 10-fold | 648 | 0.17 | 45.7 | 0.75 |
| adult | tree: gain | 10-fold | 26.2 | 0.50 | 16.8 | 0.67 |
| adult | tree: gain, pruned | 10-fold | 27.5 | 0.16 | 17.3 | 0.64 |
| adult | tree: gain ratio | 10-fold | 34.4 | 0.71 | 20.1 | 0.64 |
| adult | tree: gain ratio, pruned | 10-fold | 32.6 | 0.29 | 20.6 | 0.64 |
| adult | tree: Gini | 10-fold | 42.8 | 0.99 | 25.9 | 0.64 |
| adult | tree: Gini, pruned | 10-fold | 41.8 | 0.36 | 25.9 | 0.66 |
| adult | naive Bayes | 10-fold | 1.29 | 0.24 | 1.95 | 0.53 |
| adult | kNN (k=5) | 10-fold | 1.54 | 506 | 5.41 | 18.4 |
| adult | logistic regression | 10-fold | 382 | 0.45 | 27.6 | 0.81 |
| connect4 | tree: gain | 10-fold | 67.5 | 0.82 | 19.8 | 0.90 |
| connect4 | tree: gain, pruned | 10-fold | 68.8 | 0.50 | 21.5 | 0.72 |
| connect4 | tree: gain ratio | 10-fold | 80.1 | 0.88 | 22.5 | 0.84 |
| connect4 | tree: gain ratio, pruned | 10-fold | 89.0 | 0.67 | 27.5 | 0.99 |
| connect4 | tree: Gini | 10-fold | 120 | 1.62 | 29.9 | 1.28 |
| connect4 | tree: Gini, pruned | 10-fold | 118 | 1.24 | 29.4 | 0.80 |
| connect4 | naive Bayes | 10-fold | 3.55 | 0.70 | 3.90 | 0.71 |
| connect4 | kNN (k=5) | 10-fold | 5.73 | 2,604 | 30.4 | 467 |
| connect4 | logistic regression | 10-fold | 1,506 | 2.39 | 88.4 | 1.82 |
| census | tree: gain | 10-fold | 372 | 2.99 | 56.3 | 2.16 |
| census | tree: gain, pruned | 10-fold | 364 | 0.74 | 54.4 | 1.49 |
| census | tree: gain ratio | 10-fold | 490 | 4.00 | 68.0 | 2.11 |
| census | tree: gain ratio, pruned | 10-fold | 489 | 1.10 | 79.2 | 1.62 |
| census | tree: Gini | 10-fold | 834 | 6.45 | 104 | 1.57 |
| census | tree: Gini, pruned | 10-fold | 849 | 2.12 | 106 | 1.58 |
| census | naive Bayes | 10-fold | 19.4 | 3.31 | 13.2 | 1.52 |
| census | kNN (k=5) | 10-fold | - | - | 185 | 4,380 |
| census | kNN (k=5) | holdout | 27.0 | 80,872 | 184 | 4,368 |
| census | logistic regression | 10-fold | 5,402 | 11.7 | 388 | 5.74 |
| covertype | tree: gain | 10-fold | 669 | 4.82 | 111 | 1.24 |
| covertype | tree: gain, pruned | 10-fold | 666 | 4.82 | 113 | 1.32 |
| covertype | tree: gain ratio | 10-fold | 718 | 6.26 | 123 | 1.24 |
| covertype | tree: gain ratio, pruned | 10-fold | 738 | 6.32 | 125 | 1.37 |
| covertype | tree: Gini | 10-fold | 858 | 8.05 | 186 | 1.65 |
| covertype | tree: Gini, pruned | 10-fold | 863 | 7.92 | 188 | 1.57 |
| covertype | naive Bayes | 10-fold | 18.0 | 3.20 | 13.2 | 1.86 |
| covertype | kNN (k=5) | 10-fold | - | - | 69.9 | 2,539 |
| covertype | kNN (k=5) | holdout | 20.3 | 109,219 | 70.8 | 2,524 |
| covertype | logistic regression | 10-fold | 7,680 | 4.79 | 870 | 4.24 |
| poker | tree: gain | 10-fold | 1,139 | 27.3 | 186 | 5.28 |
| poker | tree: gain, pruned | 10-fold | 1,225 | 20.2 | 224 | 3.26 |
| poker | tree: gain ratio | 10-fold | 1,190 | 31.2 | 179 | 4.96 |
| poker | tree: gain ratio, pruned | 10-fold | 1,210 | 24.9 | 203 | 3.08 |
| poker | tree: Gini | 10-fold | 1,419 | 45.0 | 199 | 4.70 |
| poker | tree: Gini, pruned | 10-fold | 1,440 | 44.3 | 223 | 3.37 |
| poker | naive Bayes | 10-fold | 14.2 | 5.10 | 17.0 | 2.27 |
| poker | kNN (k=5) | 10-fold | - | - | 90.9 | 6,343 |
| poker | kNN (k=5) | holdout | 31.9 | 159,688 | 95.5 | 3,312 |
| poker | logistic regression | 10-fold | 13,205 | 8.20 | 1,695 | 5.45 |

## Peak memory (MB)

Host: peak resident memory of the CPU program (includes the loaded table). Device: peak GPU memory allocated by the CUDA program.

| dataset | decision tree (gain) host | device | naive Bayes host | device | kNN (k=5) host | device | logistic regression host | device |
|---|---|---|---|---|---|---|---|---|
| votes | 6 | 0 | 6 | 0 | 6 | 0 | 6 | 0 |
| tictactoe | 6 | 0 | 6 | 0 | 6 | 0 | 6 | 0 |
| splice | 9 | 1 | 8 | 1 | 10 | 1 | 10 | 2 |
| chess | 8 | 1 | 7 | 0 | 8 | 0 | 8 | 1 |
| mushroom | 8 | 1 | 8 | 1 | 9 | 1 | 9 | 2 |
| nursery | 9 | 1 | 9 | 0 | 9 | 1 | 9 | 2 |
| letter | 12 | 12 | 10 | 2 | 13 | 3 | 12 | 7 |
| krkopt | 14 | 9 | 9 | 1 | 11 | 4 | 10 | 6 |
| adult | 18 | 9 | 16 | 2 | 19 | 11 | 19 | 8 |
| connect4 | 40 | 21 | 34 | 10 | 54 | 27 | 51 | 32 |
| census | 247 | 73 | 247 | 42 | 247 | 65 | 247 | 131 |
| covertype | 115 | 278 | 91 | 28 | 128 | 91 | 139 | 107 |
| poker | 230 | 380 | 141 | 42 | 190 | 118 | 212 | 190 |

## Decision trees: size and training time

Nodes and depth are means over the folds; time is CPU training per fold.

| dataset | measure | accuracy % | nodes | depth | pruned accuracy % | pruned nodes | CPU train ms | GPU train ms |
|---|---|---|---|---|---|---|---|---|
| votes | gain | 93.1 | 66 | 7 | 94.2 | 18 | 0.07 | 2.42 |
| votes | gain ratio | 94.3 | 71 | 8 | 95.2 | 18 | 0.07 | 2.40 |
| votes | Gini | 94.2 | 48 | 9 | 94.9 | 15 | 0.09 | 3.36 |
| tictactoe | gain | 86.2 | 335 | 7 | 85.7 | 170 | 0.19 | 2.60 |
| tictactoe | gain ratio | 86.9 | 328 | 7 | 85.7 | 178 | 0.17 | 2.73 |
| tictactoe | Gini | 94.3 | 134 | 12 | 94.6 | 90 | 0.17 | 4.00 |
| splice | gain | 91.2 | 658 | 9 | 93.7 | 223 | 1.97 | 3.43 |
| splice | gain ratio | 91.8 | 698 | 10 | 94.1 | 285 | 2.14 | 3.61 |
| splice | Gini | 92.8 | 251 | 12 | 94.0 | 148 | 3.23 | 4.93 |
| chess | gain | 99.7 | 93 | 15 | 99.6 | 63 | 1.05 | 3.66 |
| chess | gain ratio | 99.7 | 91 | 16 | 99.6 | 59 | 1.07 | 3.71 |
| chess | Gini | 99.7 | 101 | 15 | 99.4 | 59 | 1.25 | 4.70 |
| mushroom | gain | 100.0 | 38 | 4 | 100.0 | 38 | 0.68 | 1.83 |
| mushroom | gain ratio | 100.0 | 29 | 5 | 100.0 | 29 | 0.64 | 1.88 |
| mushroom | Gini | 100.0 | 19 | 8 | 100.0 | 18 | 1.42 | 3.20 |
| nursery | gain | 98.9 | 1,140 | 8 | 97.0 | 508 | 1.08 | 2.97 |
| nursery | gain ratio | 98.8 | 1,139 | 8 | 97.1 | 506 | 1.09 | 2.94 |
| nursery | Gini | 99.9 | 397 | 13 | 99.5 | 325 | 1.33 | 5.04 |
| letter | gain | 88.2 | 3,919 | 20 | 88.3 | 3,550 | 21.2 | 15.0 |
| letter | gain ratio | 87.5 | 3,977 | 27 | 87.7 | 3,441 | 23.0 | 17.3 |
| letter | Gini | 87.7 | 4,190 | 29 | 87.7 | 3,690 | 24.0 | 19.9 |
| krkopt | gain | 72.8 | 19,068 | 16 | 71.4 | 11,987 | 10.8 | 11.2 |
| krkopt | gain ratio | 72.7 | 19,117 | 17 | 71.2 | 11,880 | 10.2 | 12.0 |
| krkopt | Gini | 82.0 | 13,749 | 22 | 80.6 | 8,592 | 13.1 | 28.0 |
| adult | gain | 81.7 | 24,628 | 33 | 85.1 | 3,819 | 26.2 | 16.8 |
| adult | gain ratio | 82.3 | 26,438 | 42 | 85.8 | 3,079 | 34.4 | 20.1 |
| adult | Gini | 82.2 | 14,923 | 44 | 85.8 | 2,371 | 42.8 | 25.9 |
| connect4 | gain | 75.8 | 37,725 | 20 | 78.0 | 11,112 | 67.5 | 19.8 |
| connect4 | gain ratio | 79.1 | 33,978 | 24 | 80.9 | 11,147 | 80.1 | 22.5 |
| connect4 | Gini | 77.1 | 27,630 | 36 | 78.7 | 13,534 | 120 | 29.9 |
| census | gain | 93.3 | 102,415 | 32 | 94.8 | 9,718 | 372 | 56.3 |
| census | gain ratio | 93.7 | 109,146 | 40 | 95.3 | 8,803 | 490 | 68.0 |
| census | Gini | 93.4 | 27,518 | 55 | 94.8 | 8,198 | 834 | 104 |
| covertype | gain | 95.3 | 44,186 | 32 | 95.3 | 34,956 | 669 | 111 |
| covertype | gain ratio | 94.9 | 50,778 | 47 | 94.9 | 39,401 | 718 | 123 |
| covertype | Gini | 94.3 | 54,270 | 37 | 94.4 | 38,824 | 858 | 186 |
| poker | gain | 51.8 | 787,157 | 38 | 56.7 | 249,048 | 1,139 | 186 |
| poker | gain ratio | 61.8 | 672,892 | 45 | 65.5 | 238,691 | 1,190 | 179 |
| poker | Gini | 60.7 | 587,318 | 43 | 63.3 | 307,288 | 1,419 | 199 |

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
