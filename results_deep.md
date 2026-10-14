| Model | Input | Within-Subject (%) | Cross-Subject (%) |
|---|---|---|---|
| LDA (Hudgins features) | 40 Hudgins features | 92.9 ± 5.3 | 80.1 ± 5.2 |
| 1D CNN (PyTorch) | filtered window, 8 x 40 samples | 90.6 ± 5.6 | 81.4 ± 4.5 |

CNN: 25,190 parameters, 0.39 ms per window on one CPU thread (window step is 50 ms).
