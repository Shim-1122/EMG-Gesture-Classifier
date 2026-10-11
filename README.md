# EMG Gesture Classifier

![Tests](https://github.com/Shim-1122/EMG-Gesture-Classifier/actions/workflows/tests.yml/badge.svg)

Classifies 6 hand and wrist gestures from 8-channel forearm surface EMG using Hudgins time-domain features with LDA, SVM, and random forest classifiers. All filtering is causal, so the pipeline uses only data a real-time controller would have.

![Pipeline Stages](figures/pipeline_stages.png)

## Results

36 subjects, 2 sessions each, 200 ms windows with a 50 ms step.

| Classifier | Within-Subject (%) | Cross-Subject (%) |
|---|---|---|
| LDA | 92.9 ± 5.3 | 80.1 ± 5.2 |
| SVM (RBF) | 91.4 ± 6.6 | 80.7 ± 2.7 |
| Random Forest | 91.5 ± 6.4 | 81.8 ± 3.2 |

- **Within-subject** (train on session 1, test on session 2): LDA scores highest and is the cheapest at inference (one matrix multiply per window), which is why it is the standard baseline in myoelectric control.
- **Cross-subject** (6 fixed subject folds; no subject appears in both train and test): accuracy drops 10 to 13 points due to differences in anatomy, electrode placement, and skin impedance. Clinical systems calibrate per user for this reason.
- **Window length** sets the minimum decision delay. LDA accuracy rises from 88.1% at 50 ms to 93.4% at 300 ms. About 300 ms of total controller delay is the commonly cited usability limit (Englehart and Hudgins, 2003).

![Accuracy Summary](figures/accuracy_summary.png)

Largest confusions: radial deviation predicted as wrist extension (13%) and ulnar deviation predicted as wrist flexion (9%). Each pair shares prime movers: extensor carpi radialis longus and brevis drive both radial deviation and extension; flexor carpi ulnaris drives both ulnar deviation and flexion.

![Confusion Matrix](figures/confusion_matrix.png)

## Method

1. **Resample** each recording onto a uniform 200 Hz grid (raw timestamps are irregular). EMG is linearly interpolated; labels use zero-order hold.
2. **High-pass filter** at 5 Hz (4th-order Butterworth, causal `sosfilt`) to remove DC offset and motion artifact.
3. **Window** into 200 ms segments every 50 ms. Windows spanning a gesture transition or unlabeled data are discarded.
4. **Extract features** per channel: log mean absolute value, log RMS, log waveform length, zero crossings, slope sign changes. 5 features x 8 channels = 40 per window, computed for all windows in one vectorized call.
5. **Classify** with LDA, RBF SVM, and random forest (scikit-learn), with standardization for LDA and SVM.

## Evaluation Design

Overlapping windows from one recording are highly correlated. A random window-level split puts near-duplicates in both train and test and inflates accuracy, so this project splits only by session or by subject.

## Limitations

The Myo stream in this dataset updates at about 100 Hz (roughly 90% of rows in the raw files repeat the previous sample). Surface EMG extends to about 450 Hz, so only low-frequency content is captured. Absolute accuracy is not comparable to clinical systems sampling at 1 to 2 kHz; comparisons between models and evaluation schemes remain valid.

## Usage

```bash
git clone https://github.com/Shim-1122/EMG-Gesture-Classifier.git
cd EMG-Gesture-Classifier
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py    # downloads the dataset (~17 MB) on first run; ~4 min total
pytest                    # no dataset needed
```

`run_pipeline.py` writes `figures/` and `results.md` (including the full window-length sweep).

## Project Structure

```
emg/
  data.py          download, load, resample to a uniform grid
  preprocess.py    causal filters, linear envelope, windowing
  features.py      Hudgins time-domain features (vectorized)
run_pipeline.py    evaluation, figures, results.md
tests/
```

## Tests

- Features match closed-form values for a pure sine: MAV = 2A/π, RMS = A/√2, waveform length = 4A per cycle, 2 zero crossings and 2 slope sign changes per cycle
- Thresholds reject low-amplitude noise; flat windows produce finite features
- High-pass filter removes DC and passes a 40 Hz tone; envelope tracks a burst
- Windowing never mixes labels; resampling interpolates EMG and holds labels

## Data and License

Krilova, N., Kastalskiy, I., Kazantsev, V., Makarov, V., and Lobov, S. (2018). *EMG Data for Gestures* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZP5C. CC BY 4.0.

Code: MIT License.
