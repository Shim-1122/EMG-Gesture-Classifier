"""Surface EMG gesture recognition: loading, preprocessing, and feature extraction."""
from .data import FS, GESTURES, N_CHANNELS, load_all, load_recording
from .features import FEATURES, extract_features
from .preprocess import envelope, highpass, make_windows
