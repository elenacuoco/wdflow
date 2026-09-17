"""Writes sosfiltfilt_scipy117.npz, the reference test_filtering.py holds
wdf.filtering to. It has to run under scipy 1.17, whose sosfiltfilt is the one
the golden outputs were made with:

    python make_sosfiltfilt_reference.py sosfiltfilt_scipy117.npz
"""
import sys, numpy as np, scipy
from scipy.signal import cheby2, butter, sosfiltfilt, sosfilt_zi
assert scipy.__version__.startswith("1.17"), scipy.__version__
out = {}
cases = {
    # the pipeline's band-pass, at the order the tests configure and the default
    "cheby2_o4": cheby2(4, 60.0, [4.0, 0.9 * (4096 / 2 / 4)], fs=4096, btype="bandpass", output="sos"),
    "cheby2_o10": cheby2(10, 60.0, [4.0, 0.9 * (4096 / 2 / 4)], fs=4096, btype="bandpass", output="sos"),
    # the mock data's low- and high-pass
    "butter_lp": butter(8, 0.45 * 0.5, btype="lowpass", output="sos"),
    "butter_hp": butter(4, 0.01, btype="highpass", output="sos"),
}
x = np.random.default_rng(20260917).normal(size=1000)
for name, sos in cases.items():
    out[f"{name}_sos"] = sos
    out[f"{name}_zi"] = sosfilt_zi(sos)
    out[f"{name}_y"] = sosfiltfilt(sos, x)
    out[f"{name}_y_pad0"] = sosfiltfilt(sos, x, padlen=0)
    out[f"{name}_y_pad50"] = sosfiltfilt(sos, x, padlen=50)
out["x"] = x
np.savez_compressed(sys.argv[1], **out)
print("scipy", scipy.__version__, "->", sys.argv[1], len(out), "arrays")
