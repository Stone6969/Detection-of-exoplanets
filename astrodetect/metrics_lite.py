"""Small metric helpers used by the notebooks' plots."""
import numpy as np

from .detrend import segments


def cdpp(time, flux, hours=1.0):
    """Scatter of running means over `hours` (ppm) - noise a transit competes with."""
    vals = []
    for s in segments(time):
        y = flux[s]
        y = y[np.isfinite(y)]
        if y.size < 10:
            continue
        n = max(1, int(round(hours / 24 / np.median(np.diff(time[s])))))
        if y.size > 3 * n:
            c = np.cumsum(np.r_[0, y])
            vals.append((c[n:] - c[:-n]) / n)
    return float(np.std(np.concatenate(vals)) * 1e6) if vals else float("nan")
