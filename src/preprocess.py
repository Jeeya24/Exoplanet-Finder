"""Turn a light curve into fixed-size global and local views for the model.

Works for Kepler and TESS: pass time, period, t0 and duration in the same units (days).
"""

import warnings

import numpy as np

# Hide lightkurve's harmless "tpfmodel needs oktopus" import warning (we don't use tpfmodel)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message=".*tpfmodel submodule.*")
    import lightkurve as lk

warnings.filterwarnings("ignore", module="lightkurve")


def _bin_median(phase, flux, lo, hi, n_bins):
    """Median flux in n_bins equal bins between lo and hi; empty bins are interpolated."""
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    idx = np.digitize(phase, edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    idx, flux = idx[keep], flux[keep]

    binned = np.full(n_bins, np.nan)
    order = np.argsort(idx)
    idx, flux = idx[order], flux[order]
    starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
    for s, e in zip(starts, np.r_[starts[1:], len(idx)]):
        binned[idx[s]] = np.median(flux[s:e])

    empty = np.isnan(binned)
    if empty.all():
        raise ValueError("no points in range")
    if empty.any():
        binned[empty] = np.interp(centres[empty], centres[~empty], binned[~empty])
    return binned, centres, int(empty.sum())


def prepare_views(time, flux, period, t0, duration, n_bins=201,
                  window_days=2.0, sigma_upper=3.0, return_steps=False):
    """Clean, fold and bin one light curve.

    time, t0, period, duration: days (same clock for time and t0).
    Returns (global_view, local_view), each n_bins long, with normal brightness = 0.
    With return_steps=True also returns a dict of the intermediate steps (for plotting).
    """
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    ok = np.isfinite(time) & np.isfinite(flux)
    lc = lk.LightCurve(time=time[ok], flux=flux[ok])

    def phase_of(t):
        return (t - t0 + 0.5 * period) % period - 0.5 * period

    # 1. Flatten: window in days (so it also works for TESS's 2-min cadence), transits masked
    cadence = np.median(np.diff(lc.time.value))
    window = max(window_days, 3 * duration)
    window_length = int(window / cadence) | 1          # must be odd
    # Mask +-1.5 durations, but never more than half the orbit (very long "transits"
    # would otherwise mask every point and leave nothing to fit)
    half_mask = min(1.5 * duration, 0.25 * period)
    in_transit = np.abs(phase_of(lc.time.value)) < half_mask
    flat, trend = lc.flatten(window_length=window_length, mask=in_transit, return_trend=True)

    # 2. Remove high outliers only (transits are low points)
    clean, removed = flat.remove_outliers(sigma_upper=sigma_upper, sigma_lower=np.inf, return_mask=True)

    # 3. Fold
    phase = phase_of(clean.time.value)
    f = clean.flux.value

    # 4. Bin: global = whole orbit, local = +-3 transit durations
    global_view, g_centres, _ = _bin_median(phase, f, -period / 2, period / 2, n_bins)
    local_view, l_centres, n_empty = _bin_median(phase, f, -3 * duration, 3 * duration, n_bins)

    # 5. Normal brightness = 0 (median of out-of-transit points)
    baseline = np.median(f[np.abs(phase) > half_mask])
    global_view, local_view = global_view - baseline, local_view - baseline

    if not return_steps:
        return global_view, local_view
    steps = dict(raw=lc, trend=trend, flat=flat, clean=clean, removed=removed,
                 phase=phase, global_x=g_centres, local_x=l_centres, local_empty_bins=n_empty)
    return global_view, local_view, steps
