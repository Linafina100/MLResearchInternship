"""
Shared Initial_SOC truncation logic for experiment 26 -- factored out of
build_real_empa_soc_sweep_soh_range.py (where this was first written,
following experiments/22_29_sim_to_real_validation/23_empa_soc_sweep/build_real_empa_soc_sweep_dataset.py's
original technique) since identical copies had accumulated in three
build scripts (EMPA, exp07, CALCE) and a fourth was about to be added for
the continuous-discharge-truncated synthetic data generator -- one shared
copy means a fix lands everywhere at once.
"""
import numpy as np

MIN_RAW_POINTS_AFTER_TRUNCATION = 5  # need enough raw points left to
                                      # interpolate/aggregate reliably


def truncate_at_soc(t_s, voltage, cap_ah, soc_start, min_raw_points=MIN_RAW_POINTS_AFTER_TRUNCATION):
    """Keeps only the portion of a full discharge trace (real or
    synthetic) occurring after (1 - soc_start) of THIS trace's own
    realized capacity has already been delivered. Time and Capacity are
    reset to 0 at the new starting point. Does NOT discard a variant for
    failing to reach any specific voltage zone -- callers use all
    voltage bins the mutual-coverage filter lets survive, not a
    hand-picked target zone. Returns None if too few points remain."""
    total_capacity = cap_ah[-1]
    cap_threshold = (1.0 - soc_start) * total_capacity

    if cap_threshold <= 0:
        t_kept, v_kept, cap_kept = t_s, voltage, cap_ah
    else:
        t_at_thresh = np.interp(cap_threshold, cap_ah, t_s)
        v_at_thresh = np.interp(cap_threshold, cap_ah, voltage)
        mask = cap_ah > cap_threshold
        if mask.sum() < min_raw_points - 1:
            return None
        t_kept = np.concatenate([[t_at_thresh], t_s[mask]])
        v_kept = np.concatenate([[v_at_thresh], voltage[mask]])
        cap_kept = np.concatenate([[cap_threshold], cap_ah[mask]])
        t_kept = t_kept - t_kept[0]
        cap_kept = cap_kept - cap_kept[0]

    if len(t_kept) < min_raw_points:
        return None

    return t_kept, v_kept, cap_kept


def trim_to_cutoff_voltage(t_s, voltage, cap_ah, cutoff_voltage):
    """Added for experiment 32: trims a full discharge trace so it ends
    exactly where voltage first crosses down through `cutoff_voltage`,
    interpolating the exact (time, voltage, capacity) at that crossing and
    using it as the new final point. truncate_at_soc() always treats
    whatever it's handed as cap_ah[-1] = "0% SOC" (the trace's own raw
    stopping point) -- this lets a caller make "0% SOC" mean the SAME
    physical voltage across cells/datasets/chemistries that natively stop
    at different voltages, by trimming to that voltage BEFORE calling
    truncate_at_soc(), instead of each cycle's own raw stopping point.
    Returns None if the trace never reaches cutoff_voltage (an
    incomplete/shallow cycle with no basis to locate it there), or if it
    starts at or below cutoff_voltage (not a real full discharge)."""
    below = np.where(voltage <= cutoff_voltage)[0]
    if len(below) == 0 or below[0] == 0:
        return None
    i = below[0]
    v0, v1 = voltage[i - 1], voltage[i]
    frac = 0.0 if v0 == v1 else (v0 - cutoff_voltage) / (v0 - v1)
    t_cut = t_s[i - 1] + frac * (t_s[i] - t_s[i - 1])
    cap_cut = cap_ah[i - 1] + frac * (cap_ah[i] - cap_ah[i - 1])
    t_kept = np.concatenate([t_s[:i], [t_cut]])
    v_kept = np.concatenate([voltage[:i], [cutoff_voltage]])
    cap_kept = np.concatenate([cap_ah[:i], [cap_cut]])
    return t_kept, v_kept, cap_kept
