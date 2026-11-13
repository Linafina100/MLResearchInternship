"""
Shared Initial_SOC truncation logic for UU pipeline.
"""
import numpy as np

MIN_RAW_POINTS_AFTER_TRUNCATION = 5


def truncate_at_soc(t_s, voltage, cap_ah, soc_start, min_raw_points=MIN_RAW_POINTS_AFTER_TRUNCATION):
    """Keeps only the portion of a full discharge trace occurring after
    (1 - soc_start) of this trace's own realized capacity has already been delivered.
    Time and Capacity are reset to 0 at the new starting point. Returns None if too few points remain."""
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