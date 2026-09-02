import logging

import numpy as np

from AEIC.performance.types import ThrustMode, ThrustModeValues

logger = logging.getLogger(__name__)


def EI_HCCO(
    ff_eval: np.ndarray,
    x_EI: ThrustModeValues,
    ff_cal: ThrustModeValues,
    Tamb: float | np.ndarray,
    Pamb: float | np.ndarray,
    label: str = '',
) -> np.ndarray:
    """
    BFFM2 bilinear HC/CO fit to SLS data

    Parameters
    ----------
    ff_eval : ndarray, shape (n_points,)
        Fuel flows [kg/s] at which to compute xEI. Must be 1D.
    x_EI : ThrustModeValues
        Baseline emission indices [g x / kg fuel] at four calibration fuel‐flow points.
    ff_cal : ThrustModeValues
        Calibration fuel flows [kg/s] corresponding to x_EI
    Tamb : ndarray, shape (n_points,)
        Ambient temperature [K] for cruise correction.
    Pamb : ndarray, shape (n_points,)
        Ambient pressure [Pa] for cruise correction.
    label : str, optional
        Optional engine/aircraft tag included in calibration warnings,
        e.g. when ``x_EI`` or ``ff_cal`` contain non-positive values.

    Returns
    -------
    xEI : ndarray, shape (n_points,)
        The HC+CO emission index [g x / kg fuel] at each ff_eval.
    """

    # Validate inputs and match MATLAB's handling of zero calibration EIs.
    ff_eval = np.asarray(ff_eval, dtype=float)
    if ff_eval.ndim != 1:
        raise ValueError(f'Evaluation fuel flow must be one-dimensional. ({label})')
    if not np.all(np.isfinite(ff_eval)):
        raise ValueError(f'Evaluation fuel flow must be finite. ({label})')
    if np.any(ff_eval < 0.0):
        raise ValueError(f'Evaluation fuel flow must be non-negative. ({label})')

    x_EI_values = np.asarray(x_EI.as_array(), dtype=float)
    zero_EI = x_EI_values == 0.0
    if np.any(zero_EI):
        logger.warning(f'Replacing zero calibration xEI with 0.1 g/kg. ({label})')
        x_EI_values[zero_EI] = 0.1
    if not np.all(np.isfinite(x_EI_values)) or np.any(x_EI_values <= 0.0):
        raise ValueError(
            f'Calibration emission indices must be finite and positive. ({label})'
        )
    x_EI = ThrustModeValues(x_EI_values)

    ff_cal_values = np.asarray(ff_cal.as_array(), dtype=float)
    if not np.all(np.isfinite(ff_cal_values)) or np.any(ff_cal_values <= 0.0):
        raise ValueError(
            f'Calibration fuel flows must be finite and positive. ({label})'
        )
    if np.any(np.diff(ff_cal_values) <= 0.0):
        raise ValueError(
            f'Calibration fuel flows must be strictly increasing. ({label})'
        )

    try:
        Tamb = np.broadcast_to(np.asarray(Tamb, dtype=float), ff_eval.shape)
        Pamb = np.broadcast_to(np.asarray(Pamb, dtype=float), ff_eval.shape)
    except ValueError as exc:
        raise ValueError(
            'Ambient temperature and pressure must be broadcastable to fuel flow.'
            f' ({label})'
        ) from exc
    if not np.all(np.isfinite(Tamb)) or np.any(Tamb <= 0.0):
        raise ValueError(f'Ambient temperature must be finite and positive. ({label})')
    if not np.all(np.isfinite(Pamb)) or np.any(Pamb <= 0.0):
        raise ValueError(f'Ambient pressure must be finite and positive. ({label})')

    # ----------------------------------------------------------------------------
    # 1. Compute slanted‐line parameters in log10 space
    #    slope = [log10(xEI[1]) - log10(xEI[0])] / [log10(ff_cal[1]) - log10(ff_cal[0])]
    #    base_log_fuel = log10(ff_cal[0])
    #    base_log_EI   = log10(xEI[0])
    # ----------------------------------------------------------------------------
    # Prevent log10(0) by assuming calibration flows/EIs are strictly positive
    slope_num = np.log10(x_EI[ThrustMode.APPROACH]) - np.log10(x_EI[ThrustMode.IDLE])
    slope_den = np.log10(ff_cal[ThrustMode.APPROACH]) - np.log10(
        ff_cal[ThrustMode.IDLE]
    )
    if np.isclose(slope_den, 0.0):
        slope = 0.0
    else:
        slope = slope_num / slope_den

    base_log_fuel = np.log10(ff_cal[ThrustMode.IDLE])
    base_log_EI = np.log10(x_EI[ThrustMode.IDLE])

    # ----------------------------------------------------------------------------
    # 2. Compute horizontal‐line level: midpoint of logs at calibration points 2 and 3
    #    x_horzline = 0.5 * [ log10(xEI[2]) + log10(xEI[3]) ]
    # ----------------------------------------------------------------------------
    x_horzline = 0.5 * (
        np.log10(x_EI[ThrustMode.CLIMB]) + np.log10(x_EI[ThrustMode.TAKEOFF])
    )

    # ----------------------------------------------------------------------------
    # 3. Compute intersection (in log10 fuel) between slanted and horizontal segments:
    #    x_intercept =
    #      [ 2*log10(ff_cal[0])*slope + log10(xEI[2]) + log10(xEI[3]) - 2*log10(xEI[0])]
    #                  / (2 * slope) , if slope != 0
    #    If slope == 0, force intercept := log10(ff_cal[1]) to use horizontal segment
    # ----------------------------------------------------------------------------
    if np.isclose(slope, 0.0):
        x_intercept = np.log10(ff_cal[ThrustMode.APPROACH])
    else:
        numerator = (
            2.0 * np.log10(ff_cal[ThrustMode.IDLE]) * slope
            + np.log10(x_EI[ThrustMode.CLIMB])
            + np.log10(x_EI[ThrustMode.TAKEOFF])
            - 2.0 * np.log10(x_EI[ThrustMode.IDLE])
        )
        x_intercept = numerator / (2.0 * slope)

    # ----------------------------------------------------------------------------
    # 4. Enforce SAGE v1.5 rules row‐wise (here only one "row" since 1D):
    #    (a) If x_intercept > log10(ff_cal[2]), clamp it to log10(ff_cal[2]).
    #    (b) Else if x_intercept < log10(ff_cal[1]) and slope < 0:
    #        set x_horzline := log10(xEI[1]) and clamp x_intercept := log10(ff_cal[1]).
    #    (c) Else if slope >= 0: force slope=0, base_log_fuel=0, base_log_EI=x_horzline,
    #            and clamp x_intercept := log10(ff_cal[1]).
    # ----------------------------------------------------------------------------
    log_ff_cal1 = np.log10(ff_cal[ThrustMode.APPROACH])
    log_ff_cal2 = np.log10(ff_cal[ThrustMode.CLIMB])

    if x_intercept > log_ff_cal2:
        x_intercept = log_ff_cal2

    elif (x_intercept < log_ff_cal1) and (slope < 0.0):
        x_horzline = np.log10(x_EI[ThrustMode.APPROACH])
        x_intercept = log_ff_cal1

    elif slope >= 0.0:
        slope = 0.0
        base_log_fuel = 0.0
        base_log_EI = x_horzline
        x_intercept = log_ff_cal1

    # ----------------------------------------------------------------------------
    # 5. Allocate output array and compute xEI for each evaluation point
    # ----------------------------------------------------------------------------
    n_points = len(ff_eval)
    xEI_out = np.zeros(n_points, dtype=float)

    # Compute log10 of evaluation fuel flows, masking out non‐positive flows
    log_ff = np.zeros(n_points, dtype=float)
    positive_mask = ff_eval > 0.0
    log_ff[positive_mask] = np.log10(ff_eval[positive_mask])

    # Lower segment: log_ff < x_intercept
    lower_mask = positive_mask & (log_ff < x_intercept)
    # Upper segment: log_ff >= x_intercept
    upper_mask = positive_mask & (log_ff >= x_intercept)

    # Slanted‐line formula for "lower" points
    if np.any(lower_mask):
        xEI_out[lower_mask] = 10.0 ** (
            slope * (log_ff[lower_mask] - base_log_fuel) + base_log_EI
        )

    # Horizontal‐line (constant) for "upper" points
    if np.any(upper_mask):
        xEI_out[upper_mask] = 10.0**x_horzline

    # Replace any NaNs (e.g., from log10(0) → -inf) with zero
    xEI_out[np.isnan(xEI_out)] = 0.0

    # ----------------------------------------------------------------------------
    # 6. ACRP low‐thrust correction:
    #    For any ff_eval < ff_cal[0], use:
    #       xEI_acrp = xEI * [1 + (–52) * (ff_eval – ff_cal[0])]
    #    Then overwrite those points with xEI_acrp.
    # ----------------------------------------------------------------------------
    # ACRP_slope = -52.0
    # low_thrust_mask = ff_eval < ff_cal[ThrustMode.IDLE]
    # if np.any(low_thrust_mask):
    #     delta_ff = ff_eval[low_thrust_mask] - ff_cal[ThrustMode.IDLE]
    #     xEI_acrp = xEI_out[low_thrust_mask] * (1.0 + ACRP_slope * delta_ff)
    #     xEI_out[low_thrust_mask] = xEI_acrp

    # ----------------------------------------------------------------------------
    # 7. Cruise correction:
    #    Multiply entire xEI_out by (θ^3.3)/(δ^1.02),
    #    where θ = Tamb / 288.15, δ = Pamb / 101325.
    # ----------------------------------------------------------------------------
    theta_amb = Tamb / 288.15
    delta_amb = Pamb / 101325.0
    factor = (theta_amb**3.3) / (delta_amb**1.02)
    xEI_out *= factor

    if not np.all(np.isfinite(xEI_out)) or np.any(xEI_out < 0.0):
        raise ValueError(
            f'Calculated emission indices must be finite and non-negative. ({label})'
        )

    return xEI_out
