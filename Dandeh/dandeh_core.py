import numpy as np


def gdh_anderson(T):
    """
    Compute Growing Degree Hours (GDH) for one hour temperature T (°C).
    Uses Anderson et al. (1986) formulation.
    """

    # No contribution outside the valid range
    if T < Tb or T > Tc:
        return 0.0

    # Region 1: Tb <= T < Tu   (cosine increase)
    if Tb <= T < Tu:
        return F * ((Tu - Tb) / 2.0) * (1.0 + cos(pi + pi * (T - Tb) / (Tu - Tb)))

    # Region 2: Tu <= T <= Tc   (cosine decrease)
    if Tu <= T <= Tc:
        return F * (
            (Tu - Tb) * (1.0 + cos((pi / 2.0) + (pi / 2.0) * (T - Tu) / (Tc - Tu)))
        )

    return 0.0


import math
import numpy as np
from typing import Tuple


def sunrise_sunset(
    doy: int,
    lat: float,
) -> Tuple[float, float]:
    """
    Compute sunrise and sunset hours for a given day of year and latitude.

    Parameters
    ----------
    doy : int
        Day of year (1–365 or 366).
    lat : float
        Latitude in degrees (positive north, negative south).

    Returns
    -------
    tuple[float, float]
        Sunrise hour and sunset hour (decimal hours, 0–24).
    """
    gamma = 2 * math.pi / 365 * (doy - 1)
    delta = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    # convert to radians for trig
    lat_rad = math.radians(lat)
    cos_wo = (math.sin(math.radians(-0.8333)) - math.sin(lat_rad) * math.sin(delta)) / (
        math.cos(lat_rad) * math.cos(delta)
    )
    cos_wo = np.clip(cos_wo, -1, 1)  # avoid domain errors
    omega = math.degrees(math.acos(cos_wo)) / 15  # hour angle to hours
    sunrise = 12 - omega
    sunset = 12 + omega
    return sunrise, sunset


def temp_after_sunrise(hour, tmin, tmax, sunrise, sunset):
    """Temperature between sunrise and sunset (daytime)."""
    return (tmax - tmin) * np.sin(
        np.pi * (hour - sunrise) / (sunset - sunrise + 4)
    ) + tmin


def temp_after_sunset(hour, tmin, tmax, tmin_next, sunrise, sunset, sunrise_next):
    """Temperature after sunset until next sunrise (nighttime cooling)."""
    t_sunset = temp_after_sunrise(sunset, tmin, tmax, sunrise, sunset)
    denom = np.log(24 - (sunset - sunrise_next) + 1)
    # clamp to avoid invalid log
    hour = np.clip(hour, sunset + 1e-6, 24)
    return t_sunset - ((t_sunset - tmin_next) / denom) * np.log(hour - sunset + 1)


def hourly_temp(
    hour: float,
    tmin: float,
    tmax: float,
    tmin_prev: float,
    tmax_prev: float,
    tmin_next: float,
    sunrise: float,
    sunset: float,
    sunrise_prev: float,
    sunset_prev: float,
    sunrise_next: float,
) -> float:
    """
    Compute hourly air temperature over a full 24-hour period,
    using previous and next day values for nighttime portions.

    Parameters
    ----------
    hour : float
        Hour of day (0–24).
    tmin : float
        Current day's minimum temperature.
    tmax : float
        Current day's maximum temperature.
    tmin_prev : float
        Previous day's minimum temperature.
    tmax_prev : float
        Previous day's maximum temperature.
    tmin_next : float
        Next day's minimum temperature.
    sunrise : float
        Current day's sunrise hour.
    sunset : float
        Current day's sunset hour.
    sunrise_prev : float
        Previous day's sunrise hour.
    sunset_prev : float
        Previous day's sunset hour.
    sunrise_next : float
        Next day's sunrise hour.

    Returns
    -------
    float
        Estimated air temperature at the given hour.
    """

    if sunrise <= hour <= sunset:
        # Daytime
        temp = temp_after_sunrise(hour, tmin, tmax, sunrise, sunset)

    elif hour < sunrise:
        # Before sunrise → still night: use previous day's sunset and Tmin_prev
        t_sunset_prev = temp_after_sunrise(
            sunset_prev, tmin_prev, tmax_prev, sunrise_prev, sunset_prev
        )
        denom = np.log(24 - (sunset_prev - sunrise) + 1)
        temp = t_sunset_prev - ((t_sunset_prev - tmin) / denom) * np.log(
            hour + 24 - sunset_prev + 1
        )

    else:
        # After sunset → cooling to next Tmin
        temp = temp_after_sunset(
            hour, tmin, tmax, tmin_next, sunrise, sunset, sunrise_next
        )

    return temp
