"""Mathematical, physical and unit constants used by `wdf`.

Nothing here is a choice of the analysis: these are values of nature, of the
instruments' published geometry, of units, or of closed-form formulas. Every
choice of the analysis is in ``wdf/config/defaults.toml``; this module is the
one place outside it where a number may be written
(``tests/test_analysis_defaults.py``).

Append only: one constant per line, with its meaning and its unit.
"""

# ---------------------------------------------------------------- units
#: Seconds in a day.
SECONDS_PER_DAY = 86400.0
#: Seconds in an hour.
SECONDS_PER_HOUR = 3600.0
#: Days in a Julian year.
DAYS_PER_JULIAN_YEAR = 365.25
#: Milliseconds in a second.
MS_PER_S = 1e3
#: Per cent in one.
PERCENT = 100.0
#: Degrees in a full turn.
FULL_TURN_DEG = 360.0
#: Degrees in half a turn.
HALF_TURN_DEG = 180.0
#: Degrees in a quarter turn (the largest declination).
QUARTER_TURN_DEG = 90.0
#: Decibels of an amplitude ratio: 20 log10.
DB_AMPLITUDE = 20.0
#: Decibels of a power ratio: 10 log10.
DB_POWER = 10.0
#: Base of the decibel logarithm.
DECIBEL_BASE = 10.0

# ---------------------------------------------------------------- physics
#: Speed of light in vacuum, m/s.
SPEED_OF_LIGHT = 299792458.0
#: G Msun / c^3, s.
SOLAR_MASS_S = 4.925490947e-6
#: Detector vertices in the Earth-fixed frame, m, as given in the
#: LIGO/Virgo/KAGRA instrument descriptions.
DETECTOR_VERTEX = {
    "H1": (-2161414.93, -3834695.36, 4600350.22),
    "L1": (-74276.05, -5496283.72, 3224257.02),
    "V1": (4546374.10, 842989.70, 4378576.96),
    "K1": (-3777336.02, 3484898.41, 3765313.68),
    "G1": (3856309.94, 666598.96, 5019641.42),
}
#: Greenwich mean sidereal time at the GPS epoch, rad.
GPS_EPOCH_GMST_RAD = 1.8272328613543891
#: Sidereal days per solar day.
SIDEREAL_PER_SOLAR_DAY = 1.0027379093508615

# ------------------------------------------------- closed-form formulas
#: Leading-order inspiral: df/dt = (96/5) pi^(8/3) (G Mc / c^3)^(5/3) f^(11/3).
INSPIRAL_RATE_COEFFICIENT = 96.0 / 5.0
#: Exponent 8/3 of the leading-order inspiral.
EIGHT_THIRDS = 8.0 / 3.0
#: Exponent 5/3 of the leading-order inspiral.
FIVE_THIRDS = 5.0 / 3.0
#: Exponent 3/8 of the leading-order inspiral.
THREE_EIGHTHS = 3.0 / 8.0
#: Chirp mass, (m1 m2)^(3/5) / (m1 + m2)^(1/5): the product's exponent.
CHIRP_MASS_PRODUCT_EXPONENT = 0.6
#: The sum's exponent of the chirp mass.
CHIRP_MASS_SUM_EXPONENT = 0.2

# ------------------------------------------------------------ statistics
#: Median of ``|x|`` over the standard deviation of a Gaussian (MAD -> sigma
#: divides by it).
MAD_GAUSSIAN = 0.6745
#: Standard deviation over the median absolute deviation of a Gaussian.
MAD_TO_SIGMA = 1.4826
#: Variance of a uniform distribution of width w is w^2 over this.
UNIFORM_VARIANCE_DIVISOR = 12.0
#: Kurtosis of a Gaussian.
GAUSSIAN_KURTOSIS = 3.0
#: Variance of the sample excess kurtosis of a Gaussian is this over n.
KURTOSIS_VARIANCE_FACTOR = 24.0
#: The local cosine's bell, sin(BELL_PI_FRACTION pi (1 + theta)), theta in
#: [-1, 1]: the fraction of pi that makes r(t)^2 + r(-t)^2 = 1.
BELL_PI_FRACTION = 0.25

# ------------------------------------------------------ scipy conventions
#: scipy's default `padlen` of `filtfilt`/`sosfiltfilt` is this many times the
#: number of taps of the filter (``3 * ntaps``).
SCIPY_PADLEN_PER_TAP = 3
#: Order of the moment the kurtosis is made of (the fourth standardised moment).
KURTOSIS_MOMENT_ORDER = 4

# ------------------------------------------------- noncentral energy (hrss)
#: Variance of the energy sum_k (s_k + n_k)^2 of N white Gaussian
#: coefficients of standard deviation sigma about a signal of energy S:
#: 4 S sigma^2 + 2 N sigma^4. The factor of the signal term.
NONCENTRAL_ENERGY_SIGNAL_FACTOR = 4.0
#: The power of sigma in its noise term, 2 N sigma^4.
NONCENTRAL_ENERGY_NOISE_POWER = 4

# ------------------------------------------------------------- geometry
#: Solid angle of the full sphere over pi, sr (the sphere is 4 pi sr).
SPHERE_SOLID_ANGLE_OVER_PI = 4.0
