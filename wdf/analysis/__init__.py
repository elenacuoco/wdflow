"""wdf.analysis: trigger clustering, multi-detector coincidence (classical +
GNN), background/false-alarm-probability and ROC analysis for WDF triggers.

Operates on plain pandas DataFrames / saved trigger files -- no dependency on
`wdf`'s own trigger-generation modules or on pytsa, so it works standalone.
"""
from wdf.analysis.clustering import (
    TriggerClusterer,
    WaveletPixelClusterer,
    collect_significant_pixels,
    wavegram_events,
)
from wdf.analysis.cluster_coefficients import (
    ClusterCoefficients,
    collect_cluster_coefficients,
    iter_cluster_coefficients,
    score_events_by_reconstruction,
)
from wdf.analysis.reconstruction import (
    combined_snr,
    phase_residual,
    reconstruct_clusters,
    stitch,
    waveform_overlap,
)
from wdf.analysis.wavelets import wavegram_ridge
from wdf.analysis.coincidence import CoincidenceFinder as LegacyCoincidenceFinder
from wdf.analysis.evaluation import (
    compare_statistics,
    efficiency_at_far,
    temporal_split,
    threshold_at_far,
)
from wdf.analysis.event_significance import EventCalibration
from wdf.analysis.injections import unclaimed_candidates
from wdf.analysis.ridge import (
    RIDGE_FEATURES,
    event_ridge,
    event_ridge_features,
)
from wdf.analysis.skymap import credible_area, localise, sky_grid
from wdf.analysis.modes import (
    DetectionMode,
    compare_modes,
    mode_roc,
    mode_threshold,
)
from wdf.analysis.output_schema import ensure_enwdf_column
from wdf.analysis.robust_events import (
    BackgroundEstimator as RobustBackgroundEstimator,
    ClusterConfig,
    CoincidenceConfig,
    FARConfig,
    IndexedCoincidenceFinder,
    TimeSlideFAR,
    cluster_detector_triggers,
    select_events_for_coincidence,
)

# The canonical names are the current implementations. The two earlier ones
# differ scientifically, not only in interface: `coincidence.CoincidenceFinder`
# pairs each event with its nearest neighbour without consuming it, so one
# event can appear in several candidates, and `significance.BackgroundEstimator`
# divides by the number of background candidates rather than by the background
# livetime, which is a tail percentile and not a rate. They stay importable
# under `Legacy*` for comparison and are not what a caller gets by default.
CoincidenceFinder = IndexedCoincidenceFinder
BackgroundEstimator = TimeSlideFAR

__all__ = [
    "BackgroundEstimator",
    "LegacyCoincidenceFinder",
    "ClusterConfig",
    "CoincidenceConfig",
    "FARConfig",
    "IndexedCoincidenceFinder",
    "RobustBackgroundEstimator",
    "TimeSlideFAR",
    "TriggerClusterer",
    "WaveletPixelClusterer",
    "collect_significant_pixels",
    "wavegram_events",
    "ClusterCoefficients",
    "collect_cluster_coefficients",
    "iter_cluster_coefficients",
    "score_events_by_reconstruction",
    "combined_snr",
    "reconstruct_clusters",
    "stitch",
    "phase_residual",
    "waveform_overlap",
    "wavegram_ridge",
    "CoincidenceFinder",
    "compare_statistics",
    "efficiency_at_far",
    "temporal_split",
    "threshold_at_far",
    "DetectionMode",
    "compare_modes",
    "mode_roc",
    "mode_threshold",
    "EventCalibration",
    "unclaimed_candidates",
    "event_ridge",
    "event_ridge_features",
    "RIDGE_FEATURES",
    "localise",
    "credible_area",
    "sky_grid",
    "localise",
    "credible_area",
    "sky_grid",
    "ensure_enwdf_column",
    "cluster_detector_triggers",
    "select_events_for_coincidence",
]
