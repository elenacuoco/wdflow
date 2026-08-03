"""Mock data sets with known injections, for validating the search end to end."""
from wdf.mock.dataset import draw_injections, generate_dataset, optimal_snr, project_cbc
from wdf.mock.noise import analytic_psd, coloured_noise

__all__ = [
    "analytic_psd",
    "coloured_noise",
    "draw_injections",
    "generate_dataset",
    "optimal_snr",
    "project_cbc",
]
