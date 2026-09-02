# Changelog

Versions follow [semantic versioning](https://semver.org). A release records
what the software does differently, not how it came to.

## 1.1.1 --- 2026-09-02

Documentation only; the code is unchanged from 1.1.0.

The README, the design note and the published page state what the pipeline
does: an event's energy is the norm of its stitched reconstruction and what
ranks it for detection is its loudest block, the significance conditions on the
number of tiles and is read out of sample, and the scale a block is ranked on
is the median of its neighbours' while the thresholding stays with the block
itself.

## 1.1.0 --- 2026-09-02

### The statistic that selects is not the statistic that measures

An event's energy is the norm over its tiles; what ranks it for detection is
its loudest block. A hard threshold admits every tile at a floor of `2 ln N` in
normalised energy, so a sum over tiles accumulates that floor in the noise as
well as in the signal and a tile earns its place only when its excess exceeds
half the event's mean excess per tile. Ranking on the loudest block maps each
cluster to a maximum over quantities the search already computed, so its
background is a subset of the one an ungrouped search produces and the grouping
reduces the trials factor without being able to lose a candidate.

- `network_graph` ranks its nodes on `EnWDF_window` where the node table
  carries it.

### Nothing is calibrated on its own time

- `event_significance.out_of_sample_significance` scores each fold of a
  background from the others. A calibration read on the sample it was fitted on
  scores the j-th largest event of a bin as `log((N + 1) / (j + 1))` whatever
  the data, so a rate read off it is lower than a fresh background of the same
  livetime gives.
- `event_significance.significance_off_source` scores any event from the
  background of every time fold but its own, which is what a single recorded
  stretch allows: there the background is a part of the same data and a
  candidate lives beside it.
- `EventCalibration` conditions on `n_pixels`, the tiles the statistic sums,
  instead of on the blocks the analysis grid cut. The tile count sets the
  statistic's scale under the null; the block count is a property of the grid
  and made an event's significance depend on where the grid started.
- The exponential tail divides the total excess by `k - 1` rather than `k`, one
  of the `k` values being the anchor itself. A tail of equal values measures no
  slope instead of one at the smallest positive float.
- A size column carrying a missing value is refused rather than forming a bin
  of its own below every real size. An infinite statistic is scored as the
  loudest event there can be rather than dropped. A size larger than any the
  background produced is pooled into the last bin and says so.

### A coherent statistic is a product of amplitudes

- `event_tiles` returns each tile's signed amplitude beside its energy, and
  `tile_coherence` takes the product on the amplitudes. A product of magnitudes
  is positive whatever the data, so it grew with the number of tile pairs that
  met and two long events overlapping by accident scored above two short ones
  describing one transient. The signed product has mean zero under the null,
  and the pair is ranked on its magnitude because both polarities are physical.

### A block's noise is read where the transient is not

- `scale.local_noise_scale` takes the median of the neighbouring blocks'
  scales, and `scale.on_local_scale` re-expresses a block's statistic on it
  exactly, without recomputing a coefficient. A block's own scale is measured
  on the data it holds, signal included, so a transient loud enough to matter
  is divided by a scale it inflated itself. A block whose neighbours say
  nothing keeps the scale it measured itself.

### Also

- `detector_events` carries `n_pixels`, the distinct tiles an event owns.
- `efficiency_at_far` returns the realised rate and the number of accidentals
  the threshold stands on: a requested rate is not the rate a finite background
  can realise.
- `match_injections` takes the argmax over finite statistics alone, so an
  injection whose only candidate carries no statistic is unmatched rather than
  matched to it.

## 1.0.0 --- 2026-08-20

First tagged release.
