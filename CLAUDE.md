# wdflow

The WDF pipeline: wavelet-domain search, clustering, coincidence and ranking,
on top of the p4TSA C++ core exposed as `pytsa`.

## Performance constraints

These are requirements of the target deployment, not optimisations to add once
the physics works. Code that does not meet them is not finished.

1. **Exploit the GPU wherever possible.** Learned stages run on it; array work
   is written so it can move there.
2. **Implementable on FPGA.** The search front end must be expressible in fixed
   function hardware: bounded memory, streaming, no whole-segment buffering, no
   data-dependent allocation, and a latency known before the filter runs. The
   zero-phase whitening, whose latency is exactly its own order, is the model.
3. **Realtime, for insertion into LVK low latency.** The pipeline must keep up
   with the data in a stream. Anything whose cost grows faster than linearly in
   the event rate is a defect, not a slow path.

In practice:

- Never form a dense pairwise matrix over anything event-sized. Sort the axis
  and search it: `wdf.analysis.pairs` has `neighbour_pairs` and `cross_pairs`.
  A `[:, None]` against `[None, :]` over events or pixels is the bug.
- Never build a DataFrame a row at a time, and never index one per element with
  `.iloc[]`. Build column by column.
- Hoist host-to-device transfers out of training loops: the graph does not
  change while the weights do.
- Measure before and after, and report both numbers rather than the expectation.

## Style

- A function does one thing; its docstring says what it does, with `:param:`
  and `:return:`. No explanatory comments in the body.
- No ad-hoc heuristics and no special cases: never branch for one input, never
  document around a dataset or a bug.
- No estimator that assumes a transient's shape.
- Never fix a problem by introducing another: revert a breaking change rather
  than propping it up downstream.
- English throughout, including notebooks.
