# What a trigger stores

A window of $N$ samples transforms into $N$ coefficients, and thresholding at
$\sigma\sqrt{2\ln N}$ leaves very few of them. Measured on the triggers of a
real run: **2.2 to 3.0 coefficients survive** per trigger, out of 256, 512 or
1024. That is a density between 0.2 and 1 per cent.

So a trigger is stored as the pairs that survive --- `wt_index`, the position in
the transform, and `wt_value`, the coefficient --- and not as a vector of $N$
numbers of which all but three are zero.

| window | surviving | density | dense record | sparse record | ratio |
|---:|---:|---:|---:|---:|---:|
| 256  | 2.5 | 1.0 % | 2048 B | 15 B | 137 |
| 512  | 2.3 | 0.4 % | 4096 B | 14 B | 301 |
| 1024 | 2.2 | 0.2 % | 8192 B | 13 B | 609 |

Dense is $N$ doubles; sparse is one `uint16` index and one `float32` value per
surviving coefficient.

## Why it is the right representation and not only a smaller one

**It is what the algorithm produces.** Hard thresholding sets a coefficient to
zero or leaves it exactly as it was. The zeros are not small numbers to be
compressed later: they are the algorithm's statement that nothing was there, and
writing them down at eight bytes each records that statement 1021 times per
window.

**Nothing downstream wants the dense form.** The statistic is
$\lVert c\rVert_2/\sigma$ over the survivors. The wavegram places each survivor
at the tile its index implies. The reconstruction inverts the transform, which
needs the dense vector only as a transient inside one call. Every stage asks
which coefficients survived and where, which is what the sparse form says
directly.

**It is what a hardware implementation can carry.** The per-window arithmetic of
the search is a fixed sequence of multiply-accumulate steps, with no iteration
to convergence and no data-dependent branching, so it can be placed on an FPGA.
What such an implementation cannot do comfortably is buffer and ship a dense
vector per window per length per detector: a few index-value pairs is a small
fixed record, of a size known before the data arrives, which is exactly the kind
of object a hardware pipeline is built around. The dense vector would be the
part that does not fit, and it carries no information the pairs do not.

## Reading it

`wdf.analysis.coefficients` is the only module that knows the storage.
`to_dense` expands one trigger's pairs when a caller genuinely needs the vector
--- the inverse transform does --- and `coefficient_matrix` does it for a frame.
Everything else asks those two, so the representation can change again without
the rest of the code learning about it.
