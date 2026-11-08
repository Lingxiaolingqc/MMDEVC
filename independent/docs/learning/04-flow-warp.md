# Stage 4: aligning history before averaging

Think of placing an old tracing over a new photograph. Averaging before lining up the pictures mixes different objects together.

In a tiny one-dimensional example, the previous depth row is `[2, 2, 8, 8]`. A feature moves right by one pixel. Current pixel x=3 must gather its old value from previous x=2, so its backward displacement is -1. The gathered value is 8. At current x=0, the requested previous x=-1 is outside the image; history is invalid there.

```text
previous x:       0   1   2   3
previous depth:   2   2   8   8
                           \  rightward motion
current x:        0   1   2   3
gather old x:    -1   0   1   2
valid history:    no  yes yes yes
```

Backward sampling at current pixel p uses previous coordinates `p + backward_flow[p]`. Forward flow maps in the opposite direction and cannot be substituted directly into this gather.

Forward-backward checking asks whether the two arrows make a round trip. A current-to-previous displacement of -1 followed by a previous-to-current displacement of +1 sums to zero. Large round-trip error suggests unreliable motion or occlusion, so we reject that history.

If flow is estimated on a smaller image, its displacement units must be resized too. Expanding a 2-row by 3-column flow field to 6 rows by 6 columns multiplies x displacements by 2 and y displacements by 3. Our test explicitly checks `(1,1) -> (2,3)`.

Learning checkpoint: trace both directions for one moving pixel and explain why an out-of-bounds or newly revealed pixel must use today's prediction.
