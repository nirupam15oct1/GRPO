One bit-width for the whole layer (e.g. int8) → the kernel stays fast and simple.
Per-channel scales → each channel’s range handled individually, cheaply, no layout cost.
Special-case only the few troublemakers → keep the handful of outlier channels in higher precision (a tiny mixed-precision carve-out), or clip them by percentile. A few exceptions are manageable; hundreds of per-channel bit-widths are not.
Bottom_line : vary the scale freely (cheap), vary the bit-width rarely and in blocks (expensive).
