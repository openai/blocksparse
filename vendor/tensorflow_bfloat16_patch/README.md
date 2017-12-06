# Patching Tensorflow for bfloat16 support

The blocksparse kernels show the most significant performance gains when used with the `bfloat16` half-precision data type. However, Tensorflow (as of 1.4.1) doesn't support it yet. This directory contains files changed within the TF 1.4.1 source tree to enable bfloat16 support. In the future, Tensorflow is likely going to support bfloat16 out of the box; at that date, these patches are no longer needed.
