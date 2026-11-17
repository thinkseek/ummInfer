# Measurement evidence

Committed text evidence for the matmul optimisation of the Omni thinker slice.
Nothing here is produced by the build; regenerate it with the script.

| File | Contents |
|---|---|
| `00_env.txt` | GPU inventory, `nvcc --version`, SHA256 of the baseline / optimised / bench binaries and of the image, `ldd` of the optimised binary, device build and GPU-test exit codes, GPU6 memory before/after |
| `10_baseline.txt` | Pre-optimisation binary, three serial runs on GPU6 (image with 1 new token, image to EOS, text to EOS) |
| `20_optimised.txt` | `umm-omni-bench` plus the two accepted CLI invocations |
| `30_gpu_tests.txt` | `ctest -R omni_moe_down_test` and the test binary's own output |
| `40_host_tests.txt` | Mac host build and `ctest` |

Every entry is `$ <command>`, then `exit=<real exit status>`, then the
command's complete stdout+stderr.

## Caps

Each file is capped at 64 KiB at runtime, twice: the command's output is piped
through `head -c 65536` (a runaway command is bounded by construction), and the
log is re-capped after every append. A truncated file is therefore a file whose
record ends mid-line, and the script leaves no other marker.

## Reproducing

```
# Mac / host, no GPU:
native/omni_tests/evidence/run_measurement.sh host

# H20B GPU6, from a container that can see the checkpoint:
NEW=/path/build/omni_cli/umm-omni-native \
BENCH=/path/build/omni_cli/umm-omni-bench \
BASE=/path/baseline-2b6177f/omni_cli/umm-omni-native \
MODEL=/path/to/omni IMAGE=/tmp/man_ironing.png CUDA_VISIBLE_DEVICES=6 \
native/omni_tests/evidence/run_measurement.sh gpu
```

`BASE` must be built from base commit `2b6177f` with `-DUMM_OMNI_CUDA=ON`; it is
the unmodified engine and is the only source of the "before" numbers.

The image fixture is `man_ironing.png`, 570x380, sha256
`e06917184a00b14abd70cd8ea0ff5dca9abfbbad29f7b25c02f97133d4cd060e`.
The prompt is the accepted one:
`What is the person in this image doing? Answer in one short sentence.`
