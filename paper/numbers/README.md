# In-text numbers

Measured values that the paper quotes in prose, outside any figure or table, with the
producer that wrote them. Each file is the producer's output with only the host name and batch-job id removed.

| file | producer | quoted in |
|---|---|---|
| `solve_timing_4d.json`, `solve_timing_8d.json` | `pipeline/run_solve_timing.py --dim {4,8} --n 50`, one thread on an exclusive AMD EPYC 7643 node | abstract, introduction, Sec. III B (cost of one certified solve, separable vs dense per-mode preconditioner) |
