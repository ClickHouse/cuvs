# XYZ: patched CAGRA — reproduction scripts

Research code used to measure "XYZ", i.e. CAGRA with the patches on this branch:

| commit | change |
|---|---|
| `CAGRA: count a detour only if B is closer to D than to A` | `detour_strict` pruning rule (`graph_core.cuh`) |
| `Add a fast device path to neighbors::refine` | warp-per-query refine kernel (225 ms -> 0.17 ms for 10k x 32) |
| `CAGRA (research): ADSampling early exit and 3-byte packed graph` | early exit in the single-CTA kernel (`CUVS_CAGRA_ADS_EPS`), 24-bit neighbor ids (`CUVS_CAGRA_GRAPH_U24=1`) |

Outside the library (done in `run_xyz_cagra.py`): random orthogonal rotation of data and queries,
global int8 quantization (scale from the 99.9th percentile of |x|), and packing of the graph into
3-byte ids. The rotation is what makes the ADSampling estimate valid.

**This is not production code.** Paths are hard-coded for the research machine
(`/data/scratch/...`), both new options are controlled by environment variables, and they only
apply to the single-CTA search algorithm (`search_algo::SINGLE_CTA`).

## Files

| file | purpose |
|---|---|
| `shim.cu` | ctypes wrapper over libcuvs (CAGRA float / int8 / VPQ indexes, refine, IVF-PQ, NN-descent, brute force) |
| `cg.py` | helpers: datasets, kNN graphs, `detour_strict` prune + reverse-edge merge on GPU (cupy), recall |
| `fast_rerank.py` | standalone rerank kernel (used before the refine fix) |
| `prep_big.py`, `prep_big_gpu.py` | 10M subsets of BigANN / Deep + exact ground truth |
| `build_big_graphs.py`, `prep_xyz_graphs.py` | NN-descent kNN (K=128), strict graphs (degree 32/64), stock CAGRA graph |
| `run_xyz_cagra.py` | main benchmark: CAGRA-Q baseline + XYZ (team size, ADSampling eps, `XYZ_DEG`, `XYZ_U24`) |
| `eval_big_q.py` | CAGRA-Q and IVF-PQ baselines on 10M |
| `t_build_times.py` | per-stage build times |
| `summ_xyzc.py`, `summ_graphmem.py` | result tables (QPS at recall 0.90 / 0.95 / 0.98) |

## Running

```bash
# library built from this branch and installed to /data/scratch/cuvs-install
export LD_LIBRARY_PATH=<cuvs build>/_deps/{kvikio,rmm,rapids_logger}-build:/data/scratch/cuvs-install/lib
nvcc -O2 -std=c++20 -shared -Xcompiler -fPIC --expt-relaxed-constexpr --expt-extended-lambda -arch=sm_86 \
     <cuvs/raft/rmm/cccl include dirs> shim.cu -o libshim.so -lcuvs
python prep_xyz_graphs.py sift-128-euclidean
python run_xyz_cagra.py sift-128-euclidean                 # degree 64, 4-byte graph
XYZ_U24=1 python run_xyz_cagra.py sift-128-euclidean       # degree 64, 3-byte graph
XYZ_DEG=32 XYZ_U24=1 python run_xyz_cagra.py sift-128-euclidean
python summ_graphmem.py sift-128-euclidean
```

Notes from the research machine (CUDA 13.1, driver 595, A10G):
- IVF-PQ on BigANN needs fp32 LUT / internal distances (fp16 overflows on uint8 magnitudes).
- The CAGRA index keeps a **non-owning** view of a device graph: keep the (packed) graph buffer alive.
- k-means based paths (VPQ, IVF-PQ) needed a local workaround for cuTile kernels that the driver
  could not load (`CUVS_DISABLE_CUTILE=1` in a local, uncommitted patch); not part of this branch.

## Results (A10G, 10k queries, recall@10, rerank of 32 included)

| dataset | recall | CAGRA-Q QPS | XYZ QPS (deg 64, 3-byte graph) | speedup | B/point CAGRA-Q / XYZ |
|---|---|---|---|---|---|
| SIFT-1M | 0.98 | 915k | 1436k | x1.57 | 320 / 320 |
| GloVe-1M | 0.90 / 0.95 | 303k / 111k | 406k / 146k | x1.32-1.34 | 306 / 304 |
| NYTimes | 0.90 | 331k | 496k | x1.50 | 384 / 448 |
| GIST-1M | 0.90 / 0.95 / 0.98 | 137k / 79k / 44k | 216k / 116k / 68k | x1.46-1.58 | 736 / 1152 |
| BigANN-10M | 0.98 | 555k | 876k | x1.58 | 320 / 320 |
| Deep-10M | 0.95 / 0.98 | 785k / 505k | 1226k / 754k | x1.49-1.56 | 304 / 288 |

Build time is dominated by NN-descent (~95%); XYZ builds as fast as or faster than CAGRA-Q
(e.g. Deep-10M 131 s vs 152 s).

## cuVS API (`cuvs::neighbors::xyz`, `cpp/include/cuvs/neighbors/xyz.hpp`)

`xyz::build` builds the CAGRA graph, quantizes the vectors (rotating them first when
`dim >= auto_rotation_min_dim`, default 512) and packs the graph; `xyz::search` quantizes the
queries, runs single-CTA CAGRA with ADSampling (`cagra::search_params::ads_eps`, only when rotated)
and re-ranks with `refine`. The environment-variable switches of the research commit are replaced by
`cagra::search_params::ads_eps` and `cagra::search_params::graph_packed_u24`.
Benchmark: `run_xyz_api.py` (`XYZ_ROT=1` always rotate, `0` never, `2` auto), tables via
`summ_xyzapi.py` / `summ_rot.py`.

End to end (build from scratch; query rotation, search and refine inside the measured time),
A10G, 10k queries, 3-byte graph, QPS ratio vs CAGRA-Q at equal recall@10:

| dataset | dim | rotated | not rotated | default (auto) |
|---|---|---|---|---|
| SIFT-1M | 128 | x1.39 | x1.46 | x1.46 |
| GloVe-1M | 100 | x1.29-1.30 | x1.31 | x1.31 |
| NYTimes | 256 | x1.36 | x1.35 | x1.35 |
| BigANN-10M | 128 | x1.47 | x1.51 | x1.51 |
| Deep-10M | 96 | x1.40-1.42 | x1.43-1.47 | x1.43-1.47 |
| GIST-1M | 960 | x1.39-1.44 | x0.90-0.93 | x1.39-1.44 |

Rotation does not change recall at dim <= 256 (within +-0.004) and only costs the query GEMM; at
dim 960 it is required for ADSampling.
