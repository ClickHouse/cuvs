import cupy as cp
_src = r'''
extern "C" __global__ void rerank(const float* __restrict__ X, int dim, const float* __restrict__ Q,
                                  const unsigned* __restrict__ cand, int C, int k, long long nq,
                                  long long* __restrict__ out_i, float* __restrict__ out_d)
{
  // one warp per query; each warp loops candidates, lanes split the dims (coalesced row reads)
  extern __shared__ float sm[];
  int w = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
  long long qi = (long long)blockIdx.x * nw + w;
  if (qi >= nq) return;
  float* dist = sm + w * C;
  const float* q = Q + qi * dim;
  for (int c = 0; c < C; c++) {
    unsigned id = cand[qi * C + c];
    float s = 0.f;
    if (id != 0xFFFFFFFFu) {
      const float* x = X + (long long)id * dim;
      for (int j = lane; j < dim; j += 32) { float t = x[j] - q[j]; s += t * t; }
      for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    } else s = 3.4e38f;
    if (lane == 0) dist[c] = s;
  }
  __syncwarp();
  // selection of k smallest (C <= 256): k rounds of warp argmin
  for (int r = 0; r < k; r++) {
    float bv = 3.4e38f; int bi = -1;
    for (int c = lane; c < C; c += 32) if (dist[c] < bv) { bv = dist[c]; bi = c; }
    for (int o = 16; o; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, bv, o); int oi = __shfl_xor_sync(0xffffffff, bi, o);
      if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { out_d[qi * k + r] = bv; out_i[qi * k + r] = bi >= 0 ? cand[qi * C + bi] : -1; if (bi >= 0) dist[bi] = 3.4e38f; }
    __syncwarp();
  }
}
'''
_k = cp.RawKernel(_src, 'rerank')

def rerank(X, Q, cand, k=10, nw=8):
    nq, C = cand.shape
    oi = cp.empty((nq, k), cp.int64); od = cp.empty((nq, k), cp.float32)
    _k(((nq + nw - 1) // nw,), (32 * nw,), (X, cp.int32(X.shape[1]), Q, cand, cp.int32(C), cp.int32(k), cp.int64(nq), oi, od),
       shared_mem=nw * C * 4)
    return oi, od
