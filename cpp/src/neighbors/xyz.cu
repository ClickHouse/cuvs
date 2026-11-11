/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/refine.hpp>
#include <cuvs/neighbors/xyz.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/sort.h>

#include <algorithm>
#include <cmath>

namespace cuvs::neighbors::xyz {

namespace {

constexpr int64_t kRowBatch = 1 << 18;

/** int8 code = clip(round(x * scale)); columns [dim, dim_q) are zero padding. */
RAFT_KERNEL quantize_kernel(
  const float* x, int64_t rows, uint32_t dim, uint32_t dim_q, float scale, int8_t* out)
{
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * dim_q) { return; }
  const int64_t r  = i / dim_q;
  const uint32_t c = static_cast<uint32_t>(i % dim_q);
  float v          = c < dim ? rintf(x[r * dim + c] * scale) : 0.f;
  v                = fminf(fmaxf(v, -127.f), 127.f);
  out[i]           = static_cast<int8_t>(v);
}

/** Pack uint32 neighbor ids into 3 bytes (little endian); invalid ids become 0xFFFFFF. */
RAFT_KERNEL pack_graph_kernel(const uint32_t* graph, int64_t n_edges, uint8_t* out)
{
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n_edges) { return; }
  const uint32_t v = graph[i] >= (1u << 24) ? 0xFFFFFFu : graph[i];
  out[3 * i]       = static_cast<uint8_t>(v & 0xFF);
  out[3 * i + 1]   = static_cast<uint8_t>((v >> 8) & 0xFF);
  out[3 * i + 2]   = static_cast<uint8_t>((v >> 16) & 0xFF);
}

/** rotated = x * rotation (skipped if `rotate` is false), then int8 quantization, in row batches.
 */
void rotate_and_quantize(raft::resources const& res,
                         raft::device_matrix_view<const float, int64_t> x,
                         raft::device_matrix_view<const float, int64_t> rotation,
                         bool rotate,
                         float scale,
                         uint32_t dim_q,
                         int8_t* out)
{
  const int64_t n    = x.extent(0);
  const uint32_t dim = static_cast<uint32_t>(x.extent(1));
  auto stream        = raft::resource::get_cuda_stream(res);
  const int64_t bs   = std::min<int64_t>(n, kRowBatch);
  auto tmp = raft::make_device_matrix<float, int64_t>(res, rotate ? bs : 0, rotate ? dim : 0);
  for (int64_t r0 = 0; r0 < n; r0 += bs) {
    const int64_t m  = std::min<int64_t>(bs, n - r0);
    const float* src = x.data_handle() + r0 * dim;
    if (rotate) {
      raft::linalg::gemm(res,
                         raft::make_device_matrix_view<const float, int64_t>(src, m, dim),
                         rotation,
                         raft::make_device_matrix_view<float, int64_t>(tmp.data_handle(), m, dim));
      src = tmp.data_handle();
    }
    const int64_t total = m * dim_q;
    const int threads   = 256;
    quantize_kernel<<<raft::ceildiv<int64_t>(total, threads), threads, 0, stream.get()>>>(
      src, m, dim, dim_q, scale, out + r0 * dim_q);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

}  // namespace

index::index(raft::resources const& res)
  : rotation(raft::make_device_matrix<float, int64_t>(res, 0, 0)),
    codes(raft::make_device_matrix<int8_t, int64_t>(res, 0, 0)),
    graph(raft::make_device_matrix<uint32_t, int64_t>(res, 0, 0)),
    dataset(raft::make_device_matrix_view<const float, int64_t>(nullptr, 0, 0))
{
}

auto index::memory_bytes() const -> size_t
{
  return codes.size() * sizeof(int8_t) + graph.size() * sizeof(uint32_t) +
         rotation.size() * sizeof(float);
}

auto build(raft::resources const& res,
           const index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset) -> index
{
  RAFT_EXPECTS(params.cagra_params.metric == cuvs::distance::DistanceType::L2Expanded,
               "xyz::build: only the L2Expanded metric is supported");
  auto stream        = raft::resource::get_cuda_stream(res);
  const int64_t n    = dataset.extent(0);
  const uint32_t dim = static_cast<uint32_t>(dataset.extent(1));

  index idx(res);
  idx.n_rows  = n;
  idx.dim     = dim;
  idx.dim_q   = raft::round_up_safe<uint32_t>(dim, 16);
  idx.dataset = dataset;

  // 1. CAGRA graph on the float vectors
  {
    auto float_index = cuvs::neighbors::cagra::build(
      res, params.cagra_params, cuvs::neighbors::make_device_padded_dataset_view(res, dataset));
    auto g     = float_index.graph();
    idx.degree = static_cast<uint32_t>(g.extent(1));
    idx.packed = params.pack_graph;
    if (idx.packed) {
      RAFT_EXPECTS(n < (int64_t(1) << 24) - 1, "xyz: a packed graph requires n_rows < 2^24 - 1");
      RAFT_EXPECTS(idx.degree % 4 == 0,
                   "xyz: a packed graph requires graph_degree to be a multiple of 4");
      idx.graph = raft::make_device_matrix<uint32_t, int64_t>(res, n, 3 * idx.degree / 4);
      const int64_t n_edges = n * idx.degree;
      pack_graph_kernel<<<raft::ceildiv<int64_t>(n_edges, 256), 256, 0, stream.get()>>>(
        g.data_handle(), n_edges, reinterpret_cast<uint8_t*>(idx.graph.data_handle()));
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {
      idx.graph = raft::make_device_matrix<uint32_t, int64_t>(res, n, idx.degree);
      raft::copy(idx.graph.data_handle(), g.data_handle(), g.size(), stream);
    }
    raft::resource::sync_stream(res);
  }

  idx.rotated = params.rotation == rotation_mode::kAlways ||
                (params.rotation == rotation_mode::kAuto && dim >= params.auto_rotation_min_dim);
  // 2. random orthogonal rotation: Q of the QR decomposition of a Gaussian matrix
  if (idx.rotated) {
    auto gauss = raft::make_device_matrix<float, int64_t, raft::col_major>(res, dim, dim);
    raft::random::RngState rng(params.seed);
    raft::random::normal(res, rng, gauss.data_handle(), gauss.size(), 0.f, 1.f);
    idx.rotation = raft::make_device_matrix<float, int64_t>(res, dim, dim);
    // Q is column major; read as row major it is Q^T, which is orthogonal as well.
    raft::linalg::qrGetQ(res,
                         gauss.data_handle(),
                         idx.rotation.data_handle(),
                         static_cast<int>(dim),
                         static_cast<int>(dim),
                         stream.get());
  }
  auto rot_view =
    raft::make_device_matrix_view<const float, int64_t>(idx.rotation.data_handle(), dim, dim);

  // 3. quantization scale from a percentile of |rotated x| over a strided row sample
  {
    const int64_t m = std::min<int64_t>(n, std::max<int64_t>(1, params.quantization_sample_rows));
    const int64_t stride = n / m;
    auto sample          = raft::make_device_matrix<float, int64_t>(res, m, dim);
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(sample.data_handle(),
                                    sizeof(float) * dim,
                                    dataset.data_handle(),
                                    sizeof(float) * dim * stride,
                                    sizeof(float) * dim,
                                    m,
                                    cudaMemcpyDeviceToDevice,
                                    stream.get()));
    auto rotated = raft::make_device_matrix<float, int64_t>(res, m, dim);
    if (idx.rotated) {
      raft::linalg::gemm(res, raft::make_const_mdspan(sample.view()), rot_view, rotated.view());
    } else {
      raft::copy(rotated.data_handle(), sample.data_handle(), sample.size(), stream);
    }
    auto absv =
      raft::make_device_vector_view<float, int64_t>(rotated.data_handle(), rotated.size());
    raft::linalg::map(res, absv, raft::abs_op{}, raft::make_const_mdspan(absv));
    thrust::sort(raft::resource::get_thrust_policy(res),
                 rotated.data_handle(),
                 rotated.data_handle() + rotated.size());
    const auto pos = static_cast<int64_t>(
      std::floor(params.quantization_percentile / 100.0 * static_cast<double>(rotated.size() - 1)));
    float limit = 0;
    raft::copy(&limit, rotated.data_handle() + pos, 1, stream);
    raft::resource::sync_stream(res);
    idx.scale = limit > 0 ? 127.f / limit : 1.f;
  }

  // 4. int8 codes of the rotated dataset
  idx.codes = raft::make_device_matrix<int8_t, int64_t>(res, n, idx.dim_q);
  rotate_and_quantize(
    res, dataset, rot_view, idx.rotated, idx.scale, idx.dim_q, idx.codes.data_handle());

  // 5. CAGRA index over the int8 codes and the (packed) graph; both are non-owning views
  idx.cagra_index = std::make_unique<cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>>(
    res,
    cuvs::distance::DistanceType::L2Expanded,
    cuvs::neighbors::make_device_padded_dataset_view(res,
                                                     raft::make_const_mdspan(idx.codes.view())),
    raft::make_const_mdspan(idx.graph.view()));
  raft::resource::sync_stream(res);
  return idx;
}

void search(raft::resources const& res,
            const search_params& params,
            const index& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  RAFT_EXPECTS(queries.extent(1) == idx.dim, "xyz::search: query dimensionality mismatch");
  const int64_t nq  = queries.extent(0);
  const uint32_t k  = static_cast<uint32_t>(neighbors.extent(1));
  const uint32_t kc = std::max(k, params.refine_candidates);

  // 1. rotate + quantize the queries
  auto q8 = raft::make_device_matrix<int8_t, int64_t>(res, nq, idx.dim_q);
  rotate_and_quantize(
    res,
    queries,
    raft::make_device_matrix_view<const float, int64_t>(
      idx.rotation.data_handle(), idx.rotated ? idx.dim : 0, idx.rotated ? idx.dim : 0),
    idx.rotated,
    idx.scale,
    idx.dim_q,
    q8.data_handle());

  // 2. CAGRA on the int8 codes
  auto sp             = params.cagra_params;
  sp.algo             = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
  sp.ads_eps          = idx.rotated ? params.ads_eps : 0.f;  // the estimate needs rotated data
  sp.graph_packed_u24 = idx.packed;
  sp.itopk_size       = std::max<size_t>(sp.itopk_size, kc);
  auto cand           = raft::make_device_matrix<uint32_t, int64_t>(res, nq, kc);
  auto cand_d         = raft::make_device_matrix<float, int64_t>(res, nq, kc);
  cuvs::neighbors::cagra::search(
    res, sp, *idx.cagra_index, raft::make_const_mdspan(q8.view()), cand.view(), cand_d.view());

  // 3. exact re-ranking on the float vectors
  auto cand64 = raft::make_device_matrix<int64_t, int64_t>(res, nq, kc);
  raft::linalg::map(
    res,
    raft::make_device_vector_view<int64_t, int64_t>(cand64.data_handle(), cand64.size()),
    [] __device__(uint32_t v) { return v == 0xFFFFFFFFu ? int64_t{-1} : static_cast<int64_t>(v); },
    raft::make_device_vector_view<const uint32_t, int64_t>(cand.data_handle(), cand.size()));
  cuvs::neighbors::refine(res,
                          idx.dataset,
                          queries,
                          raft::make_const_mdspan(cand64.view()),
                          neighbors,
                          distances,
                          cuvs::distance::DistanceType::L2Expanded);
}

}  // namespace cuvs::neighbors::xyz
