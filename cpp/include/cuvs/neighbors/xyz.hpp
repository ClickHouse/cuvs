/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>
#include <memory>

/**
 * XYZ: a CAGRA index searched on randomly rotated int8 vectors.
 *
 * Build:
 *   1. the CAGRA graph is built on the float dataset (`cagra::build`);
 *   2. the dataset is multiplied by a random orthogonal matrix and quantized to int8 with one
 *      global scale (taken from a high percentile of |x| to be robust to outliers);
 *   3. optionally, the graph is packed into 3-byte neighbor ids (requires n_rows < 2^24).
 *
 * Search:
 *   1. queries are rotated and quantized the same way;
 *   2. CAGRA (single-CTA) searches the int8 vectors for `refine_candidates` candidates, optionally
 *      with the ADSampling early exit (`ads_eps`), which is valid because of the rotation;
 *   3. the candidates are re-ranked exactly on the original float vectors (`refine`).
 *
 * Only the L2 metric is supported. The index keeps a non-owning view of the float dataset passed to
 * `build`; it must stay alive (on the device) while the index is used.
 */
namespace CUVS_EXPORT cuvs {
namespace neighbors::xyz {

/** When to apply the random rotation before int8 quantization. */
enum class rotation_mode {
  /** Always rotate (required for ADSampling to be valid). */
  kAlways,
  /** Never rotate; ADSampling is then disabled at search time. */
  kNever,
  /** Rotate only if dim >= `auto_rotation_min_dim` (where ADSampling can pay off). */
  kAuto
};

struct index_params {
  /** Parameters of the underlying CAGRA graph build (metric must be L2Expanded). */
  cuvs::neighbors::cagra::index_params cagra_params;
  /** Store the graph with 3-byte neighbor ids (25% less graph memory). */
  bool pack_graph = true;
  /** Percentile of |rotated x| mapped to the int8 limit 127 (values above are clipped). */
  double quantization_percentile = 99.9;
  /** Number of rows sampled to estimate the quantization scale. */
  int64_t quantization_sample_rows = 20000;
  /** Seed of the random rotation. */
  uint64_t seed = 0;
  /** Whether to rotate the vectors before quantization. */
  rotation_mode rotation = rotation_mode::kAuto;
  /** Minimum dimensionality for which `rotation_mode::kAuto` rotates. */
  uint32_t auto_rotation_min_dim = 512;
};

struct search_params {
  /** Parameters of the underlying CAGRA search; `algo` is forced to SINGLE_CTA. */
  cuvs::neighbors::cagra::search_params cagra_params;
  /** ADSampling early-exit parameter (0 disables it); see cagra::search_params::ads_eps. */
  float ads_eps = 0.75f;
  /** Number of int8 candidates re-ranked on the float vectors (>= k). */
  uint32_t refine_candidates = 32;
};

/** XYZ index. Owns the rotation, the int8 vectors and the graph. */
struct index {
  int64_t n_rows = 0;
  uint32_t dim   = 0;  // original dimensionality
  uint32_t dim_q = 0;  // int8 row width (dim padded to a multiple of 16)
  float scale    = 1;  // int8 value = round(rotated_x * scale)
  bool packed    = false;
  bool rotated   = false;  // vectors were rotated (rotation matrix is [dim, dim]); else no rotation
  uint32_t degree = 0;     // logical graph degree
  raft::device_matrix<float, int64_t> rotation;  // [dim, dim], rotated = x * rotation
  raft::device_matrix<int8_t, int64_t> codes;    // [n_rows, dim_q]
  raft::device_matrix<uint32_t, int64_t> graph;  // [n_rows, degree] or packed [n_rows, 3*degree/4]
  raft::device_matrix_view<const float, int64_t> dataset;  // float vectors for re-ranking
  std::unique_ptr<cuvs::neighbors::cagra::device_padded_index<int8_t, uint32_t>> cagra_index;

  explicit index(raft::resources const& res);

  /** Device memory used by the index for traversal (codes + graph + rotation), in bytes. */
  [[nodiscard]] auto memory_bytes() const -> size_t;
};

/** Build an XYZ index from a float dataset on the device (kept as a non-owning view). */
auto build(raft::resources const& res,
           const index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset) -> index;

/** Search k nearest neighbors (L2). */
void search(raft::resources const& res,
            const search_params& params,
            const index& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

}  // namespace neighbors::xyz
}  // namespace CUVS_EXPORT cuvs
