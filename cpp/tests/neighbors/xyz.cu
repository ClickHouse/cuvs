/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/xyz.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>

namespace cuvs::neighbors::xyz {

struct XyzInputs {
  int64_t n_rows;
  int64_t dim;
  bool pack_graph;
  float ads_eps;
  double max_recall_drop;  // allowed recall loss vs. CAGRA on the float vectors
  rotation_mode rotation;
};

class XyzTest : public ::testing::TestWithParam<XyzInputs> {};

TEST_P(XyzTest, Recall)
{
  const auto p = GetParam();
  raft::device_resources res;
  auto stream      = raft::resource::get_cuda_stream(res);
  const int64_t nq = 500;
  const int64_t k  = 10;
  auto data        = raft::make_device_matrix<float, int64_t>(res, p.n_rows, p.dim);
  auto queries     = raft::make_device_matrix<float, int64_t>(res, nq, p.dim);
  raft::random::RngState rng(1234ULL);
  // low intrinsic dimensionality, like real embeddings: x = z * A + 0.05 * noise, z in R^16
  {
    const int64_t r = 16;
    auto A          = raft::make_device_matrix<float, int64_t>(res, r, p.dim);
    auto zd         = raft::make_device_matrix<float, int64_t>(res, p.n_rows, r);
    auto zq         = raft::make_device_matrix<float, int64_t>(res, nq, r);
    raft::random::normal(res, rng, A.data_handle(), A.size(), 0.f, 1.f);
    raft::random::normal(res, rng, zd.data_handle(), zd.size(), 0.f, 1.f);
    raft::random::normal(res, rng, zq.data_handle(), zq.size(), 0.f, 1.f);
    raft::random::normal(res, rng, data.data_handle(), data.size(), 0.f, 0.05f);
    raft::random::normal(res, rng, queries.data_handle(), queries.size(), 0.f, 0.05f);
    auto td = raft::make_device_matrix<float, int64_t>(res, p.n_rows, p.dim);
    auto tq = raft::make_device_matrix<float, int64_t>(res, nq, p.dim);
    raft::linalg::gemm(
      res, raft::make_const_mdspan(zd.view()), raft::make_const_mdspan(A.view()), td.view());
    raft::linalg::gemm(
      res, raft::make_const_mdspan(zq.view()), raft::make_const_mdspan(A.view()), tq.view());
    raft::linalg::map(res,
                      data.view(),
                      raft::add_op{},
                      raft::make_const_mdspan(data.view()),
                      raft::make_const_mdspan(td.view()));
    raft::linalg::map(res,
                      queries.view(),
                      raft::add_op{},
                      raft::make_const_mdspan(queries.view()),
                      raft::make_const_mdspan(tq.view()));
  }

  index_params ip;
  ip.cagra_params.graph_degree              = 32;
  ip.cagra_params.intermediate_graph_degree = 64;
  ip.cagra_params.graph_build_params =
    cuvs::neighbors::cagra::graph_build_params::nn_descent_params(64);
  ip.pack_graph = p.pack_graph;
  ip.rotation   = p.rotation;
  auto idx      = build(res, ip, raft::make_const_mdspan(data.view()));
  EXPECT_EQ(idx.packed, p.pack_graph);
  EXPECT_EQ(idx.graph.extent(1), p.pack_graph ? 24 : 32);

  search_params sp;
  sp.cagra_params.itopk_size = 64;
  sp.ads_eps                 = p.ads_eps;
  auto nb                    = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto dd                    = raft::make_device_matrix<float, int64_t>(res, nq, k);
  search(res, sp, idx, raft::make_const_mdspan(queries.view()), nb.view(), dd.view());

  // exact neighbors
  auto bf    = cuvs::neighbors::brute_force::build(res, raft::make_const_mdspan(data.view()));
  auto gt_nb = raft::make_device_matrix<int64_t, int64_t>(res, nq, k);
  auto gt_dd = raft::make_device_matrix<float, int64_t>(res, nq, k);
  cuvs::neighbors::brute_force::search(
    res, bf, raft::make_const_mdspan(queries.view()), gt_nb.view(), gt_dd.view());

  // reference: CAGRA on the float vectors with the same graph parameters and itopk
  auto ref_index = cuvs::neighbors::cagra::build(
    res,
    ip.cagra_params,
    cuvs::neighbors::make_device_padded_dataset_view(res, raft::make_const_mdspan(data.view())));
  cuvs::neighbors::cagra::search_params rsp;
  rsp.itopk_size = 64;
  auto ref_nb    = raft::make_device_matrix<uint32_t, int64_t>(res, nq, k);
  auto ref_dd    = raft::make_device_matrix<float, int64_t>(res, nq, k);
  cuvs::neighbors::cagra::search(
    res, rsp, ref_index, raft::make_const_mdspan(queries.view()), ref_nb.view(), ref_dd.view());

  std::vector<int64_t> h(nq * k), g(nq * k);
  std::vector<uint32_t> r(nq * k);
  raft::copy(h.data(), nb.data_handle(), h.size(), stream);
  raft::copy(g.data(), gt_nb.data_handle(), g.size(), stream);
  raft::copy(r.data(), ref_nb.data_handle(), r.size(), stream);
  raft::resource::sync_stream(res);
  int64_t hits = 0, ref_hits = 0;
  for (int64_t i = 0; i < nq; i++) {
    for (int64_t a = 0; a < k; a++) {
      hits += std::count(g.begin() + i * k, g.begin() + (i + 1) * k, h[i * k + a]) > 0;
      ref_hits += std::count(g.begin() + i * k, g.begin() + (i + 1) * k, int64_t(r[i * k + a])) > 0;
    }
  }
  const double recall     = double(hits) / double(nq * k);
  const double ref_recall = double(ref_hits) / double(nq * k);
  EXPECT_GE(recall, ref_recall - p.max_recall_drop)
    << "xyz recall " << recall << ", cagra float recall " << ref_recall;
  std::cout << "xyz recall " << recall << ", cagra float recall " << ref_recall << std::endl;
}

INSTANTIATE_TEST_CASE_P(
  XyzTest,
  XyzTest,
  ::testing::Values(XyzInputs{20000, 128, false, 0.f, 0.03, rotation_mode::kNever},
                    XyzInputs{20000, 128, true, 0.f, 0.03, rotation_mode::kNever},
                    XyzInputs{20000, 128, true, 0.f, 0.03, rotation_mode::kAlways},
                    XyzInputs{20000, 256, true, 0.75f, 0.03, rotation_mode::kAlways},
                    XyzInputs{20000, 100, true, 0.75f, 0.03, rotation_mode::kAuto}));

}  // namespace cuvs::neighbors::xyz
