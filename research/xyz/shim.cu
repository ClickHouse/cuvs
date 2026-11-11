// Research shim: ctypes-friendly wrappers around libcuvs CAGRA.
#include <raft/linalg/map.cuh>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <chrono>
#include <cstdio>
#include <memory>

using namespace cuvs::neighbors;

static raft::device_resources& R()
{
  static raft::device_resources r;
  return r;
}

struct shim_handle {
  std::unique_ptr<cagra::device_padded_index<float, uint32_t>> idx;
};

extern "C" {

void* shim_index(const float* data, int64_t n, int64_t dim, const uint32_t* graph, int64_t deg)
{
  cudaDeviceSynchronize();
  auto& res = R();
  auto dv   = raft::make_device_matrix_view<const float, int64_t>(data, n, dim);
  auto view = make_device_padded_dataset_view(res, dv);
  auto gv   = raft::make_device_matrix_view<const uint32_t, int64_t>(graph, n, deg);
  auto* h   = new shim_handle;
  h->idx    = std::make_unique<cagra::device_padded_index<float, uint32_t>>(
    res, cuvs::distance::DistanceType::L2Expanded, view, gv);
  res.sync_stream();
  return h;
}

void shim_free(void* h) { delete static_cast<shim_handle*>(h); }

// returns elapsed milliseconds (synchronous, end-to-end)
double shim_search(void* hp,
                   const float* q,
                   int64_t nq,
                   int64_t dim,
                   int k,
                   int itopk,
                   int width,
                   int algo,
                   int max_iter,
                   int num_random_samplings,
                   unsigned long long rand_xor_mask,
                   uint32_t* nbr,
                   float* dist)
{
  auto* h = static_cast<shim_handle*>(hp);
  auto& res = R();
  cagra::search_params sp;
  sp.itopk_size           = itopk;
  sp.search_width         = width;
  sp.algo                 = algo < 0 ? cagra::search_algo::AUTO : (cagra::search_algo)algo;
  sp.max_iterations       = max_iter;
  sp.num_random_samplings = num_random_samplings;
  if (rand_xor_mask) sp.rand_xor_mask = rand_xor_mask;
  auto qv = raft::make_device_matrix_view<const float, int64_t>(q, nq, dim);
  auto nv = raft::make_device_matrix_view<uint32_t, int64_t>(nbr, nq, k);
  auto dv = raft::make_device_matrix_view<float, int64_t>(dist, nq, k);
  cudaDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  cagra::search(res, sp, *h->idx, qv, nv, dv);
  res.sync_stream();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Exact kNN (sqeuclidean) of queries against data.
void shim_bf_knn(const float* data,
                 int64_t n,
                 int64_t dim,
                 const float* q,
                 int64_t nq,
                 int k,
                 int64_t* nbr,
                 float* dist)
{
  cudaDeviceSynchronize();
  auto& res = R();
  auto dv   = raft::make_device_matrix_view<const float, int64_t>(data, n, dim);
  auto idx  = brute_force::build(res, dv, cuvs::distance::DistanceType::L2Expanded);
  const int64_t B = 16384;
  for (int64_t s = 0; s < nq; s += B) {
    int64_t m = std::min(B, nq - s);
    brute_force::search(res,
                        idx,
                        raft::make_device_matrix_view<const float, int64_t>(q + s * dim, m, dim),
                        raft::make_device_matrix_view<int64_t, int64_t>(nbr + s * k, m, k),
                        raft::make_device_matrix_view<float, int64_t>(dist + s * k, m, k));
  }
  res.sync_stream();
}

// Stock CAGRA optimize (device graphs).
void shim_optimize(const uint32_t* knn, int64_t n, int64_t kin, uint32_t* out, int64_t kout, int conn)
{
  cudaDeviceSynchronize();
  auto& res = R();
  cagra::helpers::optimize(
    res,
    raft::make_device_matrix_view<uint32_t, int64_t>(const_cast<uint32_t*>(knn), n, kin),
    raft::make_device_matrix_view<uint32_t, int64_t>(out, n, kout),
    conn != 0);
  res.sync_stream();
}

// Stock CAGRA build (default knn-graph algo) -> graph copied into out [n, deg]
void shim_build(const float* data, int64_t n, int64_t dim, int64_t ideg, int64_t deg, uint32_t* out)
{
  cudaDeviceSynchronize();
  auto& res = R();
  cagra::index_params ip;
  ip.intermediate_graph_degree = ideg;
  ip.graph_degree              = deg;
  auto dv  = raft::make_device_matrix_view<const float, int64_t>(data, n, dim);
  auto idx = cagra::build(res, ip, make_device_padded_dataset_view(res, dv));
  raft::copy(out, idx.graph().data_handle(), n * deg, res.get_stream());
  res.sync_stream();
}
}

extern "C" void shim_ivfpq_knn(const float* host_data, int64_t n, int64_t dim, uint32_t* host_out, int64_t K)
{
  auto& res = R();
  auto dv   = raft::make_host_matrix_view<const float, int64_t>(host_data, n, dim);
  auto ov   = raft::make_host_matrix_view<uint32_t, int64_t>(host_out, n, K);
  cagra::graph_build_params::ivf_pq_params p(raft::matrix_extent<int64_t>(n, dim),
                                             cuvs::distance::DistanceType::L2Expanded);
  cagra::build_knn_graph(res, dv, ov, p);
  res.sync_stream();
}

#include <cuvs/neighbors/nn_descent.hpp>
extern "C" void shim_nnd_knn(const float* dev_data, int64_t n, int64_t dim, uint32_t* host_out, int64_t K)
{
  auto& res = R();
  cuvs::neighbors::nn_descent::index_params p(K);
  p.return_distances = false;
  auto dv = raft::make_device_matrix_view<const float, int64_t>(dev_data, n, dim);
  auto ov = raft::make_host_matrix_view<uint32_t, int64_t>(host_out, n, K);
  auto idx = cuvs::neighbors::nn_descent::build(res, p, dv, std::make_optional(ov));
  res.sync_stream();
}

#include <cuvs/neighbors/refine.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>
struct shim_vpq {
  std::unique_ptr<cuvs::neighbors::device_vpq_dataset<half, int64_t>> ds;
  std::unique_ptr<cagra::device_pq_index<float, uint32_t, half>> idx;
};
extern "C" void* shim_vpq_index(const float* data, int64_t n, int64_t dim, const uint32_t* graph,
                                int64_t deg, int pq_dim, int vq_centers)
{
  cudaDeviceSynchronize();
  auto& res = R();
  cuvs::neighbors::vpq_params vp;
  vp.pq_bits = 8; vp.pq_dim = pq_dim; vp.vq_n_centers = vq_centers;
  auto dv = raft::make_device_matrix_view<const float, int64_t>(data, n, dim);
  auto* h = new shim_vpq;
  h->ds = std::make_unique<cuvs::neighbors::device_vpq_dataset<half, int64_t>>(
    cuvs::preprocessing::quantize::pq::make_vpq_dataset(res, vp, dv));
  auto gv = raft::make_device_matrix_view<const uint32_t, int64_t>(graph, n, deg);
  h->idx = std::make_unique<cagra::device_pq_index<float, uint32_t, half>>(
    res, cuvs::distance::DistanceType::L2Expanded, h->ds->view(), gv);
  res.sync_stream();
  return h;
}
extern "C" void shim_vpq_free(void* h) { delete static_cast<shim_vpq*>(h); }
// search on codes for `cand` candidates, then exact refine to k; returns ms (end to end)
extern "C" double shim_vpq_search(void* hp, const float* data, int64_t n, const float* q, int64_t nq,
                                  int64_t dim, int k, int cand, int itopk, uint32_t* tmp_nb,
                                  float* tmp_d, int64_t* tmp_nb64, int64_t* out_nb, float* out_d)
{
  auto* h = static_cast<shim_vpq*>(hp);
  auto& res = R();
  cagra::search_params sp; sp.itopk_size = itopk; sp.algo = cagra::search_algo::SINGLE_CTA;
  auto qv = raft::make_device_matrix_view<const float, int64_t>(q, nq, dim);
  cudaDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  cagra::search(res, sp, *h->idx, qv, raft::make_device_matrix_view<uint32_t, int64_t>(tmp_nb, nq, cand),
                raft::make_device_matrix_view<float, int64_t>(tmp_d, nq, cand));
  raft::linalg::map(res, raft::make_device_vector_view<int64_t, int64_t>(tmp_nb64, nq * cand),
                    [] __device__(uint32_t x) { return static_cast<int64_t>(x); },
                    raft::make_device_vector_view<const uint32_t, int64_t>(tmp_nb, nq * cand));
  cuvs::neighbors::refine(res, raft::make_device_matrix_view<const float, int64_t>(data, n, dim), qv,
                          raft::make_device_matrix_view<const int64_t, int64_t>(tmp_nb64, nq, cand),
                          raft::make_device_matrix_view<int64_t, int64_t>(out_nb, nq, k),
                          raft::make_device_matrix_view<float, int64_t>(out_d, nq, k),
                          cuvs::distance::DistanceType::L2Expanded);
  res.sync_stream();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

extern "C" double shim_vpq_search_only(void* hp, const float* q, int64_t nq, int64_t dim, int cand, int itopk,
                                       uint32_t* tmp_nb, float* tmp_d)
{
  auto* h = static_cast<shim_vpq*>(hp);
  auto& res = R();
  cagra::search_params sp; sp.itopk_size = itopk; sp.algo = cagra::search_algo::SINGLE_CTA;
  auto qv = raft::make_device_matrix_view<const float, int64_t>(q, nq, dim);
  cudaDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  cagra::search(res, sp, *h->idx, qv, raft::make_device_matrix_view<uint32_t, int64_t>(tmp_nb, nq, cand),
                raft::make_device_matrix_view<float, int64_t>(tmp_d, nq, cand));
  res.sync_stream();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
extern "C" double shim_refine_only(const float* data, int64_t n, const float* q, int64_t nq, int64_t dim, int k,
                                   int cand, const int64_t* nb64, int64_t* out_nb, float* out_d)
{
  auto& res = R();
  auto qv = raft::make_device_matrix_view<const float, int64_t>(q, nq, dim);
  cudaDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  cuvs::neighbors::refine(res, raft::make_device_matrix_view<const float, int64_t>(data, n, dim), qv,
                          raft::make_device_matrix_view<const int64_t, int64_t>(nb64, nq, cand),
                          raft::make_device_matrix_view<int64_t, int64_t>(out_nb, nq, k),
                          raft::make_device_matrix_view<float, int64_t>(out_d, nq, k),
                          cuvs::distance::DistanceType::L2Expanded);
  res.sync_stream();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

#include <cuvs/selection/select_k.hpp>
extern "C" void shim_select_k(const float* vals, int64_t rows, int64_t cols, int k, float* out_v, int64_t* out_i)
{
  cudaDeviceSynchronize();
  auto& res = R();
  cuvs::selection::select_k(res,
                            raft::make_device_matrix_view<const float, int64_t>(vals, rows, cols),
                            std::nullopt,
                            raft::make_device_matrix_view<float, int64_t>(out_v, rows, k),
                            raft::make_device_matrix_view<int64_t, int64_t>(out_i, rows, k),
                            true);
  res.sync_stream();
}

#include <cuvs/neighbors/ivf_pq.hpp>
struct shim_ivfpq { std::unique_ptr<cuvs::neighbors::ivf_pq::index<int64_t>> idx; };
extern "C" void* shim_ivfpq_build(const float* data, int64_t n, int64_t dim, int n_lists, int pq_dim, int pq_bits, double* build_ms)
{
  cudaDeviceSynchronize();
  auto& res = R();
  cuvs::neighbors::ivf_pq::index_params p;
  p.n_lists = n_lists; p.pq_dim = pq_dim; p.pq_bits = pq_bits; p.kmeans_trainset_fraction = 0.1;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto* h = new shim_ivfpq;
  h->idx = std::make_unique<cuvs::neighbors::ivf_pq::index<int64_t>>(cuvs::neighbors::ivf_pq::build(
    res, p, raft::make_device_matrix_view<const float, int64_t>(data, n, dim)));
  res.sync_stream();
  *build_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
  return h;
}
extern "C" void shim_ivfpq_free(void* h) { delete static_cast<shim_ivfpq*>(h); }
extern "C" double shim_ivfpq_search(void* hp, const float* q, int64_t nq, int64_t dim, int k, int n_probes,
                                    int64_t* nb, float* dd, int fp16)
{
  auto* h = static_cast<shim_ivfpq*>(hp);
  auto& res = R();
  cuvs::neighbors::ivf_pq::search_params sp; sp.n_probes = n_probes;
  if (fp16) { sp.lut_dtype = CUDA_R_16F; sp.internal_distance_dtype = CUDA_R_16F; }
  cudaDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  cuvs::neighbors::ivf_pq::search(res, sp, *h->idx, raft::make_device_matrix_view<const float, int64_t>(q, nq, dim),
                                  raft::make_device_matrix_view<int64_t, int64_t>(nb, nq, k),
                                  raft::make_device_matrix_view<float, int64_t>(dd, nq, k));
  res.sync_stream();
  return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

struct shim_handle_i8 { std::unique_ptr<cagra::device_padded_index<int8_t, uint32_t>> idx; };
extern "C" void* shim_index_i8(const int8_t* data, int64_t n, int64_t dim, const uint32_t* graph, int64_t deg)
{
  cudaDeviceSynchronize();
  auto& res = R();
  auto dv   = raft::make_device_matrix_view<const int8_t, int64_t>(data, n, dim);
  auto view = make_device_padded_dataset_view(res, dv);
  auto gv   = raft::make_device_matrix_view<const uint32_t, int64_t>(graph, n, deg);
  auto* h   = new shim_handle_i8;
  h->idx    = std::make_unique<cagra::device_padded_index<int8_t, uint32_t>>(
    res, cuvs::distance::DistanceType::L2Expanded, view, gv);
  res.sync_stream();
  return h;
}
extern "C" void shim_free_i8(void* h) { delete static_cast<shim_handle_i8*>(h); }
extern "C" double shim_search_i8(void* hp, const int8_t* q, int64_t nq, int64_t dim, int k, int itopk, int team,
                                 uint32_t* nbr, float* dist)
{
  auto* h = static_cast<shim_handle_i8*>(hp);
  auto& res = R();
  cagra::search_params sp; sp.itopk_size = itopk; sp.algo = cagra::search_algo::SINGLE_CTA; sp.team_size = team;
  cudaDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  cagra::search(res, sp, *h->idx, raft::make_device_matrix_view<const int8_t, int64_t>(q, nq, dim),
                raft::make_device_matrix_view<uint32_t, int64_t>(nbr, nq, k),
                raft::make_device_matrix_view<float, int64_t>(dist, nq, k));
  res.sync_stream();
  return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

#include <cuvs/neighbors/xyz.hpp>
struct shim_xyz { std::unique_ptr<cuvs::neighbors::xyz::index> idx; };
extern "C" void* shim_xyz_build(const float* data, int64_t n, int64_t dim, int deg, int ideg, int pack,
                                double* build_ms, int rotate)
{
  cudaDeviceSynchronize();
  auto& res = R();
  cuvs::neighbors::xyz::index_params ip;
  ip.cagra_params.graph_degree              = deg;
  ip.cagra_params.intermediate_graph_degree = ideg;
  ip.cagra_params.graph_build_params = cagra::graph_build_params::nn_descent_params(ideg);
  ip.pack_graph = pack != 0;
  ip.rotation   = rotate == 1   ? cuvs::neighbors::xyz::rotation_mode::kAlways
                  : rotate == 0 ? cuvs::neighbors::xyz::rotation_mode::kNever
                                : cuvs::neighbors::xyz::rotation_mode::kAuto;
  auto t0 = std::chrono::high_resolution_clock::now();
  auto* h = new shim_xyz;
  h->idx  = std::make_unique<cuvs::neighbors::xyz::index>(cuvs::neighbors::xyz::build(
    res, ip, raft::make_device_matrix_view<const float, int64_t>(data, n, dim)));
  res.sync_stream();
  *build_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
  return h;
}
extern "C" void shim_xyz_free(void* h) { delete static_cast<shim_xyz*>(h); }
extern "C" size_t shim_xyz_memory(void* h) { return static_cast<shim_xyz*>(h)->idx->memory_bytes(); }
extern "C" double shim_xyz_search(void* hp, const float* q, int64_t nq, int64_t dim, int k, int itopk, int team,
                                  float eps, int refine_c, int64_t* nb, float* dd)
{
  auto* h = static_cast<shim_xyz*>(hp);
  auto& res = R();
  cuvs::neighbors::xyz::search_params sp;
  sp.cagra_params.itopk_size = itopk; sp.cagra_params.team_size = team;
  sp.ads_eps = eps; sp.refine_candidates = refine_c;
  cudaDeviceSynchronize();
  auto t0 = std::chrono::high_resolution_clock::now();
  cuvs::neighbors::xyz::search(res, sp, *h->idx, raft::make_device_matrix_view<const float, int64_t>(q, nq, dim),
                               raft::make_device_matrix_view<int64_t, int64_t>(nb, nq, k),
                               raft::make_device_matrix_view<float, int64_t>(dd, nq, k));
  res.sync_stream();
  return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}
