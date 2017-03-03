// This file is part of the dune-gdt project:
//   https://github.com/dune-community/dune-gdt
// Copyright 2010-2017 dune-gdt developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   Felix Schindler (2017)

#include "config.h"

#if HAVE_DUNE_PYBINDXI

#include <boost/numeric/conversion/cast.hpp>

#include <dune/pybindxi/pybind11.h>
#include <dune/pybindxi/stl.h>

#if HAVE_DUNE_FEM
#include <dune/fem/misc/mpimanager.hh>
#endif

#include <dune/xt/common/string.hh>
#include <dune/xt/common/timedlogging.hh>
#include <dune/xt/common/configuration.pbh>
#include <dune/xt/common/fvector.pbh>

#include <dune/xt/grid/intersection.hh>
#include <dune/xt/grid/type_traits.hh>

#include <dune/xt/la/container.hh>

#include <dune/gdt/functionals/l2.pbh>
#include <dune/gdt/functionals/elliptic-ipdg.pbh>

#include "bindings.hh"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace Dune;
using namespace Dune::GDT::bindings;


template <class I, bool required = true>
struct for_Grid_and_Intersection
{
  template <class G>
  static void addbind(py::module& m, const std::string& grid_id, const std::string& id)
  {
    auto constraints = DirichletConstraints<I>::bind(m, grid_id + id);
//    DirichletConstraints<I>::template addbind<Dune::XT::LA::Backends::common_dense>(constraints);
//#if HAVE_EIGEN
//    DirichletConstraints<I>::template addbind<Dune::XT::LA::Backends::eigen_dense>(constraints);
//    DirichletConstraints<I>::template addbind<Dune::XT::LA::Backends::eigen_sparse>(constraints);
//#endif
#if HAVE_DUNE_ISTL
    DirichletConstraints<I>::template addbind<Dune::XT::LA::Backends::istl_sparse>(constraints);
#endif
  }
};

template <class I>
struct for_Grid_and_Intersection<I, false>
{
  template <class G>
  static void addbind(py::module& /*m*/, const std::string& /*grid_id*/, const std::string& /*id*/)
  {
  }
};


template <class S, class V, class F, bool lagrange = Dune::GDT::is_cg_space<S>::value>
struct addbind_for_lagrange_space
{
  template <class BSA>
  void operator()(py::module& m, BSA& bound_system_assembler, const std::string& space_id, const std::string& la_id)
  {
    // DirichletProjectionLocalizableOperator
    DirichletProjectionLocalizableOperator<typename S::GridViewType, F, Dune::GDT::DiscreteFunction<S, V>>::bind(
        m, space_id, la_id);
    // DirichletConstraints
    DirichletConstraints<typename XT::Grid::Intersection<typename S::GridViewType>::type>::addbind(
        bound_system_assembler);
  } // ... operator()(...)
}; // struct addbind_for_lagrange_space

template <class S, class V, class F>
struct addbind_for_lagrange_space<S, V, F, false>
{
  template <class BSA>
  void operator()(py::module& /*m*/,
                  BSA& /*bound_system_assembler*/,
                  const std::string& /*space_id*/,
                  const std::string& /*la_id*/)
  {
  }
}; // struct addbind_for_lagrange_space< ..., false>


template <class SP>
void addbind_for_space(py::module& m,
                       const std::string& grid_id,
                       const std::string& layer_id,
                       const std::string& space_id,
                       const std::string& backend)
{
  using namespace Dune;
  using namespace Dune::XT;
  using Common::to_string;
  using Common::to_lower;

  typedef typename SP::type S;
  typedef typename XT::Grid::extract_grid<typename S::GridViewType>::type G;
  typedef typename S::EntityType E;
  typedef typename S::DomainFieldType D;
  static const size_t d = S::dimDomain;
  typedef typename S::RangeFieldType R;
  typedef typename LA::Container<R, LA::Backends::istl_sparse>::MatrixType M;
  typedef typename LA::Container<R, LA::Backends::istl_sparse>::VectorType V;
  static const size_t r = S::dimRange;
  static const size_t rC = S::dimRangeCols;
  typedef Functions::LocalizableFunctionInterface<E, D, d, R, 1, 1> ScalarFunction;
  typedef Functions::LocalizableFunctionInterface<E, D, d, R, d, d> TensorFunction;
  const std::string r_ = to_string(r);
  const std::string rC_ = to_string(rC);
  const std::string p_ = to_string(int(S::polOrder)); // without the int(...) we get linker errors on module import
  const std::string space_suffix = r_ + "x" + rC_ + "__p" + p_ + backend;
  // Space
  SpaceInterface<S>::bind(m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix);
  m.def(std::string("make_" + to_lower(space_id) + "_space__" + space_suffix + "__" + layer_id).c_str(),
        [](XT::Grid::GridProvider<G>& grid_provider, const int level) { return SP::create(grid_provider, level); },
        "grid_provider"_a,
        "level"_a = 0,
        py::keep_alive<0, 1>());
// DiscreteFunction
//  ConstDiscreteFunction<S, COMMON_DENSE_VECTOR>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "common_dense");
//  DiscreteFunction<S, COMMON_DENSE_VECTOR>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "common_dense");
//#if HAVE_EIGEN
//  ConstDiscreteFunction<S, EIGEN_DENSE_VECTOR>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_dense");
//  DiscreteFunction<S, EIGEN_DENSE_VECTOR>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_dense");
//#endif // HAVE_EIGEN
#if HAVE_DUNE_ISTL
  ConstDiscreteFunction<S, ISTL_DENSE_VECTOR>::bind(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_dense");
  DiscreteFunction<S, ISTL_DENSE_VECTOR>::bind(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_dense");
#endif // HAVE_DUNE_ISTL
  // SystemAssembler
  auto system_assembler =
      SystemAssembler<S>::bind(m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix);
  addbind_for_lagrange_space<S, COMMON_DENSE_VECTOR, ScalarFunction>()(
      m, system_assembler, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "common_dense");
//#if HAVE_EIGEN
//  addbind_for_lagrange_space<S, EIGEN_DENSE_VECTOR, ScalarFunction>()(
//      m, system_assembler, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_dense");
//#endif
//#if HAVE_DUNE_ISTL
//  addbind_for_lagrange_space<S, ISTL_DENSE_VECTOR, ScalarFunction>()(
//      m, system_assembler, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_dense");
//#endif
// EllipticMatrixOperator
//  EllipticMatrixOperator<ScalarFunction, TensorFunction, S, COMMON_DENSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "common_dense");
//  EllipticMatrixOperator<ScalarFunction, void, S, COMMON_DENSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "common_dense");
//  EllipticMatrixOperator<TensorFunction, void, S, COMMON_DENSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "common_dense");
//#if HAVE_EIGEN
//  EllipticMatrixOperator<ScalarFunction, TensorFunction, S, EIGEN_DENSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_dense");
//  EllipticMatrixOperator<ScalarFunction, void, S, EIGEN_DENSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_dense");
//  EllipticMatrixOperator<TensorFunction, void, S, EIGEN_DENSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_dense");
//  EllipticMatrixOperator<ScalarFunction, TensorFunction, S, EIGEN_SPARSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_sparse");
//  EllipticMatrixOperator<ScalarFunction, void, S, EIGEN_SPARSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_sparse");
//  EllipticMatrixOperator<TensorFunction, void, S, EIGEN_SPARSE_MATRIX>::bind(
//      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "eigen_sparse");
//#endif // HAVE_EIGEN
#if HAVE_DUNE_ISTL
  EllipticMatrixOperator<ScalarFunction, TensorFunction, S, ISTL_SPARSE_MATRIX>::bind(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse");
  EllipticMatrixOperator<ScalarFunction, void, S, ISTL_SPARSE_MATRIX>::bind(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse");
  EllipticMatrixOperator<TensorFunction, void, S, ISTL_SPARSE_MATRIX>::bind(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse");
#endif // HAVE_DUNE_ISTL
// EllipticIpdgMatrixOperator
#define BIND_ELLIPTIT_IPDG_OP(_MAT, _la, _MET, _met)                                                                   \
  EllipticIpdgMatrixOperator<ScalarFunction,                                                                           \
                             TensorFunction,                                                                           \
                             S,                                                                                        \
                             GDT::LocalEllipticIpdgIntegrands::Method::_MET,                                           \
                             _MAT>::bind(m,                                                                            \
                                         space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix,      \
                                         _la,                                                                          \
                                         _met);                                                                        \
  EllipticIpdgMatrixOperator<ScalarFunction, void, S, GDT::LocalEllipticIpdgIntegrands::Method::_MET, _MAT>::bind(     \
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, _la, _met);                          \
  EllipticIpdgMatrixOperator<TensorFunction, void, S, GDT::LocalEllipticIpdgIntegrands::Method::_MET, _MAT>::bind(     \
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, _la, _met)

//  BIND_ELLIPTIT_IPDG_OP(COMMON_DENSE_MATRIX, "common_dense", sipdg, "Sipdg");
//  BIND_ELLIPTIT_IPDG_OP(COMMON_DENSE_MATRIX, "common_dense", swipdg, "Swipdg");
//  BIND_ELLIPTIT_IPDG_OP(COMMON_DENSE_MATRIX, "common_dense", swipdg_affine_factor, "SwipdgAffineFactor");
//  BIND_ELLIPTIT_IPDG_OP(COMMON_DENSE_MATRIX, "common_dense", swipdg_affine_tensor, "SwipdgAffineTensor");
//#if HAVE_EIGEN
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_DENSE_MATRIX, "eigen_dense", sipdg, "Sipdg");
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_DENSE_MATRIX, "eigen_dense", swipdg, "Swipdg");
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_DENSE_MATRIX, "eigen_dense", swipdg_affine_factor, "SwipdgAffineFactor");
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_DENSE_MATRIX, "eigen_dense", swipdg_affine_tensor, "SwipdgAffineTensor");
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_SPARSE_MATRIX, "eigen_sparse", sipdg, "SIPDG");
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_SPARSE_MATRIX, "eigen_sparse", sipdg, "Sipdg");
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_SPARSE_MATRIX, "eigen_sparse", swipdg, "Swipdg");
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_SPARSE_MATRIX, "eigen_sparse", swipdg_affine_factor, "SwipdgAffineFactor");
//  BIND_ELLIPTIT_IPDG_OP(EIGEN_SPARSE_MATRIX, "eigen_sparse", swipdg_affine_tensor, "SwipdgAffineTensor");
//#endif
#if HAVE_DUNE_ISTL
  BIND_ELLIPTIT_IPDG_OP(ISTL_SPARSE_MATRIX, "istl_sparse", sipdg, "Sipdg");
  BIND_ELLIPTIT_IPDG_OP(ISTL_SPARSE_MATRIX, "istl_sparse", swipdg, "Swipdg");
  BIND_ELLIPTIT_IPDG_OP(ISTL_SPARSE_MATRIX, "istl_sparse", swipdg_affine_factor, "SwipdgAffineFactor");
  BIND_ELLIPTIT_IPDG_OP(ISTL_SPARSE_MATRIX, "istl_sparse", swipdg_affine_tensor, "SwipdgAffineTensor");
#endif

  // L2VolumeVectorFunctional
  GDT::bind_l2_volume_vector_functional<ScalarFunction, S, V>(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse");
  // L2FaceVectorFunctional
  GDT::bind_l2_face_vector_functional<ScalarFunction, S, V>(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse");
  // EllipticIpdgDirichletVectorFunctional
  // - SIPDG
  GDT::bind_elliptic_ipdg_dirichlet_vector_functional<ScalarFunction,
                                                      ScalarFunction,
                                                      TensorFunction,
                                                      S,
                                                      GDT::LocalEllipticIpdgIntegrands::Method::sipdg,
                                                      V>(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse", "Sipdg");
  GDT::bind_elliptic_ipdg_dirichlet_vector_functional<ScalarFunction,
                                                      ScalarFunction,
                                                      void,
                                                      S,
                                                      GDT::LocalEllipticIpdgIntegrands::Method::sipdg,
                                                      V>(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse", "Sipdg");
  // - SWIPDG
  GDT::bind_elliptic_ipdg_dirichlet_vector_functional<ScalarFunction,
                                                      ScalarFunction,
                                                      TensorFunction,
                                                      S,
                                                      GDT::LocalEllipticIpdgIntegrands::Method::swipdg,
                                                      V>(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse", "Swipdg");
  GDT::bind_elliptic_ipdg_dirichlet_vector_functional<ScalarFunction,
                                                      ScalarFunction,
                                                      void,
                                                      S,
                                                      GDT::LocalEllipticIpdgIntegrands::Method::swipdg,
                                                      V>(
      m, space_id + "Space__" + grid_id + "_" + layer_id + "_to_" + space_suffix, "istl_sparse", "Swipdg");
  // SWIPDG, affine wrt diffusion factor
  GDT::bind_elliptic_ipdg_dirichlet_vector_functional<ScalarFunction,
                                                      ScalarFunction,
                                                      TensorFunction,
                                                      S,
                                                      GDT::LocalEllipticIpdgIntegrands::Method::swipdg_affine_factor,
                                                      V>(m,
                                                         space_id + "Space__" + grid_id + "_" + layer_id + "_to_"
                                                             + space_suffix,
                                                         "istl_sparse",
                                                         "Swipdg_Affine_Factor");
  GDT::bind_elliptic_ipdg_dirichlet_vector_functional<ScalarFunction,
                                                      ScalarFunction,
                                                      void,
                                                      S,
                                                      GDT::LocalEllipticIpdgIntegrands::Method::swipdg_affine_factor,
                                                      V>(m,
                                                         space_id + "Space__" + grid_id + "_" + layer_id + "_to_"
                                                             + space_suffix,
                                                         "istl_sparse",
                                                         "Swipdg_Affine_Factor");
  // SWIPDG, affine wrt diffusion tensor
  GDT::bind_elliptic_ipdg_dirichlet_vector_functional<ScalarFunction,
                                                      ScalarFunction,
                                                      TensorFunction,
                                                      S,
                                                      GDT::LocalEllipticIpdgIntegrands::Method::swipdg_affine_tensor,
                                                      V>(m,
                                                         space_id + "Space__" + grid_id + "_" + layer_id + "_to_"
                                                             + space_suffix,
                                                         "istl_sparse",
                                                         "Swipdg_Affine_Tensor");
  GDT::bind_elliptic_ipdg_dirichlet_vector_functional<ScalarFunction,
                                                      ScalarFunction,
                                                      void,
                                                      S,
                                                      GDT::LocalEllipticIpdgIntegrands::Method::swipdg_affine_tensor,
                                                      V>(m,
                                                         space_id + "Space__" + grid_id + "_" + layer_id + "_to_"
                                                             + space_suffix,
                                                         "istl_sparse",
                                                         "Swipdg_Affine_Tensor");
} // ... addbind_for_space(...)


template <class G>
void addbind_for_grid(py::module& m, const std::string& grid_id)
{
  using namespace Dune;
  using namespace Dune::XT;
  using namespace Dune::GDT;
//  // FV
//  addbind_for_space<FvSpaceProvider<G, XT::Grid::Layers::leaf, ChooseSpaceBackend::gdt, double, 1, 1>>(
//      m, grid_id, "leaf", "Fv", "");
//  addbind_for_space<FvSpaceProvider<G, XT::Grid::Layers::level, ChooseSpaceBackend::gdt, double, 1, 1>>(
//      m, grid_id, "level", "Fv", "");
// CG
#if HAVE_DUNE_FEM
  addbind_for_space<CgSpaceProvider<G, XT::Grid::Layers::leaf, ChooseSpaceBackend::fem, 1, double, 1, 1>>(
      m, grid_id, "leaf", "Cg", "__fem");
  addbind_for_space<CgSpaceProvider<G, XT::Grid::Layers::level, ChooseSpaceBackend::fem, 1, double, 1, 1>>(
      m, grid_id, "level", "Cg", "__fem");
#endif
//#if HAVE_DUNE_PDELAB
//  addbind_for_space<CgSpaceProvider<G, XT::Grid::Layers::leaf, ChooseSpaceBackend::pdelab, 1, double, 1, 1>>(
//      m, grid_id, "leaf", "Cg", "__pdelab");
//  addbind_for_space<CgSpaceProvider<G, XT::Grid::Layers::level, ChooseSpaceBackend::pdelab, 1, double, 1, 1>>(
//      m, grid_id, "level", "Cg", "__pdelab");
//#endif
// DG
#if HAVE_DUNE_FEM
  addbind_for_space<DgSpaceProvider<G, XT::Grid::Layers::leaf, ChooseSpaceBackend::fem, 1, double, 1, 1>>(
      m, grid_id, "leaf", "Dg", "__fem");
  addbind_for_space<DgSpaceProvider<G, XT::Grid::Layers::level, ChooseSpaceBackend::fem, 1, double, 1, 1>>(
      m, grid_id, "level", "Dg", "__fem");
#endif
  //#if HAVE_DUNE_PDELAB //                                            these need to be guarded against simplicial grids
  //  addbind_for_space<DgSpaceProvider<G, XT::Grid::Layers::leaf, ChooseSpaceBackend::pdelab, 1, double, 1, 1>>(
  //      m, grid_id, "leaf", "Dg", "__pdelab");
  //  addbind_for_space<DgSpaceProvider<G, XT::Grid::Layers::level, ChooseSpaceBackend::pdelab, 1, double, 1, 1>>(
  //      m, grid_id, "level", "Dg", "__pdelab");
  //#endif
  //// RT
  //#if HAVE_DUNE_PDELAB //      enabling those leads to problems in bindings of the functionals and vectors, those need
  //  addbind_for_space<RtSpaceProvider<G, //                                                     to be disabled for bad
  //  dimensions
  //                                    XT::Grid::Layers::leaf,
  //                                    ChooseSpaceBackend::pdelab,
  //                                    0,
  //                                    double,
  //                                    G::dimension,
  //                                    1>>(m, grid_id, "leaf", "Rt", "__pdelab");
  //  addbind_for_space<RtSpaceProvider<G, XT::Grid::Layers::level, ChooseSpaceBackend::pdelab, 0, double, G::dimension,
  //  1>>(
  //      m, grid_id, "level", "Rt", "__pdelab");
  //#endif

  typedef typename XT::Grid::Layer<G, XT::Grid::Layers::level, XT::Grid::Backends::view>::type LevelView;
  typedef typename XT::Grid::Layer<G, XT::Grid::Layers::leaf, XT::Grid::Backends::view>::type LeafView;
  typedef typename XT::Grid::Intersection<LeafView>::type FVI;
  typedef typename XT::Grid::Intersection<LevelView>::type LVI;
#if HAVE_DUNE_FEM
  typedef typename XT::Grid::Layer<G, XT::Grid::Layers::level, XT::Grid::Backends::part>::type LevelPart;
  typedef typename XT::Grid::Layer<G, XT::Grid::Layers::leaf, XT::Grid::Backends::part>::type LeafPart;
  typedef typename XT::Grid::Intersection<LeafPart>::type FPI;
  typedef typename XT::Grid::Intersection<LevelPart>::type LPI;
#endif
  for_Grid_and_Intersection<FVI, true>::template addbind<G>(m,
                                                            grid_id,
                                                            (std::is_same<FVI, LVI>::value
#if HAVE_DUNE_FEM
                                                             && std::is_same<FVI, FPI>::value
                                                             && std::is_same<FVI, LPI>::value
#endif
                                                             )
                                                                ? ""
                                                                : "_leaf_view");
  for_Grid_and_Intersection<LVI, !(std::is_same<LVI, FVI>::value)>::template addbind<G>(m, grid_id, "_level_view");
#if HAVE_DUNE_FEM
  for_Grid_and_Intersection<FPI,
                            !(std::is_same<FPI, FVI>::value
                              || std::is_same<FPI, LVI>::value)>::template addbind<G>(m, grid_id, "_leaf_part");
  for_Grid_and_Intersection<LPI,
                            !(std::is_same<LPI, FVI>::value || std::is_same<LPI, LVI>::value
                              || std::is_same<LPI, FPI>::value)>::template addbind<G>(m, grid_id, "_level_part");
#endif // HAVE_DUNE_FEM
} // ... addbind_for_grid(...)


PYBIND11_PLUGIN(_gdt)
{
  py::module m("_gdt", "dune-gdt");

  py::module::import("dune.xt.common");
  py::module::import("dune.xt.grid");
  py::module::import("dune.xt.functions");
  py::module::import("dune.xt.la");

  m.def("init_mpi",
        [](const std::vector<std::string>& args) {
          int argc = boost::numeric_cast<int>(args.size());
          char** argv = Dune::XT::Common::vector_to_main_args(args);
#if HAVE_DUNE_FEM
          Dune::Fem::MPIManager::initialize(argc, argv);
#else
          Dune::MPIHelper::instance(argc, argv);
#endif
        },
        "args"_a = std::vector<std::string>());

//  addbind_for_grid<Dune::YaspGrid<2, Dune::EquidistantOffsetCoordinates<double, 2>>>(m, "2d_cube_yaspgrid");
#if HAVE_DUNE_ALUGRID
  addbind_for_grid<Dune::ALUGrid<2, 2, Dune::simplex, Dune::conforming>>(m, "2d_simplex_aluconform");
#endif

  m.def("init_logger",
        [](const ssize_t max_info_level,
           const ssize_t max_debug_level,
           const bool enable_warnings,
           const bool enable_colors,
           const std::string& info_color,
           const std::string& debug_color,
           const std::string& warning_color) {
          Dune::XT::Common::TimedLogger().create(
              max_info_level, max_debug_level, enable_warnings, enable_colors, info_color, debug_color, warning_color);
        },
        "max_info_level"_a = -1,
        "max_debug_level"_a = -1,
        "enable_warnings"_a = true,
        "enable_colors"_a = true,
        "info_color"_a = "blue",
        "debug_color"_a = "darkgray",
        "warning_color"_a = "red");

  return m.ptr();
}


#endif // HAVE_DUNE_PYBINDXI
