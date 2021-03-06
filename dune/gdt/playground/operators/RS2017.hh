// This file is part of the dune-gdt project:
//   https://github.com/dune-community/dune-gdt
// Copyright 2010-2017 dune-gdt developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   Felix Schindler (2017)

#ifndef DUNE_GDT_PLAYGROUND_OPERATORS_RS2017_HH
#define DUNE_GDT_PLAYGROUND_OPERATORS_RS2017_HH

#include <type_traits>

#include <dune/common/typetraits.hh>

#include <dune/pybindxi/pybind11.h>
#include <dune/pybindxi/stl.h>

#include <dune/xt/common/numeric_cast.hh>
#include <dune/xt/la/container/istl.hh>
#include <dune/xt/grid/dd/subdomains/grid.hh>
#include <dune/xt/grid/grids.bindings.hh>
#include <dune/xt/grid/layers.hh>
#include <dune/xt/grid/type_traits.hh>
#include <dune/xt/functions/interfaces/localizable-function.hh>

#include <dune/gdt/assembler/system.bindings.hh>
#include <dune/gdt/functionals/elliptic-ipdg.hh>
#include <dune/gdt/functionals/l2.hh>
#include <dune/gdt/local/integrands/elliptic.hh>
#include <dune/gdt/local/integrands/elliptic-ipdg.hh>
#include <dune/gdt/local/integrands/lambda.hh>
#include <dune/gdt/local/operators/integrals.hh>
#include <dune/gdt/operators/base.hh>
#include <dune/gdt/operators/base.bindings.hh>
#include <dune/gdt/operators/elliptic-ipdg.hh>
#include <dune/gdt/operators/fluxreconstruction.hh>
#include <dune/gdt/operators/l2.hh>
#include <dune/gdt/spaces.hh>

namespace Dune {
namespace GDT {
namespace RS2017 {

#if HAVE_DUNE_FEM && HAVE_DUNE_ISTL && HAVE_DUNE_PDELAB


template <class G>
class DiffusiveFluxAaProduct
    : public GDT::MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                                     typename GDT::SpaceProvider<G,
                                                                 XT::Grid::Layers::dd_subdomain,
                                                                 GDT::SpaceType::dg,
                                                                 GDT::Backends::fem,
                                                                 1,
                                                                 double,
                                                                 1>::type,
                                     typename XT::Grid::Layer<G,
                                                              XT::Grid::Layers::dd_subdomain,
                                                              XT::Grid::Backends::part,
                                                              XT::Grid::DD::SubdomainGrid<G>>::type>
{
  static_assert(XT::Grid::is_grid<G>::value, "");
  typedef typename GDT::
      SpaceProvider<G, XT::Grid::Layers::dd_subdomain, GDT::SpaceType::dg, GDT::Backends::fem, 1, double, 1>
          SP;
  typedef GDT::MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                                  typename SP::type,
                                  typename XT::Grid::Layer<G,
                                                           XT::Grid::Layers::dd_subdomain,
                                                           XT::Grid::Backends::part,
                                                           XT::Grid::DD::SubdomainGrid<G>>::type>
      BaseType;
  typedef DiffusiveFluxAaProduct<G> ThisType;

public:
  using typename BaseType::GridLayerType;
  using typename BaseType::RangeSpaceType;

  typedef XT::Grid::extract_entity_t<GridLayerType> E;
  typedef XT::Grid::extract_intersection_t<GridLayerType> I;
  typedef typename G::ctype D;
  static const constexpr size_t d = G::dimension;
  typedef double R;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1> ScalarFunctionType;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, d, d> TensorFunctionType;
  typedef typename RangeSpaceType::BaseFunctionSetType BasisType;

  static void bind(pybind11::module& m)
  {
    namespace py = pybind11;
    using namespace pybind11::literals;

    GDT::bindings::MatrixOperatorBase<ThisType>::bind(
        m,
        XT::Common::to_camel_case("RS2017_diffusive_flux_aa_product_matrix_operator_subdomain_"
                                  + XT::Grid::bindings::grid_name<G>::value())
            .c_str());

    m.def("RS2017_make_diffusive_flux_aa_product_matrix_operator_on_subdomain",
          [](const XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
             const ssize_t subdomain,
             const RangeSpaceType& space,
             const ScalarFunctionType& lambda_hat,
             const ScalarFunctionType& lambda_u,
             const ScalarFunctionType& lambda_v,
             const TensorFunctionType& kappa,
             const size_t over_integrate) {
            return new ThisType(
                space,
                dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain, XT::Grid::Backends::part>(
                    XT::Common::numeric_cast<size_t>(subdomain)),
                lambda_hat,
                lambda_u,
                lambda_v,
                kappa,
                over_integrate);
          },
          "dd_grid_provider"_a,
          "subdomain"_a,
          "space"_a,
          "lambda_hat"_a,
          "lambda_u"_a,
          "lambda_v"_a,
          "kappa"_a,
          "over_integrate"_a = 2);
  } // ... bind(...)

  DiffusiveFluxAaProduct(RangeSpaceType space,
                         GridLayerType grd_lyr,
                         const ScalarFunctionType& lambda_hat,
                         const ScalarFunctionType& lambda_u,
                         const ScalarFunctionType& lambda_v,
                         const TensorFunctionType& kappa,
                         const size_t over_integrate = 2)
    : BaseType(space, grd_lyr)
    , unit_matrix_(XT::Common::from_string<XT::Common::FieldMatrix<D, d, d>>("[1 0 0; 0 1 0; 0 0 1]"))
    , over_integrate_(over_integrate)
    , local_operator_(
          [&](const auto& test_base, const auto& ansatz_base) {
            const auto& entity = test_base.entity();
            const auto local_lambda_hat = lambda_hat.local_function(entity);
            const auto local_lambda_u = lambda_u.local_function(entity);
            const auto local_lambda_v = lambda_v.local_function(entity);
            const auto local_kappa = kappa.local_function(entity);
            const auto integrand_order = local_lambda_hat->order() + local_lambda_u->order() + local_lambda_v->order()
                                         + 3 * local_kappa->order()
                                         + size_t(std::max(ssize_t(test_base.order()) - 1, ssize_t(0)))
                                         + size_t(std::max(ssize_t(ansatz_base.order()) - 1, ssize_t(0)));
            return integrand_order + over_integrate_;
          },
          [&](const auto& test_base, const auto& ansatz_base, const auto& local_point, auto& ret) {
            const auto& entity = test_base.entity();
            const auto local_lambda_hat = lambda_hat.local_function(entity);
            const auto local_lambda_u = lambda_u.local_function(entity);
            const auto local_lambda_v = lambda_v.local_function(entity);
            const auto local_kappa = kappa.local_function(entity);
            XT::Common::FieldMatrix<D, d, d> diffusion_hat_inverse = local_kappa->evaluate(local_point);
            XT::Common::FieldMatrix<D, d, d> diffusion_u = diffusion_hat_inverse;
            XT::Common::FieldMatrix<D, d, d> diffusion_v = diffusion_hat_inverse;
            diffusion_hat_inverse *= local_lambda_hat->evaluate(local_point);
#ifndef NDEBUG
            const auto diffusion_hat = diffusion_hat_inverse;
#endif
            diffusion_hat_inverse.invert();
#ifndef NDEBUG
            // there is no documented way to tell if the inversion was successfull
            if (XT::Common::FloatCmp::ne(diffusion_hat_inverse * diffusion_hat, unit_matrix_))
              DUNE_THROW(XT::Common::Exceptions::internal_error,
                         "Local inversion of lambda_hat*kappa failed!\n\nx = "
                             << local_point
                             << "\nlocal_lambda_hat(x) = "
                             << local_lambda_hat->evaluate(local_point)
                             << "\nlocal_kappa(x) = "
                             << local_kappa->evaluate(local_point)
                             << "\ninverse = "
                             << diffusion_hat_inverse
                             << "\ninverse * (local_lambda_hat(x)*local_kappa(x))) = "
                             << diffusion_hat_inverse * diffusion_hat);
#endif
            diffusion_u *= local_lambda_u->evaluate(local_point);
            diffusion_v *= local_lambda_v->evaluate(local_point);
            const auto test_grads = test_base.jacobian(local_point);
            const auto ansatz_grads = ansatz_base.jacobian(local_point);
            ret *= 0.;
            for (size_t ii = 0; ii < test_base.size(); ++ii)
              for (size_t jj = 0; jj < ansatz_base.size(); ++jj)
                ret[ii][jj] =
                    ((diffusion_hat_inverse * (diffusion_u * ansatz_grads[jj][0])) * (diffusion_v * test_grads[ii][0]));
          })
  {
    this->append(local_operator_);
  }

  DiffusiveFluxAaProduct(const ThisType&) = delete;
  DiffusiveFluxAaProduct(ThisType&&) = delete;

private:
  const XT::Common::FieldMatrix<D, d, d> unit_matrix_;
  const size_t over_integrate_;
  const GDT::LocalVolumeIntegralOperator<GDT::LocalLambdaBinaryVolumeIntegrand<E, R, 1>, BasisType> local_operator_;
}; // class DiffusiveFluxAaProduct


template <class G>
class DiffusiveFluxAbProduct
    : public GDT::
          MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                             typename GDT::SpaceProvider<G,
                                                         XT::Grid::Layers::dd_subdomain,
                                                         GDT::SpaceType::dg,
                                                         GDT::Backends::fem,
                                                         1,
                                                         double,
                                                         1>::type,
                             typename XT::Grid::Layer<G,
                                                      XT::Grid::Layers::dd_subdomain,
                                                      XT::Grid::Backends::part,
                                                      XT::Grid::DD::SubdomainGrid<G>>::type,
                             GDT::RestrictedSpace<typename GDT::SpaceProvider<G,
                                                                              XT::Grid::Layers::leaf,
                                                                              GDT::SpaceType::rt,
                                                                              GDT::Backends::pdelab,
                                                                              0,
                                                                              double,
                                                                              G::dimension>::type,
                                                  typename XT::Grid::Layer<G,
                                                                           XT::Grid::Layers::dd_subdomain,
                                                                           XT::Grid::Backends::part,
                                                                           XT::Grid::DD::SubdomainGrid<G>>::type>>
{
  static_assert(XT::Grid::is_grid<G>::value, "");
  typedef GDT::MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                                  typename GDT::SpaceProvider<G,
                                                              XT::Grid::Layers::dd_subdomain,
                                                              GDT::SpaceType::dg,
                                                              GDT::Backends::fem,
                                                              1,
                                                              double,
                                                              1>::type,
                                  typename XT::Grid::Layer<G,
                                                           XT::Grid::Layers::dd_subdomain,
                                                           XT::Grid::Backends::part,
                                                           XT::Grid::DD::SubdomainGrid<G>>::type,
                                  GDT::RestrictedSpace<typename GDT::SpaceProvider<G,
                                                                                   XT::Grid::Layers::leaf,
                                                                                   GDT::SpaceType::rt,
                                                                                   GDT::Backends::pdelab,
                                                                                   0,
                                                                                   double,
                                                                                   G::dimension>::type,
                                                       typename XT::Grid::Layer<G,
                                                                                XT::Grid::Layers::dd_subdomain,
                                                                                XT::Grid::Backends::part,
                                                                                XT::Grid::DD::SubdomainGrid<G>>::type>>
      BaseType;
  typedef DiffusiveFluxAbProduct<G> ThisType;

public:
  using typename BaseType::GridLayerType;
  using typename BaseType::RangeSpaceType;
  using typename BaseType::RangeBaseType;
  using typename BaseType::SourceSpaceType;
  using typename BaseType::SourceBaseType;

  typedef XT::Grid::extract_entity_t<GridLayerType> E;
  typedef XT::Grid::extract_intersection_t<GridLayerType> I;
  typedef typename G::ctype D;
  static const constexpr size_t d = G::dimension;
  typedef double R;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1> ScalarFunctionType;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, d, d> TensorFunctionType;

  static void bind(pybind11::module& m)
  {
    namespace py = pybind11;
    using namespace pybind11::literals;

    py::class_<ThisType, XT::Grid::Walker<GridLayerType>> c(
        m,
        XT::Common::to_camel_case("RS2017_diffusive_flux_ab_product_matrix_operator_subdomain_"
                                  + XT::Grid::bindings::grid_name<G>::value())
            .c_str());
    c.def("assemble", [](ThisType& self) { self.assemble(); });
    c.def("matrix", [](ThisType& self) { return self.matrix(); });

    m.def("RS2017_make_diffusive_flux_ab_product_matrix_operator_on_subdomain",
          [](const XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
             const ssize_t subdomain,
             const RangeSpaceType& range_space,
             const SourceSpaceType& source_space,
             const ScalarFunctionType& lambda_hat,
             const ScalarFunctionType& lambda_range,
             const TensorFunctionType& kappa,
             const size_t over_integrate) {
            return new ThisType(
                range_space,
                source_space,
                dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain, XT::Grid::Backends::part>(
                    XT::Common::numeric_cast<size_t>(subdomain)),
                lambda_hat,
                lambda_range,
                kappa,
                over_integrate);
          },
          "dd_grid_provider"_a,
          "subdomain"_a,
          "range_space"_a,
          "source_space"_a,
          "lambda_hat"_a,
          "lambda_range"_a,
          "kappa"_a,
          "over_integrate"_a = 2);
  } // ... bind(...)

  DiffusiveFluxAbProduct(RangeSpaceType range_space,
                         SourceSpaceType source_space,
                         GridLayerType grd_lyr,
                         const ScalarFunctionType& lambda_hat,
                         const ScalarFunctionType& lambda_range,
                         const TensorFunctionType& kappa,
                         const size_t over_integrate = 2)
    : BaseType(range_space, source_space, grd_lyr)
    , unit_matrix_(XT::Common::from_string<XT::Common::FieldMatrix<D, d, d>>("[1 0 0; 0 1 0; 0 0 1]"))
    , over_integrate_(over_integrate)
    , local_operator_(
          [&](const auto& test_base, const auto& ansatz_base) {
            const auto& entity = test_base.entity();
            const auto local_lambda_hat = lambda_hat.local_function(entity);
            const auto local_lambda_range = lambda_range.local_function(entity);
            const auto local_kappa = kappa.local_function(entity);
            const auto integrand_order =
                local_lambda_hat->order() + local_lambda_range->order() + 2 * local_kappa->order()
                + size_t(std::max(ssize_t(test_base.order()) - 1, ssize_t(0))) + ansatz_base.order();
            return integrand_order + over_integrate_;
          },
          [&](const auto& test_base, const auto& ansatz_base, const auto& local_point, auto& ret) {
            const auto& entity = test_base.entity();
            const auto local_lambda_hat = lambda_hat.local_function(entity);
            const auto local_lambda_range = lambda_range.local_function(entity);
            const auto local_kappa = kappa.local_function(entity);
            XT::Common::FieldMatrix<D, d, d> diffusion_hat_inverse = local_kappa->evaluate(local_point);
            XT::Common::FieldMatrix<D, d, d> diffusion_range = diffusion_hat_inverse;
            diffusion_hat_inverse *= local_lambda_hat->evaluate(local_point);
#ifndef NDEBUG
            const auto diffusion_hat = diffusion_hat_inverse;
#endif
            diffusion_hat_inverse.invert();
#ifndef NDEBUG
            // there is no documented way to tell if the inversion was successfull
            if (XT::Common::FloatCmp::ne(diffusion_hat_inverse * diffusion_hat, unit_matrix_))
              DUNE_THROW(XT::Common::Exceptions::internal_error,
                         "Local inversion of lambda_hat*kappa failed!\n\nx = "
                             << local_point
                             << "\nlocal_lambda_hat(x) = "
                             << local_lambda_hat->evaluate(local_point)
                             << "\nlocal_kappa(x) = "
                             << local_kappa->evaluate(local_point)
                             << "\ninverse = "
                             << diffusion_hat_inverse
                             << "\ninverse * (local_lambda_hat(x)*local_kappa(x))) = "
                             << diffusion_hat_inverse * diffusion_hat);
#endif
            diffusion_range *= local_lambda_range->evaluate(local_point);
            const auto test_grads = test_base.jacobian(local_point);
            const auto ansatz_values = ansatz_base.evaluate(local_point);
            ret *= 0.;
            for (size_t ii = 0; ii < test_base.size(); ++ii)
              for (size_t jj = 0; jj < ansatz_base.size(); ++jj)
                ret[ii][jj] = ((diffusion_hat_inverse * (diffusion_range * test_grads[ii][0])) * (ansatz_values[jj]));
          })
  {
    this->append(local_operator_);
  }

  DiffusiveFluxAbProduct(const ThisType&) = delete;
  DiffusiveFluxAbProduct(ThisType&&) = delete;

private:
  const XT::Common::FieldMatrix<D, d, d> unit_matrix_;
  const size_t over_integrate_;
  const GDT::LocalVolumeIntegralOperator<GDT::LocalLambdaBinaryVolumeIntegrand<E, R, 1, 1, d, 1>,
                                         RangeBaseType,
                                         SourceBaseType>
      local_operator_;
}; // class DiffusiveFluxAbProduct


template <class G>
class DiffusiveFluxBbProduct
    : public GDT::
          MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                             GDT::RestrictedSpace<typename GDT::SpaceProvider<G,
                                                                              XT::Grid::Layers::leaf,
                                                                              GDT::SpaceType::rt,
                                                                              GDT::Backends::pdelab,
                                                                              0,
                                                                              double,
                                                                              G::dimension>::type,
                                                  typename XT::Grid::Layer<G,
                                                                           XT::Grid::Layers::dd_subdomain,
                                                                           XT::Grid::Backends::part,
                                                                           XT::Grid::DD::SubdomainGrid<G>>::type>,
                             typename XT::Grid::Layer<G,
                                                      XT::Grid::Layers::dd_subdomain,
                                                      XT::Grid::Backends::part,
                                                      XT::Grid::DD::SubdomainGrid<G>>::type>
{
  static_assert(XT::Grid::is_grid<G>::value, "");
  typedef GDT::MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                                  GDT::RestrictedSpace<typename GDT::SpaceProvider<G,
                                                                                   XT::Grid::Layers::leaf,
                                                                                   GDT::SpaceType::rt,
                                                                                   GDT::Backends::pdelab,
                                                                                   0,
                                                                                   double,
                                                                                   G::dimension>::type,
                                                       typename XT::Grid::Layer<G,
                                                                                XT::Grid::Layers::dd_subdomain,
                                                                                XT::Grid::Backends::part,
                                                                                XT::Grid::DD::SubdomainGrid<G>>::type>,
                                  typename XT::Grid::Layer<G,
                                                           XT::Grid::Layers::dd_subdomain,
                                                           XT::Grid::Backends::part,
                                                           XT::Grid::DD::SubdomainGrid<G>>::type>
      BaseType;
  typedef DiffusiveFluxBbProduct<G> ThisType;

public:
  using typename BaseType::GridLayerType;
  using typename BaseType::RangeSpaceType;
  using typename BaseType::RangeBaseType;

  typedef XT::Grid::extract_entity_t<GridLayerType> E;
  typedef XT::Grid::extract_intersection_t<GridLayerType> I;
  typedef typename G::ctype D;
  static const constexpr size_t d = G::dimension;
  typedef double R;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1> ScalarFunctionType;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, d, d> TensorFunctionType;

  static void bind(pybind11::module& m)
  {
    namespace py = pybind11;
    using namespace pybind11::literals;

    py::class_<ThisType, XT::Grid::Walker<GridLayerType>> c(
        m,
        XT::Common::to_camel_case("RS2017_diffusive_flux_bb_product_matrix_operator_subdomain_"
                                  + XT::Grid::bindings::grid_name<G>::value())
            .c_str());
    c.def("assemble", [](ThisType& self) { self.assemble(); });
    c.def("matrix", [](ThisType& self) { return self.matrix(); });

    m.def("RS2017_make_diffusive_flux_bb_product_matrix_operator_on_subdomain",
          [](const XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
             const ssize_t subdomain,
             const RangeSpaceType& space,
             const ScalarFunctionType& lambda_hat,
             const TensorFunctionType& kappa,
             const size_t over_integrate) {
            return new ThisType(
                space,
                dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain, XT::Grid::Backends::part>(
                    XT::Common::numeric_cast<size_t>(subdomain)),
                lambda_hat,
                kappa,
                over_integrate);
          },
          "dd_grid_provider"_a,
          "subdomain"_a,
          "space"_a,
          "lambda_hat"_a,
          "kappa"_a,
          "over_integrate"_a = 2);
  } // ... bind(...)

  DiffusiveFluxBbProduct(RangeSpaceType space,
                         GridLayerType grd_lyr,
                         const ScalarFunctionType& lambda_hat,
                         const TensorFunctionType& kappa,
                         const size_t over_integrate = 2)
    : BaseType(space, grd_lyr)
    , unit_matrix_(XT::Common::from_string<XT::Common::FieldMatrix<D, d, d>>("[1 0 0; 0 1 0; 0 0 1]"))
    , over_integrate_(over_integrate)
    , local_operator_(
          [&](const auto& test_base, const auto& ansatz_base) {
            const auto& entity = test_base.entity();
            const auto local_lambda_hat = lambda_hat.local_function(entity);
            const auto local_kappa = kappa.local_function(entity);
            const auto integrand_order =
                local_lambda_hat->order() + local_kappa->order() + test_base.order() + ansatz_base.order();
            return integrand_order + over_integrate_;
          },
          [&](const auto& test_base, const auto& ansatz_base, const auto& local_point, auto& ret) {
            const auto& entity = test_base.entity();
            const auto local_lambda_hat = lambda_hat.local_function(entity);
            const auto local_kappa = kappa.local_function(entity);
            XT::Common::FieldMatrix<D, d, d> diffusion_hat_inverse = local_kappa->evaluate(local_point);
            diffusion_hat_inverse *= local_lambda_hat->evaluate(local_point);
#ifndef NDEBUG
            const auto diffusion_hat = diffusion_hat_inverse;
#endif
            diffusion_hat_inverse.invert();
#ifndef NDEBUG
            // there is no documented way to tell if the inversion was successfull
            if (XT::Common::FloatCmp::ne(diffusion_hat_inverse * diffusion_hat, unit_matrix_))
              DUNE_THROW(XT::Common::Exceptions::internal_error,
                         "Local inversion of lambda_hat*kappa failed!\n\nx = "
                             << local_point
                             << "\nlocal_lambda_hat(x) = "
                             << local_lambda_hat->evaluate(local_point)
                             << "\nlocal_kappa(x) = "
                             << local_kappa->evaluate(local_point)
                             << "\ninverse = "
                             << diffusion_hat_inverse
                             << "\ninverse * (local_lambda_hat(x)*local_kappa(x))) = "
                             << diffusion_hat_inverse * diffusion_hat);
#endif
            const auto test_values = test_base.evaluate(local_point);
            const auto ansatz_values = ansatz_base.evaluate(local_point);
            ret *= 0.;
            for (size_t ii = 0; ii < test_base.size(); ++ii)
              for (size_t jj = 0; jj < ansatz_base.size(); ++jj)
                ret[ii][jj] = ((diffusion_hat_inverse * test_values[ii]) * ansatz_values[jj]);
          })
  {
    this->append(local_operator_);
  }

  DiffusiveFluxBbProduct(const ThisType&) = delete;
  DiffusiveFluxBbProduct(ThisType&&) = delete;

private:
  const XT::Common::FieldMatrix<D, d, d> unit_matrix_;
  const size_t over_integrate_;
  const GDT::LocalVolumeIntegralOperator<GDT::LocalLambdaBinaryVolumeIntegrand<E, R, d, 1>, RangeBaseType>
      local_operator_;
}; // class DiffusiveFluxBbProduct


template <class G>
class SwipdgPenaltySubdomainProduct
    : public GDT::MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                                     typename GDT::SpaceProvider<G,
                                                                 XT::Grid::Layers::dd_subdomain,
                                                                 GDT::SpaceType::dg,
                                                                 GDT::Backends::fem,
                                                                 1,
                                                                 double,
                                                                 1>::type,
                                     typename XT::Grid::Layer<G,
                                                              XT::Grid::Layers::dd_subdomain,
                                                              XT::Grid::Backends::part,
                                                              XT::Grid::DD::SubdomainGrid<G>>::type>
{
  static_assert(XT::Grid::is_grid<G>::value, "");
  typedef GDT::SpaceProvider<G, XT::Grid::Layers::dd_subdomain, GDT::SpaceType::dg, GDT::Backends::fem, 1, double, 1>
      SP;
  typedef GDT::MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                                  typename SP::type,
                                  typename XT::Grid::Layer<G,
                                                           XT::Grid::Layers::dd_subdomain,
                                                           XT::Grid::Backends::part,
                                                           XT::Grid::DD::SubdomainGrid<G>>::type>
      BaseType;
  typedef SwipdgPenaltySubdomainProduct<G> ThisType;

public:
  using typename BaseType::GridLayerType;
  using typename BaseType::RangeSpaceType;

  typedef XT::Grid::extract_entity_t<GridLayerType> E;
  typedef XT::Grid::extract_intersection_t<GridLayerType> I;
  typedef typename G::ctype D;
  static const constexpr size_t d = G::dimension;
  typedef double R;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1> ScalarFunctionType;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, d, d> TensorFunctionType;
  typedef typename RangeSpaceType::BaseFunctionSetType BasisType;

  static void bind(pybind11::module& m)
  {
    namespace py = pybind11;
    using namespace pybind11::literals;

    GDT::bindings::MatrixOperatorBase<ThisType>::bind(
        m,
        XT::Common::to_camel_case("RS2017_penalty_product_matrix_operator_subdomain_"
                                  + XT::Grid::bindings::grid_name<G>::value())
            .c_str());

    m.def("RS2017_make_penalty_product_matrix_operator_on_subdomain",
          [](const XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
             const ssize_t subdomain,
             const XT::Grid::BoundaryInfo<I>& boundary_info,
             const RangeSpaceType& space,
             const ScalarFunctionType& lambda,
             const TensorFunctionType& kappa,
             const size_t over_integrate) {
            return new ThisType(
                space,
                dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain, XT::Grid::Backends::part>(
                    XT::Common::numeric_cast<size_t>(subdomain)),
                boundary_info,
                lambda,
                kappa,
                over_integrate);
          },
          "dd_grid_provider"_a,
          "subdomain"_a,
          "boundary_info"_a,
          "space"_a,
          "lambda_bar"_a,
          "kappa"_a,
          "over_integrate"_a = 2);
  } // ... bind(...)

  SwipdgPenaltySubdomainProduct(RangeSpaceType space,
                                GridLayerType grd_lyr,
                                const XT::Grid::BoundaryInfo<I>& boundary_info,
                                const ScalarFunctionType& lambda,
                                const TensorFunctionType& kappa,
                                const size_t over_integrate = 2)
    : BaseType(space, grd_lyr)
    , lambda_(lambda)
    , kappa_(kappa)
    , over_integrate_(over_integrate)
    , local_coupling_operator_(
          // the order lambda
          [&](const auto& test_base_en,
              const auto& ansatz_base_en,
              const auto& test_base_ne,
              const auto& ansatz_base_ne) {
            const auto& entity = test_base_en.entity();
            const auto& neighbor = test_base_ne.entity();
            const auto local_lambda_en = lambda_.local_function(entity);
            const auto local_lambda_ne = lambda_.local_function(neighbor);
            const auto local_kappa_en = kappa_.local_function(entity);
            const auto local_kappa_ne = kappa_.local_function(neighbor);
            const auto integrand_order = std::max(local_lambda_en->order(), local_lambda_ne->order())
                                         + std::max(local_kappa_en->order(), local_kappa_ne->order())
                                         + std::max(test_base_en.order(), test_base_ne.order())
                                         + std::max(ansatz_base_en.order(), ansatz_base_ne.order());
            return integrand_order + over_integrate_;
          },
          // The evaluate lambda, this is the penalty part of LocalEllipticIpdgIntegrands::Inner from
          // dune/gdt/local/integrands/elliptic-ipdg.hh, swipdg_affine_factor variant.
          [&](const auto& test_base_en,
              const auto& ansatz_base_en,
              const auto& test_base_ne,
              const auto& ansatz_base_ne,
              const auto& intersection,
              const auto& local_point,
              auto& ret_en_en,
              auto& ret_ne_ne,
              auto& ret_en_ne,
              auto& ret_ne_en) {
            // clear ret
            ret_en_en *= 0.0;
            ret_ne_ne *= 0.0;
            ret_en_ne *= 0.0;
            ret_ne_en *= 0.0;
            // convert local point (which is in intersection coordinates) to entity/neighbor coordinates
            const auto local_point_en = intersection.geometryInInside().global(local_point);
            const auto local_point_ne = intersection.geometryInOutside().global(local_point);
            const auto normal = intersection.unitOuterNormal(local_point);
            // evaluate local function
            const auto& entity = test_base_en.entity();
            const auto& neighbor = test_base_ne.entity();
            const auto lambda_val_en = lambda_.local_function(entity)->evaluate(local_point_en);
            const auto lambda_val_ne = lambda_.local_function(neighbor)->evaluate(local_point_ne);
            const XT::Common::FieldMatrix<D, d, d> kappa_val_en =
                kappa_.local_function(entity)->evaluate(local_point_en);
            const XT::Common::FieldMatrix<D, d, d> kappa_val_ne =
                kappa_.local_function(neighbor)->evaluate(local_point_ne);
            // compute penalty
            const size_t max_polorder =
                std::max(test_base_en.order(),
                         std::max(ansatz_base_en.order(), std::max(test_base_ne.order(), ansatz_base_ne.order())));
            const R sigma = GDT::LocalEllipticIpdgIntegrands::internal::inner_sigma(max_polorder);
            const R delta_plus = normal * (kappa_val_ne * normal);
            const R delta_minus = normal * (kappa_val_en * normal);
            const R gamma = (delta_plus * delta_minus) / (delta_plus + delta_minus);
            const auto h = intersection.geometry().volume();
            const auto beta = GDT::LocalEllipticIpdgIntegrands::internal::default_beta(d);
            const R penalty = (0.5 * (lambda_val_en + lambda_val_ne) * sigma * gamma) / std::pow(h, beta);
            // evaluate bases
            const size_t rows_en = test_base_en.size();
            const size_t cols_en = ansatz_base_en.size();
            const size_t rows_ne = test_base_ne.size();
            const size_t cols_ne = ansatz_base_ne.size();
            const auto test_values_en = test_base_en.evaluate(local_point_en);
            const auto ansatz_values_en = ansatz_base_en.evaluate(local_point_en);
            const auto test_values_ne = test_base_ne.evaluate(local_point_ne);
            const auto ansatz_values_ne = ansatz_base_ne.evaluate(local_point_ne);
            // compute integrals, loop over all combinations of basis functions
            assert(ret_en_en.rows() >= rows_en && ret_en_en.cols() >= cols_en);
            assert(ret_en_ne.rows() >= rows_en && ret_en_ne.cols() >= cols_ne);
            assert(ret_ne_en.rows() >= rows_ne && ret_ne_en.cols() >= cols_en);
            assert(ret_ne_ne.rows() >= rows_ne && ret_ne_ne.cols() >= cols_ne);
            for (size_t ii = 0; ii < rows_en; ++ii) {
              for (size_t jj = 0; jj < cols_en; ++jj)
                ret_en_en[ii][jj] += penalty * ansatz_values_en[jj] * test_values_en[ii];
              for (size_t jj = 0; jj < cols_ne; ++jj)
                ret_en_ne[ii][jj] += -1.0 * penalty * ansatz_values_ne[jj] * test_values_en[ii];
            }
            for (size_t ii = 0; ii < rows_ne; ++ii) {
              for (size_t jj = 0; jj < cols_en; ++jj)
                ret_ne_en[ii][jj] += -1.0 * penalty * ansatz_values_en[jj] * test_values_ne[ii];
              for (size_t jj = 0; jj < cols_ne; ++jj)
                ret_ne_ne[ii][jj] += penalty * ansatz_values_ne[jj] * test_values_ne[ii];
            }
          })
    , local_boundary_operator_(
          // the order lambda
          [&](const auto& test_base, const auto& ansatz_base) {
            const auto& entity = test_base.entity();
            const auto local_lambda = lambda_.local_function(entity);
            const auto local_kappa = kappa_.local_function(entity);
            const auto integrand_order =
                local_lambda->order() + local_kappa->order() + test_base.order() + ansatz_base.order();
            return integrand_order + over_integrate_;
          },
          // The evaluate lambda, this is the penalty part of LocalEllipticIpdgIntegrands::BoundaryLHS from
          // dune/gdt/local/integrands/elliptic-ipdg.hh, swipdg_affine_factor variant.
          [&](const auto& test_base,
              const auto& ansatz_base,
              const auto& intersection,
              const auto& local_point,
              auto& ret) {
            // clear ret
            ret *= 0.0;
            // get local point (which is in intersection coordinates) in entity coordinates
            const auto local_point_entity = intersection.geometryInInside().global(local_point);
            const auto normal = intersection.unitOuterNormal(local_point);
            // evaluate local functions
            const auto& entity = test_base.entity();
            XT::Common::FieldMatrix<D, d, d> diffusion = kappa_.local_function(entity)->evaluate(local_point_entity);
            diffusion *= lambda_.local_function(entity)->evaluate(local_point_entity);
            // compute penalty
            const size_t max_polorder = std::max(test_base.order(), ansatz_base.order());
            const R sigma = GDT::LocalEllipticIpdgIntegrands::internal::boundary_sigma(max_polorder);
            const R gamma = normal * (diffusion * normal);
            const auto h = intersection.geometry().volume();
            const auto beta = GDT::LocalEllipticIpdgIntegrands::internal::default_beta(d);
            const R penalty = (sigma * gamma) / std::pow(h, beta);
            // evaluate bases
            const size_t rows = test_base.size();
            const size_t cols = ansatz_base.size();
            const auto test_values = test_base.evaluate(local_point_entity);
            const auto ansatz_values = ansatz_base.evaluate(local_point_entity);
            // compute integrals, loop over all combinations of basis functions
            assert(ret.rows() >= rows && ret.cols() >= cols);
            for (size_t ii = 0; ii < rows; ++ii)
              for (size_t jj = 0; jj < cols; ++jj)
                ret[ii][jj] += penalty * ansatz_values[jj] * test_values[ii];
          })
  {
    this->append(local_coupling_operator_, new XT::Grid::ApplyOn::InnerIntersectionsPrimally<GridLayerType>());
    this->append(local_boundary_operator_, new XT::Grid::ApplyOn::DirichletIntersections<GridLayerType>(boundary_info));
  }

  SwipdgPenaltySubdomainProduct(const ThisType&) = delete;
  SwipdgPenaltySubdomainProduct(ThisType&&) = delete;

private:
  const ScalarFunctionType& lambda_;
  const TensorFunctionType& kappa_;
  const size_t over_integrate_;
  const GDT::LocalCouplingIntegralOperator<GDT::LocalLambdaQuaternaryFaceIntegrand<E, I>, BasisType, I>
      local_coupling_operator_;
  const GDT::LocalBoundaryIntegralOperator<GDT::LocalLambdaBinaryFaceIntegrand<E, I>, BasisType, I>
      local_boundary_operator_;
}; // class SwipdgPenaltySubdomainProduct


template <class G>
void bind_neighborhood_discretization(pybind11::module& m)
{
  namespace py = pybind11;
  using namespace pybind11::literals;

  typedef typename G::template Codim<0>::Entity E;
  typedef double D;
  static const constexpr size_t d = 2;
  typedef double R;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1> DF;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, d, d> DT;

  typedef typename XT::Grid::Layer<G,
                                   XT::Grid::Layers::dd_subdomain_oversampled,
                                   XT::Grid::Backends::part,
                                   XT::Grid::DD::SubdomainGrid<G>>::type NGL;
  typedef GDT::
      SpaceProvider<G, XT::Grid::Layers::dd_subdomain, GDT::SpaceType::block_dg, GDT::Backends::fem, 1, double, 1>
          SP;
  typedef typename SP::type S;
  typedef XT::LA::IstlDenseVector<R> V;
  typedef XT::LA::IstlRowMajorSparseMatrix<R> M;

  try { // we might not be the first to add this SystemAssembler
    GDT::bindings::SystemAssembler<SP, XT::Grid::Layers::dd_subdomain_oversampled, XT::Grid::Backends::part>::bind(m);
  } catch (std::runtime_error&) {
  }

  m.def("RS2017_make_neighborhood_system_assembler",
        [](XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
           const ssize_t subdomain,
           const S& neighborhood_space) {
          return new GDT::SystemAssembler<S, NGL>(
              neighborhood_space,
              dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain_oversampled, XT::Grid::Backends::part>(
                  XT::Common::numeric_cast<size_t>(subdomain)));
        });

  typedef GDT::
      EllipticIpdgMatrixOperator<DF, DT, S, GDT::LocalEllipticIpdgIntegrands::Method::swipdg_affine_factor, M, NGL>
          DgMatrixOperator;
  py::class_<DgMatrixOperator, GDT::SystemAssembler<S, NGL>> dg_matrix_operator(
      m, "EllipticIpdgMatrixOperatorNeighborhood");
  dg_matrix_operator.def("matrix", [](DgMatrixOperator& self) { return self.matrix(); });

  m.def("RS2017_make_elliptic_swipdg_matrix_operator_on_neighborhood",
        [](XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
           const ssize_t subdomain,
           XT::Grid::BoundaryInfo<XT::Grid::extract_intersection_t<NGL>>& boundary_info,
           const S& neighborhood_space,
           const DF& lambda,
           const DT& kappa,
           const ssize_t over_integrate) {
          return new DgMatrixOperator(
              over_integrate,
              boundary_info,
              lambda,
              kappa,
              neighborhood_space,
              dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain_oversampled, XT::Grid::Backends::part>(
                  XT::Common::numeric_cast<size_t>(subdomain)));
        },
        "dd_grid_provider"_a,
        "subdomain"_a,
        "boundary_info"_a,
        "neighborhood_space"_a,
        "lambda"_a,
        "kappa"_a,
        "over_integrate"_a = 2);

  typedef GDT::EllipticIpdgDirichletVectorFunctional<DF,
                                                     DF,
                                                     DT,
                                                     S,
                                                     GDT::LocalEllipticIpdgIntegrands::Method::swipdg_affine_factor,
                                                     V,
                                                     NGL>
      DgVectorFunctional;
  py::class_<DgVectorFunctional, GDT::SystemAssembler<S, NGL>> dg_vector_functional(
      m, "EllipticSwipdgVectorFunctionalNeighborhood");
  dg_vector_functional.def("vector", [](DgVectorFunctional& self) { return self.vector(); });

  m.def("RS2017_make_elliptic_swipdg_vector_functional_on_neighborhood",
        [](XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
           const ssize_t subdomain,
           XT::Grid::BoundaryInfo<XT::Grid::extract_intersection_t<NGL>>& boundary_info,
           const S& neighborhood_space,
           const DF& g_D,
           const DF& lambda,
           const DT& kappa,
           const ssize_t over_integrate) {
          return new DgVectorFunctional(
              over_integrate,
              boundary_info,
              g_D,
              lambda,
              kappa,
              neighborhood_space,
              dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain_oversampled, XT::Grid::Backends::part>(
                  XT::Common::numeric_cast<size_t>(subdomain)));
        },
        "dd_grid_provider"_a,
        "subdomain"_a,
        "boundary_info"_a,
        "neighborhood_space"_a,
        "g_D"_a,
        "lambda"_a,
        "kappa"_a,
        "over_integrate"_a = 2);

  typedef GDT::L2VolumeVectorFunctional<DF, S, V, NGL> L2VolumeFunctional;
  py::class_<L2VolumeFunctional, GDT::SystemAssembler<S, NGL>> l2_volume_vector_functional(
      m, "L2VolumeVectorFunctionalNeighborhood");
  l2_volume_vector_functional.def("vector", [](L2VolumeFunctional& self) { return self.vector(); });

  m.def("RS2017_make_l2_vector_functional_on_neighborhood",
        [](XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
           const ssize_t subdomain,
           const S& neighborhood_space,
           const DF& f,
           const ssize_t over_integrate) {
          return new L2VolumeFunctional(
              over_integrate,
              f,
              neighborhood_space,
              dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain_oversampled, XT::Grid::Backends::part>(
                  XT::Common::numeric_cast<size_t>(subdomain)));
        },
        "dd_grid_provider"_a,
        "subdomain"_a,
        "neighborhood_space"_a,
        "f"_a,
        "over_integrate"_a = 2);
} // ... bind_neighborhood_discretization(...)


template <class G>
class HdivSemiProduct
    : public GDT::
          MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                             GDT::RestrictedSpace<typename GDT::SpaceProvider<G,
                                                                              XT::Grid::Layers::leaf,
                                                                              GDT::SpaceType::rt,
                                                                              GDT::Backends::pdelab,
                                                                              0,
                                                                              double,
                                                                              G::dimension>::type,
                                                  typename XT::Grid::Layer<G,
                                                                           XT::Grid::Layers::dd_subdomain,
                                                                           XT::Grid::Backends::part,
                                                                           XT::Grid::DD::SubdomainGrid<G>>::type>,
                             typename XT::Grid::Layer<G,
                                                      XT::Grid::Layers::dd_subdomain,
                                                      XT::Grid::Backends::part,
                                                      XT::Grid::DD::SubdomainGrid<G>>::type>
{
  static_assert(XT::Grid::is_grid<G>::value, "");
  typedef GDT::MatrixOperatorBase<XT::LA::IstlRowMajorSparseMatrix<double>,
                                  GDT::RestrictedSpace<typename GDT::SpaceProvider<G,
                                                                                   XT::Grid::Layers::leaf,
                                                                                   GDT::SpaceType::rt,
                                                                                   GDT::Backends::pdelab,
                                                                                   0,
                                                                                   double,
                                                                                   G::dimension>::type,
                                                       typename XT::Grid::Layer<G,
                                                                                XT::Grid::Layers::dd_subdomain,
                                                                                XT::Grid::Backends::part,
                                                                                XT::Grid::DD::SubdomainGrid<G>>::type>,
                                  typename XT::Grid::Layer<G,
                                                           XT::Grid::Layers::dd_subdomain,
                                                           XT::Grid::Backends::part,
                                                           XT::Grid::DD::SubdomainGrid<G>>::type>
      BaseType;
  typedef HdivSemiProduct<G> ThisType;

public:
  using typename BaseType::GridLayerType;
  using typename BaseType::RangeSpaceType;

  typedef XT::Grid::extract_entity_t<GridLayerType> E;
  typedef XT::Grid::extract_intersection_t<GridLayerType> I;
  typedef typename G::ctype D;
  static const constexpr size_t d = G::dimension;
  typedef double R;
  typedef typename RangeSpaceType::BaseFunctionSetType BasisType;

  static void bind(pybind11::module& m)
  {
    namespace py = pybind11;
    using namespace pybind11::literals;

    py::class_<ThisType, XT::Grid::Walker<GridLayerType>> c(
        m,
        XT::Common::to_camel_case("RS2017_Hdiv_semi_product_matrix_operator_subdomain_"
                                  + XT::Grid::bindings::grid_name<G>::value())
            .c_str());
    c.def("assemble", [](ThisType& self) { self.assemble(); });
    c.def("matrix", [](ThisType& self) { return self.matrix(); });

    m.def("RS2017_make_Hdiv_semi_product_matrix_operator_on_subdomain",
          [](const XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
             const ssize_t subdomain,
             const RangeSpaceType& space,
             const size_t over_integrate) {
            return new ThisType(
                space,
                dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain, XT::Grid::Backends::part>(
                    XT::Common::numeric_cast<size_t>(subdomain)),
                over_integrate);
          },
          "dd_grid_provider"_a,
          "subdomain"_a,
          "space"_a,
          "over_integrate"_a = 2);
  } // ... bind(...)

  HdivSemiProduct(RangeSpaceType space, GridLayerType grd_lyr, const size_t over_integrate = 2)
    : BaseType(space, grd_lyr)
    , over_integrate_(over_integrate)
    , local_operator_(
          [&](const auto& test_base, const auto& ansatz_base) {
            const auto integrand_order = std::max(ssize_t(test_base.order()) - 1, ssize_t(0))
                                         + std::max(ssize_t(ansatz_base.order()) - 1, ssize_t(0));
            return size_t(integrand_order) + over_integrate_;
          },
          [&](const auto& test_base, const auto& ansatz_base, const auto& local_point, auto& ret) {
            const auto test_gradient = test_base.jacobian(local_point);
            const auto ansatz_gradient = ansatz_base.jacobian(local_point);
            for (size_t ii = 0; ii < test_base.size(); ++ii)
              for (size_t jj = 0; jj < ansatz_base.size(); ++jj) {
                R test_divergence = 0.;
                R ansatz_divergence = 0.;
                for (size_t dd = 0; dd < d; ++dd) {
                  test_divergence += test_gradient[ii][dd][dd];
                  ansatz_divergence += ansatz_gradient[jj][dd][dd];
                }
                ret[ii][jj] = test_divergence * ansatz_divergence;
              }
          })
  {
    this->append(local_operator_);
  }

  HdivSemiProduct(const ThisType&) = delete;
  HdivSemiProduct(ThisType&&) = delete;

private:
  const size_t over_integrate_;
  const GDT::LocalVolumeIntegralOperator<GDT::LocalLambdaBinaryVolumeIntegrand<E, R, d>, BasisType> local_operator_;
}; // class HdivSemiProduct


template <class G>
class ResidualPartFunctional
    : public GDT::
          VectorFunctionalBase<XT::LA::IstlDenseVector<double>,
                               GDT::RestrictedSpace<typename GDT::SpaceProvider<G,
                                                                                XT::Grid::Layers::leaf,
                                                                                GDT::SpaceType::rt,
                                                                                GDT::Backends::pdelab,
                                                                                0,
                                                                                double,
                                                                                G::dimension>::type,
                                                    typename XT::Grid::Layer<G,
                                                                             XT::Grid::Layers::dd_subdomain,
                                                                             XT::Grid::Backends::part,
                                                                             XT::Grid::DD::SubdomainGrid<G>>::type>,
                               typename XT::Grid::Layer<G,
                                                        XT::Grid::Layers::dd_subdomain,
                                                        XT::Grid::Backends::part,
                                                        XT::Grid::DD::SubdomainGrid<G>>::type>
{
  static_assert(XT::Grid::is_grid<G>::value, "");
  typedef GDT::RestrictedSpace<
      typename GDT::
          SpaceProvider<G, XT::Grid::Layers::leaf, GDT::SpaceType::rt, GDT::Backends::pdelab, 0, double, G::dimension>::
              type,
      typename XT::Grid::
          Layer<G, XT::Grid::Layers::dd_subdomain, XT::Grid::Backends::part, XT::Grid::DD::SubdomainGrid<G>>::type>
      RtSpaceType;
  typedef GDT::VectorFunctionalBase<XT::LA::IstlDenseVector<double>,
                                    RtSpaceType,
                                    typename XT::Grid::Layer<G,
                                                             XT::Grid::Layers::dd_subdomain,
                                                             XT::Grid::Backends::part,
                                                             XT::Grid::DD::SubdomainGrid<G>>::type>
      BaseType;
  typedef ResidualPartFunctional<G> ThisType;

public:
  using typename BaseType::GridLayerType;
  using typename BaseType::SpaceType;

  typedef XT::Grid::extract_entity_t<GridLayerType> E;
  typedef XT::Grid::extract_intersection_t<GridLayerType> I;
  typedef typename G::ctype D;
  static const constexpr size_t d = G::dimension;
  typedef double R;
  typedef XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1> ScalarFunctionType;
  typedef typename SpaceType::BaseFunctionSetType BasisType;

  static void bind(pybind11::module& m)
  {
    namespace py = pybind11;
    using namespace pybind11::literals;

    py::class_<ThisType, XT::Grid::Walker<GridLayerType>> c(
        m,
        XT::Common::to_camel_case("RS2017_residual_part_vector_functional_subdomain_"
                                  + XT::Grid::bindings::grid_name<G>::value())
            .c_str());
    c.def("assemble", [](ThisType& self) { self.assemble(); });
    c.def("vector", [](ThisType& self) { return self.vector(); });

    m.def("RS2017_make_residual_part_vector_functional_on_subdomain",
          [](const XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
             const ssize_t subdomain,
             const SpaceType& space,
             const ScalarFunctionType& f,
             const size_t over_integrate) {
            return new ThisType(
                space,
                dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain, XT::Grid::Backends::part>(
                    XT::Common::numeric_cast<size_t>(subdomain)),
                f,
                over_integrate);
          },
          "dd_grid_provider"_a,
          "subdomain"_a,
          "space"_a,
          "f"_a,
          "over_integrate"_a = 2);
  } // ... bind(...)

  ResidualPartFunctional(SpaceType space,
                         GridLayerType grd_lyr,
                         const ScalarFunctionType& f,
                         const size_t over_integrate = 0)
    : BaseType(space, grd_lyr)
    , f_(f)
    , over_integrate_(over_integrate)
    , local_functional_(
          [&](const auto& test_base) {
            const auto integrand_order = ssize_t(f_.local_function(test_base.entity())->order())
                                         + std::max(ssize_t(test_base.order()) - 1, ssize_t(0));
            return size_t(integrand_order) + over_integrate_;
          },
          [&](const auto& test_base, const auto& local_point, auto& ret) {
            const auto local_f = f_.local_function(test_base.entity());
            ret *= 0.;
            // f \times \divergence test
            const auto test_jacobians = test_base.jacobian(local_point);
            for (size_t ii = 0; ii < test_base.size(); ++ii)
              for (size_t dd = 0; dd < d; ++dd)
                ret[ii] += test_jacobians[ii][dd][dd];
            ret *= local_f->evaluate(local_point);
          })
  {
    this->append(local_functional_);
  }

  ResidualPartFunctional(const ThisType&) = delete;
  ResidualPartFunctional(ThisType&&) = delete;

private:
  const ScalarFunctionType& f_;
  const size_t over_integrate_;
  const GDT::LocalVolumeIntegralFunctional<GDT::LocalLambdaUnaryVolumeIntegrand<E, R, d, 1>, BasisType>
      local_functional_;
}; // class ResidualPartFunctional


template <class G>
void bind_neighborhood_reconstruction(pybind11::module& m)
{
  namespace py = pybind11;
  using namespace pybind11::literals;

  typedef typename G::template Codim<0>::Entity E;
  typedef double D;
  static const constexpr size_t d = 2;
  typedef double R;
  typedef typename XT::Grid::Layer<G, XT::Grid::Layers::leaf, XT::Grid::Backends::view>::type LeafViewType;
  typedef GDT::DunePdelabRtSpaceWrapper<LeafViewType, 0, double, d> RtSpaceType;
  typedef typename XT::Grid::Layer<G,
                                   XT::Grid::Layers::dd_subdomain_oversampled,
                                   XT::Grid::Backends::part,
                                   XT::Grid::DD::SubdomainGrid<G>>::type NeighborHoodGridLayer;
  typedef GDT::RestrictedSpace<RtSpaceType, NeighborHoodGridLayer> NeighborhoodRtSpaceType;
  typedef XT::LA::IstlDenseVector<R> VectorType;
  typedef GDT::
      LocalizableDiffusiveFluxReconstructionOperator<NeighborHoodGridLayer,
                                                     XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1>,
                                                     GDT::DiscreteFunction<NeighborhoodRtSpaceType, VectorType>,
                                                     GDT::LocalEllipticIpdgIntegrands::Method::swipdg_affine_factor>
          LocalizableDiffusiveFluxReconstructionOperatorForRestrictedSpaceType;

  m.def("RS2017_apply_diffusive_flux_reconstruction_in_neighborhood",
        [](XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
           const ssize_t subdomain,
           const XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1>& lambda,
           const XT::Functions::LocalizableFunctionInterface<E, D, d, R, d, d>& kappa,
           const XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1>& u,
           typename LocalizableDiffusiveFluxReconstructionOperatorForRestrictedSpaceType::RangeType& reconstructed_u,
           const ssize_t over_integrate) {
          py::gil_scoped_release DUNE_UNUSED(release);
          LocalizableDiffusiveFluxReconstructionOperatorForRestrictedSpaceType(
              dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain_oversampled, XT::Grid::Backends::part>(
                  XT::Common::numeric_cast<size_t>(subdomain)),
              lambda,
              kappa,
              u,
              reconstructed_u,
              over_integrate)
              .apply();
        },
        "dd_grid_provider"_a,
        "subdomain"_a,
        "lambda_hat"_a,
        "kappa"_a,
        "u"_a,
        "reconstructed_u"_a,
        "over_integrate"_a = 2);

  typedef GDT::
      LocalizableDiffusiveFluxReconstructionOperator<NeighborHoodGridLayer,
                                                     XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1>,
                                                     GDT::DiscreteFunction<RtSpaceType, VectorType>,
                                                     GDT::LocalEllipticIpdgIntegrands::Method::swipdg_affine_factor>
          LocalizableDiffusiveFluxReconstructionOperatorForLeafSpaceType;

  m.def("RS2017_apply_diffusive_flux_reconstruction_in_neighborhood",
        [](XT::Grid::GridProvider<G, XT::Grid::DD::SubdomainGrid<G>>& dd_grid_provider,
           const ssize_t subdomain,
           const XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1>& lambda,
           const XT::Functions::LocalizableFunctionInterface<E, D, d, R, d, d>& kappa,
           const XT::Functions::LocalizableFunctionInterface<E, D, d, R, 1>& u,
           typename LocalizableDiffusiveFluxReconstructionOperatorForLeafSpaceType::RangeType& reconstructed_u,
           const ssize_t over_integrate) {
          py::gil_scoped_release DUNE_UNUSED(release);
          LocalizableDiffusiveFluxReconstructionOperatorForLeafSpaceType(
              dd_grid_provider.template layer<XT::Grid::Layers::dd_subdomain_oversampled, XT::Grid::Backends::part>(
                  XT::Common::numeric_cast<size_t>(subdomain)),
              lambda,
              kappa,
              u,
              reconstructed_u,
              over_integrate)
              .apply();
        },
        "dd_grid_provider"_a,
        "subdomain"_a,
        "lambda_hat"_a,
        "kappa"_a,
        "u"_a,
        "reconstructed_u"_a,
        "over_integrate"_a = 2);
} // ... bind_neighborhood_reconstruction(...)


#else // HAVE_DUNE_FEM && HAVE_DUNE_ISTL && HAVE_DUNE_PDELAB


template <class G>
class DiffusiveFluxAaProduct
{
  static_assert(AlwaysFalse<G>::value, "You are missing dune-fem, dune-pdelab or dune-istl!");
};


template <class G>
class DiffusiveFluxAbProduct
{
  static_assert(AlwaysFalse<G>::value, "You are missing dune-fem, dune-pdelab or dune-istl!");
};


template <class G>
class DiffusiveFluxBbProduct
{
  static_assert(AlwaysFalse<G>::value, "You are missing dune-fem, dune-pdelab or dune-istl!");
};


template <class G>
class SwipdgPenaltySubdomainProduct
{
  static_assert(AlwaysFalse<G>::value, "You are missing dune-fem, dune-pdelab or dune-istl!");
};


template <class G>
void bind_neighborhood_discretization(pybind11::module& /*m*/)
{
  static_assert(AlwaysFalse<G>::value, "You are missing dune-fem, dune-pdelab or dune-istl!");
}


template <class G>
class HdivSemiProduct
{
  static_assert(AlwaysFalse<G>::value, "You are missing dune-fem, dune-pdelab or dune-istl!");
};


template <class G>
class ResidualPartFunctional
{
  static_assert(AlwaysFalse<G>::value, "You are missing dune-fem, dune-pdelab or dune-istl!");
};


template <class G>
void bind_neighborhood_reconstruction(pybind11::module& /*m*/)
{
  static_assert(AlwaysFalse<G>::value, "You are missing dune-fem, dune-pdelab or dune-istl!");
}


#endif // HAVE_DUNE_FEM && HAVE_DUNE_ISTL && HAVE_DUNE_PDELAB

} // namespace RS2017
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_PLAYGROUND_OPERATORS_RS2017_HH
