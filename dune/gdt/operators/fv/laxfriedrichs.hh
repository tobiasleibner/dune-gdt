// This file is part of the dune-gdt project:
//   https://github.com/dune-community/dune-gdt
// Copyright 2010-2017 dune-gdt developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   Felix Schindler (2016 - 2017)
//   Rene Milk       (2016 - 2017)
//   Tobias Leibner  (2016)

#ifndef DUNE_GDT_OPERATORS_FV_LAXFRIEDRICHS_HH
#define DUNE_GDT_OPERATORS_FV_LAXFRIEDRICHS_HH

#include <dune/gdt/local/fluxes/laxfriedrichs.hh>
#include <dune/gdt/operators/interfaces.hh>

#include "base.hh"

namespace Dune {
namespace GDT {

template <class AnalyticalFluxImp,
          class BoundaryValueImp,
          class LocalizableFunctionImp,
          size_t polOrder,
          SlopeLimiters slope_lim,
          class RealizabilityLimiterImp,
          class Traits>
class AdvectionLaxFriedrichsOperator;


namespace internal {


template <class AnalyticalFluxImp,
          class BoundaryValueImp,
          class LocalizableFunctionImp,
          size_t reconstruction_order,
          SlopeLimiters slope_lim,
          class RealizabilityLimiterImp>
class AdvectionLaxFriedrichsOperatorTraits : public AdvectionTraitsBase<AnalyticalFluxImp,
                                                                        BoundaryValueImp,
                                                                        reconstruction_order,
                                                                        slope_lim,
                                                                        RealizabilityLimiterImp>
{
  static_assert(XT::Functions::is_localizable_function<LocalizableFunctionImp>::value,
                "LocalizableFunctionImp has to be derived from XT::Functions::LocalizableFunctionInterface!");

  typedef AdvectionTraitsBase<AnalyticalFluxImp,
                              BoundaryValueImp,
                              reconstruction_order,
                              slope_lim,
                              RealizabilityLimiterImp>
      BaseType;

public:
  typedef LocalizableFunctionImp LocalizableFunctionType;
  using typename BaseType::AnalyticalFluxType;
  using typename BaseType::BoundaryValueType;
  typedef typename Dune::GDT::LaxFriedrichsLocalNumericalCouplingFlux<AnalyticalFluxType, LocalizableFunctionType>
      NumericalCouplingFluxType;
  typedef typename Dune::GDT::LaxFriedrichsLocalDirichletNumericalBoundaryFlux<AnalyticalFluxType,
                                                                               BoundaryValueType,
                                                                               LocalizableFunctionType>
      NumericalBoundaryFluxType;
  typedef AdvectionLaxFriedrichsOperator<AnalyticalFluxImp,
                                         BoundaryValueImp,
                                         LocalizableFunctionImp,
                                         reconstruction_order,
                                         slope_lim,
                                         RealizabilityLimiterImp,
                                         AdvectionLaxFriedrichsOperatorTraits>
      derived_type;
}; // class AdvectionLaxFriedrichsOperatorTraits


} // namespace internal

template <class AnalyticalFluxImp,
          class BoundaryValueImp,
          class LocalizableFunctionImp,
          size_t polOrder = 0,
          SlopeLimiters slope_lim = SlopeLimiters::minmod,
          class RealizabilityLimiterImp = NonLimitingRealizabilityLimiter<typename AnalyticalFluxImp::EntityType>,
          class Traits = internal::AdvectionLaxFriedrichsOperatorTraits<AnalyticalFluxImp,
                                                                        BoundaryValueImp,
                                                                        LocalizableFunctionImp,
                                                                        polOrder,
                                                                        slope_lim,
                                                                        RealizabilityLimiterImp>>
class AdvectionLaxFriedrichsOperator : public Dune::GDT::OperatorInterface<Traits>, public AdvectionOperatorBase<Traits>
{
  typedef AdvectionOperatorBase<Traits> BaseType;

public:
  using typename BaseType::AnalyticalFluxType;
  using typename BaseType::BoundaryValueType;
  using typename BaseType::DomainType;
  using typename BaseType::OnedQuadratureType;
  using typename BaseType::RangeFieldType;
  typedef typename Traits::LocalizableFunctionType LocalizableFunctionType;

  AdvectionLaxFriedrichsOperator(const AnalyticalFluxType& analytical_flux,
                                 const BoundaryValueType& boundary_values,
                                 const LocalizableFunctionType& dx,
                                 const bool use_local_laxfriedrichs_flux = false,
                                 const bool is_linear = false,
                                 const RangeFieldType alpha = BoundaryValueImp::dimDomain,
                                 const DomainType lambda = DomainType(0))
    : BaseType(analytical_flux, boundary_values, is_linear)
    , dx_(dx)
    , use_local_laxfriedrichs_flux_(use_local_laxfriedrichs_flux)
    , is_linear_(is_linear)
    , alpha_(alpha)
    , lambda_(lambda)
  {
  }

  AdvectionLaxFriedrichsOperator(const AnalyticalFluxType& analytical_flux,
                                 const BoundaryValueType& boundary_values,
                                 const LocalizableFunctionType& dx,
                                 const OnedQuadratureType& quadrature_1d,
                                 const std::shared_ptr<RealizabilityLimiterImp>& realizability_limiter = nullptr,
                                 const bool use_local_laxfriedrichs_flux = false,
                                 const bool is_linear = false,
                                 const RangeFieldType alpha = BoundaryValueImp::dimDomain,
                                 const DomainType lambda = DomainType(0))
    : BaseType(analytical_flux, boundary_values, is_linear, quadrature_1d, realizability_limiter)
    , dx_(dx)
    , use_local_laxfriedrichs_flux_(use_local_laxfriedrichs_flux)
    , is_linear_(is_linear)
    , alpha_(alpha)
    , lambda_(lambda)
  {
  }

  template <class SourceType, class RangeType>
  void apply(const SourceType& source, RangeType& range, const XT::Common::Parameter& param) const
  {
    BaseType::apply(source, range, param, dx_, use_local_laxfriedrichs_flux_, is_linear_, alpha_, lambda_);
  }

private:
  const LocalizableFunctionType& dx_;
  const bool use_local_laxfriedrichs_flux_;
  const bool is_linear_;
  const RangeFieldType alpha_;
  const DomainType lambda_;
}; // class AdvectionLaxFriedrichsOperator


} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_OPERATORS_FV_LAXFRIEDRICHS_HH
