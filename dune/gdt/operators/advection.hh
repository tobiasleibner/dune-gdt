// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_OPERATORS_ADVECTION_HH
#define DUNE_GDT_OPERATORS_ADVECTION_HH

#include <memory>
#include <type_traits>

#include <dune/stuff/aliases.hh>
#include <dune/stuff/common/memory.hh>
#include <dune/stuff/common/string.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/expression.hh>
#include <dune/stuff/grid/walker/apply-on.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container/common.hh>
#include <dune/stuff/la/container/interfaces.hh>

#include <dune/gdt/spaces/interface.hh>
#include <dune/gdt/localevaluation/godunov.hh>
#include <dune/gdt/localevaluation/laxfriedrichs.hh>
#include <dune/gdt/localevaluation/laxwendroff.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/discretefunction/default.hh>

#include <dune/gdt/playground/spaces/dg/pdelabproduct.hh>

#include "interfaces.hh"

namespace Dune {
namespace GDT {
namespace Operators {

enum class SlopeLimiters { minmod, mc, superbee, no_slope };

// forwards
template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionLaxFriedrichsLocalizable;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionLaxFriedrichs;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionGodunovLocalizable;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionGodunov;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionLaxWendroffLocalizable;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionLaxWendroff;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
class AdvectionGodunovWithReconstructionLocalizable;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp, SlopeLimiters slopeLimiter >
class AdvectionGodunovWithReconstruction;


namespace internal {


template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionLaxFriedrichsLocalizableTraits
{
  static_assert(std::is_base_of< Stuff::GlobalFunctionInterface< typename AnalyticalFluxImp::EntityType,
                                                                 typename SourceImp::RangeFieldType,
                                                                 SourceImp::dimRange,
                                                                 typename SourceImp::DomainFieldType,
                                                                 SourceImp::dimRange,
                                                                 SourceImp::dimDomain >,
                                 AnalyticalFluxImp >::value,
                "AnalyticalFluxImp has to be derived from Stuff::GlobalFunctionInterface!");
  static_assert(Stuff::is_localizable_function< LocalizableFunctionImp >::value,
                "LocalizableFunctionImp has to be derived from Stuff::LocalizableFunctionInterface!");
  static_assert(is_discrete_function< SourceImp >::value, "SourceImp has to be derived from DiscreteFunction!");
  static_assert(is_discrete_function< RangeImp >::value, "RangeImp has to be derived from DiscreteFunction!");
public:
  typedef AdvectionLaxFriedrichsLocalizable< AnalyticalFluxImp,
                                             LocalizableFunctionImp,
                                             SourceImp,
                                             BoundaryValueImp,
                                             RangeImp >                derived_type;
  typedef AnalyticalFluxImp                                            AnalyticalFluxType;
  typedef LocalizableFunctionImp                                       LocalizableFunctionType;
  typedef SourceImp                                                    SourceType;
  typedef BoundaryValueImp                                             BoundaryValueType;
  typedef RangeImp                                                     RangeType;
  typedef typename SourceType::RangeFieldType                          RangeFieldType;
  typedef typename RangeType::SpaceType::GridViewType                  GridViewType;
  typedef typename GridViewType::ctype                                 FieldType;
}; // class AdvectionLaxFriedrichsLocalizableTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionLaxFriedrichsTraits
{
  static_assert(std::is_base_of< Stuff::GlobalFunctionInterface< typename AnalyticalFluxImp::EntityType,
                                                                 typename FVSpaceImp::RangeFieldType,
                                                                 FVSpaceImp::dimRange,
                                                                 typename FVSpaceImp::DomainFieldType,
                                                                 FVSpaceImp::dimRange,
                                                                 FVSpaceImp::dimDomain >,
                                 AnalyticalFluxImp >::value,
                "AnalyticalFluxImp has to be derived from Stuff::GlobalFunctionInterface!");
  static_assert(Stuff::is_localizable_function< LocalizableFunctionImp >::value,
                "LocalizableFunctionImp has to be derived from Stuff::LocalizableFunctionInterface!");
  static_assert(is_space< FVSpaceImp >::value, "FVSpaceImp has to be derived from SpaceInterface!");
public:
  typedef AdvectionLaxFriedrichs< AnalyticalFluxImp, LocalizableFunctionImp,BoundaryValueImp, FVSpaceImp > derived_type;
  typedef AnalyticalFluxImp                                                                     AnalyticalFluxType;
  typedef LocalizableFunctionImp                                                                LocalizableFunctionType;
  typedef BoundaryValueImp                                                                      BoundaryValueType;
  typedef FVSpaceImp                                                                            FVSpaceType;
  typedef typename FVSpaceType::GridViewType                                                    GridViewType;
  typedef typename FVSpaceType::DomainFieldType                                                 FieldType;
}; // class LaxFriedrichsTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionGodunovLocalizableTraits
    : public AdvectionLaxFriedrichsLocalizableTraits< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp >
{
public:
  typedef AdvectionGodunovLocalizable< AnalyticalFluxImp,
                                             LocalizableFunctionImp,
                                             SourceImp,
                                             BoundaryValueImp,
                                             RangeImp >           derived_type;
}; // class AdvectionGodunovLocalizableTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionGodunovTraits
    : public AdvectionLaxFriedrichsTraits< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp >
{
public:
  typedef AdvectionGodunov< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp > derived_type;
}; // class AdvectionGodunovTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionLaxWendroffLocalizableTraits
    : public AdvectionLaxFriedrichsLocalizableTraits< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp >
{
public:
  typedef AdvectionLaxWendroffLocalizable< AnalyticalFluxImp,
                                           LocalizableFunctionImp,
                                           SourceImp,
                                           BoundaryValueImp,
                                           RangeImp >           derived_type;
}; // class AdvectionLaxWendroffLocalizableTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionLaxWendroffTraits
    : public AdvectionLaxFriedrichsTraits< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp >
{
public:
  typedef AdvectionLaxWendroff< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp > derived_type;
}; // class AdvectionLaxWendroffTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
class AdvectionGodunovWithReconstructionLocalizableTraits
    : public AdvectionLaxFriedrichsLocalizableTraits< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp >
{
public:
  typedef AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp,
                                                         LocalizableFunctionImp,
                                                         SourceImp,
                                                         BoundaryValueImp,
                                                         RangeImp,
                                                         slopeLimiter >           derived_type;
}; // class AdvectionGodunovWithReconstructionLocalizableTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp, SlopeLimiters slopeLimiter >
class AdvectionGodunovWithReconstructionTraits
    : public AdvectionLaxFriedrichsTraits< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp >
{
public:
  typedef AdvectionGodunovWithReconstruction< AnalyticalFluxImp,
                                              LocalizableFunctionImp,
                                              BoundaryValueImp,
                                              FVSpaceImp,
                                              slopeLimiter > derived_type;
}; // class AdvectionGodunovWithReconstructionTraits


} // namespace internal


template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionLaxFriedrichsLocalizable
  : public Dune::GDT::LocalizableOperatorInterface<
                             internal::AdvectionLaxFriedrichsLocalizableTraits< AnalyticalFluxImp,
                                                                                LocalizableFunctionImp,
                                                                                SourceImp,
                                                                                BoundaryValueImp,
                                                                                RangeImp > >
  , public SystemAssembler< typename RangeImp::SpaceType >
{
  typedef Dune::GDT::LocalizableOperatorInterface<
                             internal::AdvectionLaxFriedrichsLocalizableTraits< AnalyticalFluxImp,
                                                                                LocalizableFunctionImp,
                                                                                SourceImp,
                                                                                BoundaryValueImp,
                                                                                RangeImp > >    OperatorBaseType;
  typedef SystemAssembler< typename RangeImp::SpaceType >                                       AssemblerBaseType;
public:
  typedef internal::AdvectionLaxFriedrichsLocalizableTraits< AnalyticalFluxImp,
                                                             LocalizableFunctionImp,
                                                             SourceImp,
                                                             BoundaryValueImp,
                                                             RangeImp >                       Traits;

  typedef typename Traits::GridViewType                                                       GridViewType;
  typedef typename Traits::SourceType                                                         SourceType;
  typedef typename Traits::RangeType                                                          RangeType;
  typedef typename Traits::AnalyticalFluxType                                                 AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType                                            LocalizableFunctionType;
  typedef typename Traits::BoundaryValueType::ExpressionFunctionType                          BoundaryValueType;

  typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Inner< LocalizableFunctionImp > NumericalFluxType;
  typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Dirichlet< LocalizableFunctionImp,
                                                                         BoundaryValueType >  NumericalBoundaryFluxType;
  typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType >                    LocalOperatorType;
  typedef typename Dune::GDT::LocalOperator::Codim1FVBoundary< NumericalBoundaryFluxType >    LocalBoundaryOperatorType;
  typedef typename LocalAssembler::Codim1CouplingFV< LocalOperatorType >                      InnerAssemblerType;
  typedef typename LocalAssembler::Codim1BoundaryFV< LocalBoundaryOperatorType >              BoundaryAssemblerType;

  AdvectionLaxFriedrichsLocalizable(const AnalyticalFluxType& analytical_flux,
                                    const LocalizableFunctionType& dx,
                                    const SourceType& source,
                                    const BoundaryValueType& boundary_values,
                                    RangeType& range,
                                    const bool use_local)
    : OperatorBaseType()
    , AssemblerBaseType(range.space())
    , local_operator_(analytical_flux, dx, use_local)
    , local_boundary_operator_(analytical_flux, dx, boundary_values, use_local)
    , inner_assembler_(local_operator_)
    , boundary_assembler_(local_boundary_operator_)
    , source_(source)
    , range_(range)
  {}

  const GridViewType& grid_view() const
  {
    return range_.space().grid_view();
  }

  const SourceType& source() const
  {
    return source_;
  }

  const RangeType& range() const
  {
    return range_;
  }

  RangeType& range()
  {
    return range_;
  }

using AssemblerBaseType::add;
using AssemblerBaseType::assemble;

  void apply()
  {
    this->add(inner_assembler_, source_, range_, new DSG::ApplyOn::InnerIntersections< GridViewType >());
    this->add(inner_assembler_, source_, range_, new DSG::ApplyOn::PeriodicIntersections< GridViewType >());
    this->add(boundary_assembler_, source_, range_, new DSG::ApplyOn::NonPeriodicBoundaryIntersections< GridViewType >());
    this->assemble();
  }

private:
  const LocalOperatorType local_operator_;
  const LocalBoundaryOperatorType local_boundary_operator_;
  const InnerAssemblerType inner_assembler_;
  const BoundaryAssemblerType boundary_assembler_;
  const SourceType& source_;
  RangeType& range_;
}; // class AdvectionLaxFriedrichsLocalizable



template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionLaxFriedrichs
  : public Dune::GDT::OperatorInterface< internal::AdvectionLaxFriedrichsTraits<  AnalyticalFluxImp,
                                                                                  LocalizableFunctionImp,
                                                                                  BoundaryValueImp,
                                                                                  FVSpaceImp > >
{
  typedef Dune::GDT::OperatorInterface< internal::AdvectionLaxFriedrichsTraits<  AnalyticalFluxImp,
                                                                                 LocalizableFunctionImp,
                                                                                 BoundaryValueImp,
                                                                                 FVSpaceImp > > OperatorBaseType;

public:
  typedef internal::AdvectionLaxFriedrichsTraits< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp > Traits;
  typedef typename Traits::GridViewType            GridViewType;
  typedef typename Traits::AnalyticalFluxType      AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType LocalizableFunctionType;
  typedef typename Traits::BoundaryValueType       BoundaryValueType;
  typedef typename Traits::FVSpaceType             FVSpaceType;

  AdvectionLaxFriedrichs(const AnalyticalFluxType& analytical_flux,
                         const LocalizableFunctionType& dx,
                         const BoundaryValueType& boundary_values,
                         const FVSpaceType& fv_space,
                         const bool use_local = false)
    : OperatorBaseType()
    , analytical_flux_(analytical_flux)
    , dx_(dx)
    , boundary_values_(boundary_values)
    , fv_space_(fv_space)
    , use_local_(use_local)
  {}

  const GridViewType& grid_view() const
  {
    return fv_space_.grid_view();
  }

  template< class SourceType, class RangeType >
  void apply(const SourceType& source, RangeType& range, const double time = 0.0) const
  {
    typename BoundaryValueType::ExpressionFunctionType current_boundary_values = boundary_values_.evaluate_at_time(time);
    AdvectionLaxFriedrichsLocalizable< AnalyticalFluxType,
                                       LocalizableFunctionType,
                                       SourceType,
                                       BoundaryValueType,
                                       RangeType > localizable_operator(analytical_flux_, dx_, source, current_boundary_values, range, use_local_);
    localizable_operator.apply();
  }

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& dx_;
  const BoundaryValueType&  boundary_values_;
  const FVSpaceType& fv_space_;
  const bool use_local_;
}; // class AdvectionLaxFriedrichs

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionGodunovLocalizable
  : public Dune::GDT::LocalizableOperatorInterface<
                             internal::AdvectionGodunovLocalizableTraits< AnalyticalFluxImp,
                                                                          LocalizableFunctionImp,
                                                                          SourceImp,
                                                                          BoundaryValueImp,
                                                                          RangeImp > >
  , public SystemAssembler< typename RangeImp::SpaceType >
{
  typedef Dune::GDT::LocalizableOperatorInterface<
                             internal::AdvectionGodunovLocalizableTraits< AnalyticalFluxImp,
                                                                          LocalizableFunctionImp,
                                                                          SourceImp,
                                                                          BoundaryValueImp,
                                                                          RangeImp > >        OperatorBaseType;
  typedef SystemAssembler< typename RangeImp::SpaceType >                                     AssemblerBaseType;
public:
  typedef internal::AdvectionGodunovLocalizableTraits< AnalyticalFluxImp,
                                                       LocalizableFunctionImp,
                                                       SourceImp,
                                                       BoundaryValueImp,
                                                       RangeImp >                             Traits;

  typedef typename Traits::GridViewType                                                       GridViewType;
  typedef typename Traits::SourceType                                                         SourceType;
  typedef typename Traits::RangeType                                                          RangeType;
  typedef typename Traits::AnalyticalFluxType                                                 AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType                                            LocalizableFunctionType;
  typedef typename Traits::BoundaryValueType                                                  BoundaryValueType;

  typedef typename Dune::GDT::LocalEvaluation::Godunov::Inner< LocalizableFunctionImp >       NumericalFluxType;
  typedef typename Dune::GDT::LocalEvaluation::Godunov::Dirichlet< LocalizableFunctionImp,
                                                                   BoundaryValueType >        NumericalBoundaryFluxType;
  typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType >                    LocalOperatorType;
  typedef typename Dune::GDT::LocalOperator::Codim1FVBoundary< NumericalBoundaryFluxType >    LocalBoundaryOperatorType;
  typedef typename LocalAssembler::Codim1CouplingFV< LocalOperatorType >                      InnerAssemblerType;
  typedef typename LocalAssembler::Codim1BoundaryFV< LocalBoundaryOperatorType >              BoundaryAssemblerType;

  AdvectionGodunovLocalizable(const AnalyticalFluxType& analytical_flux,
                              const LocalizableFunctionType& dx,
                              const double dt,
                              const SourceType& source,
                              const BoundaryValueType& boundary_values,
                              RangeType& range,
                              const bool is_linear)
    : OperatorBaseType()
    , AssemblerBaseType(range.space())
    , local_operator_(analytical_flux, dx, dt, is_linear)
    , local_boundary_operator_(analytical_flux, dx, dt, boundary_values, is_linear)
    , inner_assembler_(local_operator_)
    , boundary_assembler_(local_boundary_operator_)
    , source_(source)
    , range_(range)
  {}

  const GridViewType& grid_view() const
  {
    return range_.space().grid_view();
  }

  const SourceType& source() const
  {
    return source_;
  }

  const RangeType& range() const
  {
    return range_;
  }

  RangeType& range()
  {
    return range_;
  }

using AssemblerBaseType::add;
using AssemblerBaseType::assemble;

  void apply()
  {
    this->add(inner_assembler_, source_, range_, new DSG::ApplyOn::InnerIntersections< GridViewType >());
    this->add(inner_assembler_, source_, range_, new DSG::ApplyOn::PeriodicIntersections< GridViewType >());
    this->add(boundary_assembler_, source_, range_, new DSG::ApplyOn::NonPeriodicBoundaryIntersections< GridViewType >());
    this->assemble();
  }

private:
  const LocalOperatorType local_operator_;
  const LocalBoundaryOperatorType local_boundary_operator_;
  const InnerAssemblerType inner_assembler_;
  const BoundaryAssemblerType boundary_assembler_;
  const SourceType& source_;
  RangeType& range_;
}; // class AdvectionGodunovLocalizable



template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionGodunov
  : public Dune::GDT::OperatorInterface< internal::AdvectionGodunovTraits<  AnalyticalFluxImp,
                                                                            LocalizableFunctionImp,
                                                                            BoundaryValueImp,
                                                                            FVSpaceImp > >
{
  typedef Dune::GDT::OperatorInterface< internal::AdvectionGodunovTraits<  AnalyticalFluxImp,
                                                                           LocalizableFunctionImp,
                                                                           BoundaryValueImp,
                                                                           FVSpaceImp > > OperatorBaseType;

public:
  typedef internal::AdvectionGodunovTraits< AnalyticalFluxImp,
                                            LocalizableFunctionImp,
                                            BoundaryValueImp,
                                            FVSpaceImp >          Traits;
  typedef typename Traits::GridViewType                           GridViewType;
  typedef typename Traits::AnalyticalFluxType                     AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType                LocalizableFunctionType;
  typedef typename Traits::BoundaryValueType                      BoundaryValueType;
  typedef typename Traits::FVSpaceType                            FVSpaceType;

  AdvectionGodunov(const AnalyticalFluxType& analytical_flux,
                         const LocalizableFunctionType& dx,
                         const double dt,
                         const BoundaryValueType& boundary_values,
                         const FVSpaceType& fv_space,
                         const bool is_linear = false)
    : OperatorBaseType()
    , analytical_flux_(analytical_flux)
    , dx_(dx)
    , dt_(dt)
    , boundary_values_(boundary_values)
    , fv_space_(fv_space)
    , is_linear_(is_linear)
  {}

  const GridViewType& grid_view() const
  {
    return fv_space_.grid_view();
  }

  template< class SourceType, class RangeType >
  void apply(const SourceType& source, RangeType& range, const double time = 0.0, const bool = false, const double = 0) const
  {
    typename BoundaryValueType::ExpressionFunctionType current_boundary_values = boundary_values_.evaluate_at_time(time);
    AdvectionGodunovLocalizable<       AnalyticalFluxType,
                                       LocalizableFunctionType,
                                       SourceType,
                                       typename BoundaryValueType::ExpressionFunctionType,
                                       RangeType > localizable_operator(analytical_flux_, dx_, dt_, source, current_boundary_values, range, is_linear_);
    localizable_operator.apply();
  }

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& dx_;
  const double dt_;
  const BoundaryValueType&  boundary_values_;
  const FVSpaceType& fv_space_;
  const bool is_linear_;
}; // class AdvectionGodunov

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionLaxWendroffLocalizable
  : public Dune::GDT::LocalizableOperatorInterface<
                             internal::AdvectionLaxWendroffLocalizableTraits< AnalyticalFluxImp,
                                                                          LocalizableFunctionImp,
                                                                          SourceImp,
                                                                          BoundaryValueImp,
                                                                          RangeImp > >
  , public SystemAssembler< typename RangeImp::SpaceType >
{
  typedef Dune::GDT::LocalizableOperatorInterface<
                             internal::AdvectionLaxWendroffLocalizableTraits< AnalyticalFluxImp,
                                                                          LocalizableFunctionImp,
                                                                          SourceImp,
                                                                          BoundaryValueImp,
                                                                          RangeImp > >        OperatorBaseType;
  typedef SystemAssembler< typename RangeImp::SpaceType >                                     AssemblerBaseType;
public:
  typedef internal::AdvectionLaxWendroffLocalizableTraits< AnalyticalFluxImp,
                                                       LocalizableFunctionImp,
                                                       SourceImp,
                                                       BoundaryValueImp,
                                                       RangeImp >                             Traits;

  typedef typename Traits::GridViewType                                                       GridViewType;
  typedef typename Traits::SourceType                                                         SourceType;
  typedef typename Traits::RangeType                                                          RangeType;
  typedef typename Traits::AnalyticalFluxType                                                 AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType                                            LocalizableFunctionType;
  typedef typename Traits::BoundaryValueType                                                  BoundaryValueType;

  typedef typename Dune::GDT::LocalEvaluation::LaxWendroff::Inner< LocalizableFunctionImp >       NumericalFluxType;
  typedef typename Dune::GDT::LocalEvaluation::LaxWendroff::Dirichlet< LocalizableFunctionImp,
                                                                   BoundaryValueType >        NumericalBoundaryFluxType;
  typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType >                    LocalOperatorType;
  typedef typename Dune::GDT::LocalOperator::Codim1FVBoundary< NumericalBoundaryFluxType >    LocalBoundaryOperatorType;
  typedef typename LocalAssembler::Codim1CouplingFV< LocalOperatorType >                      InnerAssemblerType;
  typedef typename LocalAssembler::Codim1BoundaryFV< LocalBoundaryOperatorType >              BoundaryAssemblerType;

  AdvectionLaxWendroffLocalizable(const AnalyticalFluxType& analytical_flux,
                                  const LocalizableFunctionType& dx,
                                  const double dt,
                                  const SourceType& source,
                                  const BoundaryValueType& boundary_values,
                                  RangeType& range,
                                  const bool is_linear)
    : OperatorBaseType()
    , AssemblerBaseType(range.space())
    , local_operator_(analytical_flux, dx, dt, is_linear)
    , local_boundary_operator_(analytical_flux, dx, dt, boundary_values, is_linear)
    , inner_assembler_(local_operator_)
    , boundary_assembler_(local_boundary_operator_)
    , source_(source)
    , range_(range)
  {}

  const GridViewType& grid_view() const
  {
    return range_.space().grid_view();
  }

  const SourceType& source() const
  {
    return source_;
  }

  const RangeType& range() const
  {
    return range_;
  }

  RangeType& range()
  {
    return range_;
  }

using AssemblerBaseType::add;
using AssemblerBaseType::assemble;

  void apply()
  {
    this->add(inner_assembler_, source_, range_, new DSG::ApplyOn::InnerIntersections< GridViewType >());
    this->add(inner_assembler_, source_, range_, new DSG::ApplyOn::PeriodicIntersections< GridViewType >());
    this->add(boundary_assembler_, source_, range_, new DSG::ApplyOn::NonPeriodicBoundaryIntersections< GridViewType >());
    this->assemble();
  }

private:
  const LocalOperatorType local_operator_;
  const LocalBoundaryOperatorType local_boundary_operator_;
  const InnerAssemblerType inner_assembler_;
  const BoundaryAssemblerType boundary_assembler_;
  const SourceType& source_;
  RangeType& range_;
}; // class AdvectionLaxWendroffLocalizable



template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionLaxWendroff
  : public Dune::GDT::OperatorInterface< internal::AdvectionLaxWendroffTraits<  AnalyticalFluxImp,
                                                                            LocalizableFunctionImp,
                                                                            BoundaryValueImp,
                                                                            FVSpaceImp > >
{
  typedef Dune::GDT::OperatorInterface< internal::AdvectionLaxWendroffTraits<  AnalyticalFluxImp,
                                                                           LocalizableFunctionImp,
                                                                           BoundaryValueImp,
                                                                           FVSpaceImp > > OperatorBaseType;

public:
  typedef internal::AdvectionLaxWendroffTraits< AnalyticalFluxImp,
                                            LocalizableFunctionImp,
                                            BoundaryValueImp,
                                            FVSpaceImp >          Traits;
  typedef typename Traits::GridViewType                           GridViewType;
  typedef typename Traits::AnalyticalFluxType                     AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType                LocalizableFunctionType;
  typedef typename Traits::BoundaryValueType                      BoundaryValueType;
  typedef typename Traits::FVSpaceType                            FVSpaceType;

  AdvectionLaxWendroff(const AnalyticalFluxType& analytical_flux,
                       const LocalizableFunctionType& dx,
                       const double dt,
                       const BoundaryValueType& boundary_values,
                       const FVSpaceType& fv_space,
                       const bool is_linear = false)
    : OperatorBaseType()
    , analytical_flux_(analytical_flux)
    , dx_(dx)
    , dt_(dt)
    , boundary_values_(boundary_values)
    , fv_space_(fv_space)
    , is_linear_(is_linear)
  {}

  const GridViewType& grid_view() const
  {
    return fv_space_.grid_view();
  }

  template< class SourceType, class RangeType >
  void apply(const SourceType& source, RangeType& range, const double time = 0.0, const bool = false, const double = 0) const
  {
    typename BoundaryValueType::ExpressionFunctionType current_boundary_values = boundary_values_.evaluate_at_time(time);
    AdvectionLaxWendroffLocalizable<   AnalyticalFluxType,
                                       LocalizableFunctionType,
                                       SourceType,
                                       typename BoundaryValueType::ExpressionFunctionType,
                                       RangeType > localizable_operator(analytical_flux_, dx_, dt_, source, current_boundary_values, range, is_linear_);
    localizable_operator.apply();
  }

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& dx_;
  const double dt_;
  const BoundaryValueType&  boundary_values_;
  const FVSpaceType& fv_space_;
  const bool is_linear_;
}; // class AdvectionLaxWendroff


namespace internal {


template< SlopeLimiters slopeLimiter, class VectorType >
struct ChooseLimiter
{
  static VectorType limit(const VectorType slope_left,
                          const VectorType slope_right,
                          const VectorType centered_slope);
};

template< class VectorType >
struct ChooseLimiter< SlopeLimiters::minmod, VectorType >
{
  static VectorType limit(const VectorType slope_left,
                          const VectorType slope_right,
                          const VectorType /*centered_slope*/ = VectorType(0))
  {
    VectorType ret;
    for (size_t ii = 0; ii < slope_left.size(); ++ii) {
      const auto slope_left_abs = std::abs(slope_left[ii]);
      const auto slope_right_abs = std::abs(slope_right[ii]);
      if (slope_left_abs < slope_right_abs && slope_left[ii]*slope_right[ii] > 0)
        ret[ii] = slope_left[ii];
      else if (DSC::FloatCmp::ge(slope_left_abs, slope_right_abs) && slope_left[ii]*slope_right[ii] > 0)
        ret[ii] = slope_right[ii];
      else
        ret[ii] = 0.0;
    }
    return ret;
  }
};

template< class VectorType >
struct ChooseLimiter< SlopeLimiters::superbee, VectorType >
{
  static VectorType limit(const VectorType slope_left,
                          const VectorType slope_right,
                          const VectorType /*centered_slope*/)
  {
    typedef ChooseLimiter< SlopeLimiters::minmod, VectorType > MinmodType;
    return maxmod(MinmodType::limit(slope_left, slope_right*2.0), MinmodType::limit(slope_left*2.0, slope_right));
  }

  static VectorType maxmod(const VectorType slope_left,
                           const VectorType slope_right)
  {
    VectorType ret;
    for (size_t ii = 0; ii < slope_left.size(); ++ii) {
      const auto slope_left_abs = std::abs(slope_left[ii]);
      const auto slope_right_abs = std::abs(slope_right[ii]);
      if (slope_left_abs > slope_right_abs && slope_left[ii]*slope_right[ii] > 0)
        ret[ii] = slope_left[ii];
      else if (DSC::FloatCmp::le(slope_left_abs, slope_right_abs) && slope_left[ii]*slope_right[ii] > 0)
        ret[ii] = slope_right[ii];
      else
        ret[ii] = 0.0;
    }
    return ret;
  }
};

template< class VectorType >
struct ChooseLimiter< SlopeLimiters::mc, VectorType >
{
  static VectorType limit(const VectorType slope_left,
                          const VectorType slope_right,
                          const VectorType centered_slope)
  {
    typedef ChooseLimiter< SlopeLimiters::minmod, VectorType > MinmodType;
    return MinmodType::limit(MinmodType::limit(slope_left*2.0, slope_right*2.0), centered_slope);
  }
};

template< class VectorType >
struct ChooseLimiter< SlopeLimiters::no_slope, VectorType >
{
  static VectorType limit(const VectorType /*slope_left*/,
                          const VectorType /*slope_right*/,
                          const VectorType /*centered_slope*/)
  {
    return VectorType(0);
  }
};


} // namespace internal


template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
class AdvectionGodunovWithReconstructionLocalizable
  : public Dune::GDT::LocalizableOperatorInterface<
                             internal::AdvectionGodunovWithReconstructionLocalizableTraits< AnalyticalFluxImp,
                                                                                            LocalizableFunctionImp,
                                                                                            SourceImp,
                                                                                            BoundaryValueImp,
                                                                                            RangeImp,
                                                                                            slopeLimiter > >
  , public SystemAssembler< typename RangeImp::SpaceType >
{
  typedef Dune::GDT::LocalizableOperatorInterface<
                             internal::AdvectionGodunovWithReconstructionLocalizableTraits< AnalyticalFluxImp,
                                                                                            LocalizableFunctionImp,
                                                                                            SourceImp,
                                                                                            BoundaryValueImp,
                                                                                            RangeImp,
                                                                                            slopeLimiter > > OperatorBaseType;
  typedef SystemAssembler< typename RangeImp::SpaceType >                                     AssemblerBaseType;
public:
  typedef internal::AdvectionGodunovWithReconstructionLocalizableTraits< AnalyticalFluxImp,
                                                       LocalizableFunctionImp,
                                                       SourceImp,
                                                       BoundaryValueImp,
                                                       RangeImp,
                                                       slopeLimiter >                            Traits;

  typedef typename Traits::GridViewType                                                       GridViewType;
  static_assert(GridViewType::dimension == 1, "Not implemented for dimDomain > 1!");
  typedef typename Traits::SourceType                                                         SourceType;
  typedef typename Traits::RangeType                                                          RangeType;
  typedef typename Traits::RangeFieldType                                                     RangeFieldType;
  typedef typename Traits::AnalyticalFluxType                                                 AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType                                            LocalizableFunctionType;
  typedef typename Traits::BoundaryValueType                                                  BoundaryValueType;

  typedef typename Dune::GDT::LocalEvaluation::Godunov::Inner< LocalizableFunctionImp >       NumericalFluxType;
  typedef typename Dune::GDT::LocalEvaluation::Godunov::Dirichlet< LocalizableFunctionImp,
                                                                   BoundaryValueType >        NumericalBoundaryFluxType;
  typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType >                    LocalOperatorType;
  typedef typename Dune::GDT::LocalOperator::Codim1FVBoundary< NumericalBoundaryFluxType >    LocalBoundaryOperatorType;
  typedef typename LocalAssembler::Codim1CouplingFV< LocalOperatorType >                      InnerAssemblerType;
  typedef typename LocalAssembler::Codim1BoundaryFV< LocalBoundaryOperatorType >              BoundaryAssemblerType;

  typedef typename Dune::Stuff::LA::EigenDenseMatrix< RangeFieldType >                        EigenMatrixType;

  typedef typename GridViewType::Grid                                                         GridType;
  typedef typename GridType::template Codim< 0 >::Entity                                      EntityType;
  typedef typename GridType::template Codim< 0 >::EntityPointer                               EntityPointerType;
  static const size_t dimRange = SourceType::dimRange;
  static const size_t dimRangeCols = SourceType::dimRangeCols;
  typedef typename DSC::FieldMatrix< RangeFieldType, dimRange, dimRange >                     StuffFieldMatrixType;
  typedef typename DSC::FieldVector< RangeFieldType, dimRange >                               StuffFieldVectorType;

  typedef Spaces::DG::PdelabBasedProduct< GridViewType,
                                          1, //polOrder
                                          RangeFieldType,
                                          dimRange,
                                          dimRangeCols >               DGSpaceType;
  typedef DiscreteFunction< DGSpaceType, typename SourceType::VectorType >  ReconstructedDiscreteFunctionType;

  AdvectionGodunovWithReconstructionLocalizable(const AnalyticalFluxType& analytical_flux,
                                                const LocalizableFunctionType& dx,
                                                const double dt,
                                                const SourceType& source,
                                                const BoundaryValueType& boundary_values,
                                                RangeType& range,
                                                const bool is_linear)
    : OperatorBaseType()
    , AssemblerBaseType(range.space())
    , analytical_flux_(analytical_flux)
    , local_operator_(analytical_flux, dx, dt, is_linear)
    , boundary_values_(boundary_values)
    , local_boundary_operator_(analytical_flux, dx, dt, boundary_values_, is_linear)
    , inner_assembler_(local_operator_)
    , boundary_assembler_(local_boundary_operator_)
    , source_(source)
    , range_(range)
    , grid_view_(source.space().grid_view())
  {
    if (first_run_) {
      dg_space_ = DSC::make_unique< DGSpaceType >(grid_view_);
      reconstruction_ = DSC::make_unique< ReconstructedDiscreteFunctionType >(*dg_space_, "reconstructed");
      first_run_ = false;
    }
    reconstruct_linear(is_linear);
  }

  const GridViewType& grid_view() const
  {
    return range_.space().grid_view();
  }

  const SourceType& source() const
  {
    return source_;
  }

  const RangeType& range() const
  {
    return range_;
  }

  RangeType& range()
  {
    return range_;
  }

  using AssemblerBaseType::add;
  using AssemblerBaseType::assemble;

  void apply()
  {
    this->add(inner_assembler_, *reconstruction_, range_, new DSG::ApplyOn::InnerIntersections< GridViewType >());
    this->add(inner_assembler_, *reconstruction_, range_, new DSG::ApplyOn::PeriodicIntersections< GridViewType >());
    this->add(boundary_assembler_, *reconstruction_, range_, new DSG::ApplyOn::NonPeriodicBoundaryIntersections< GridViewType >());
    this->assemble();
  }

  void visualize_reconstruction(const size_t save_counter) {
    reconstruction_->template visualize_factor< 0 >("reconstruction_" + DSC::toString(save_counter));
  }

private:
  void reconstruct_linear(const bool is_linear) {
    const auto it_end = grid_view_.template end< 0 >();
    auto it = grid_view_.template begin< 0 >();
    // create EntityPointers for current entity and two neighbors to the right and left
    EntityPointerType left_neighbor_ptr(*it);
    EntityPointerType right_neighbor_ptr(*it);
    // create vectors to store boundary values on left and right boundary
    typename SourceType::RangeType right_boundary_value;
    typename SourceType::RangeType left_boundary_value;
    // get reconstruction vector and mapper
    auto& reconstruction_vector = reconstruction_->vector();
    const auto& reconstruction_mapper = reconstruction_->space().factor_mapper();
    // walk the grid to reconstruct
    for (; it != it_end; ++it) {
      const auto& entity = *it;
      bool on_left_boundary(false);
      bool on_right_boundary(false);
      const auto entity_center = entity.geometry().center();
      // walk over intersections to get neighbors
      const auto i_it_end = grid_view_.iend(entity);
      for (auto i_it = grid_view_.ibegin(entity); i_it != i_it_end; ++i_it) {
        const auto& intersection = *i_it;
        const auto& neighbor = intersection.neighbor() ? *(intersection.outside()) : entity;
        if (intersection.neighbor()) {
          if ((neighbor.geometry().center()[0] < entity_center[0] && !(intersection.boundary())) || (neighbor.geometry().center()[0] > entity_center[0] && intersection.boundary()))
            left_neighbor_ptr = EntityPointerType(neighbor);
          else
            right_neighbor_ptr = EntityPointerType(neighbor);
        } else {
          if (intersection.geometry().center()[0] < entity_center[0]) {
            on_left_boundary = true;
            left_boundary_value = boundary_values_.local_function(entity)->evaluate(intersection.geometryInInside().center());
          } else {
            on_right_boundary = true;
            right_boundary_value = boundary_values_.local_function(entity)->evaluate(intersection.geometryInInside().center());
          }
        }
      }
      // get values of discrete function
      const auto u_left = on_left_boundary
                          ? left_boundary_value
                          : source_.local_discrete_function(*left_neighbor_ptr)->evaluate(left_neighbor_ptr->geometry().local(left_neighbor_ptr->geometry().center()));
      const auto u_right = on_right_boundary
                           ? right_boundary_value
                           : source_.local_discrete_function(*right_neighbor_ptr)->evaluate(right_neighbor_ptr->geometry().local(right_neighbor_ptr->geometry().center()));
      const auto u_entity = source_.local_discrete_function(entity)->evaluate(entity.geometry().local(entity_center));

      // diagonalize the system of equations from u_t + A*u_x = 0 to w_t + D*w_x = 0 where D = R^(-1)*A*R, w = R^(-1)*u and R matrix of eigenvectors of A
      if (!eigenvectors_calculated_) {
#if HAVE_EIGEN
        // create EigenSolver
        ::Eigen::EigenSolver< typename EigenMatrixType::BackendType > eigen_solver(DSC::fromString< EigenMatrixType >(DSC::toString(analytical_flux_.jacobian(u_entity))).backend());
        assert(eigen_solver.info() == ::Eigen::Success);
        const auto eigen_eigenvectors = eigen_solver.eigenvectors();
#  ifndef NDEBUG
        for (size_t ii = 0; ii < dimRange; ++ii)
          for (size_t jj = 0; jj < dimRange; ++jj)
            assert(eigen_eigenvectors(ii,jj).imag() < 1e-15);
#  endif
        const EigenMatrixType eigenvectors(eigen_eigenvectors.real()); // <- this should be an Eigen matrix of std::complex
        const EigenMatrixType eigenvectors_inverse(eigen_eigenvectors.inverse().real());
        eigenvectors_ = DSC::fromString< StuffFieldMatrixType >(DSC::toString(eigenvectors));
        eigenvectors_inverse_ = DSC::fromString< StuffFieldMatrixType >(DSC::toString(eigenvectors_inverse));
        if (is_linear)
          eigenvectors_calculated_ = true;
#else
        static_assert(AlwaysFalse< bool >::value, "You are missing eigen!");
#endif
      }
      const StuffFieldVectorType w_left(eigenvectors_inverse_*u_left);
      const StuffFieldVectorType w_right(eigenvectors_inverse_*u_right);
      const StuffFieldVectorType w_entity(eigenvectors_inverse_*u_entity);

      const StuffFieldVectorType w_slope_left = w_entity - w_left;
      const StuffFieldVectorType w_slope_right = w_right - w_entity;
      const StuffFieldVectorType w_centered_slope = w_right*RangeFieldType(0.5) - w_left*RangeFieldType(0.5);
      const StuffFieldVectorType w_slope = internal::ChooseLimiter< slopeLimiter, StuffFieldVectorType >::limit(w_slope_left, w_slope_right, w_centered_slope);

      const StuffFieldVectorType w_value_left = w_entity - w_slope*RangeFieldType(0.5);
      const StuffFieldVectorType w_value_right = w_entity + w_slope*RangeFieldType(0.5);

      const StuffFieldVectorType reconstructed_value_left(eigenvectors_*w_value_left);
      const StuffFieldVectorType reconstructed_value_right(eigenvectors_*w_value_right);

      for (size_t factor_index = 0; factor_index < dimRange; ++factor_index) {
        // set values on dofs, dof with local index 0 for each factor space corresponds to 1 - x, local index 1 to x
        reconstruction_vector.set_entry(reconstruction_mapper.mapToGlobal(factor_index, entity, 0),
                                        reconstructed_value_left[factor_index]);
        reconstruction_vector.set_entry(reconstruction_mapper.mapToGlobal(factor_index, entity, 1),
                                        reconstructed_value_right[factor_index]);
      }
    }
  }

  const AnalyticalFluxType& analytical_flux_;
  const LocalOperatorType local_operator_;
  const LocalBoundaryOperatorType local_boundary_operator_;
  const InnerAssemblerType inner_assembler_;
  const BoundaryValueType& boundary_values_;
  const BoundaryAssemblerType boundary_assembler_;
  const SourceType& source_;
  RangeType& range_;
  const GridViewType& grid_view_;
  static bool first_run_;
  static std::unique_ptr< DGSpaceType > dg_space_;
  static std::unique_ptr< ReconstructedDiscreteFunctionType > reconstruction_;
  static StuffFieldMatrixType eigenvectors_;
  static StuffFieldMatrixType eigenvectors_inverse_;
  static bool eigenvectors_calculated_;
}; // class AdvectionGodunovWithReconstructionLocalizable

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
bool
AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::first_run_(true);

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
std::unique_ptr< typename AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::DGSpaceType >
AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::dg_space_;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
std::unique_ptr< typename AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::ReconstructedDiscreteFunctionType >
AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::reconstruction_;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
typename AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::StuffFieldMatrixType
AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::eigenvectors_(0);

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
typename AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::StuffFieldMatrixType
AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::eigenvectors_inverse_(0);

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp, SlopeLimiters slopeLimiter >
bool
AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp, slopeLimiter >::eigenvectors_calculated_(false);


template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp, SlopeLimiters slopeLimiter >
class AdvectionGodunovWithReconstruction
  : public Dune::GDT::OperatorInterface< internal::AdvectionGodunovWithReconstructionTraits<  AnalyticalFluxImp,
                                                                            LocalizableFunctionImp,
                                                                            BoundaryValueImp,
                                                                            FVSpaceImp, slopeLimiter > >
{
  typedef Dune::GDT::OperatorInterface< internal::AdvectionGodunovWithReconstructionTraits<  AnalyticalFluxImp,
                                                                           LocalizableFunctionImp,
                                                                           BoundaryValueImp,
                                                                           FVSpaceImp, slopeLimiter > > OperatorBaseType;

public:
  typedef internal::AdvectionGodunovWithReconstructionTraits< AnalyticalFluxImp,
                                            LocalizableFunctionImp,
                                            BoundaryValueImp,
                                            FVSpaceImp, slopeLimiter >          Traits;
  typedef typename Traits::GridViewType                           GridViewType;
  typedef typename Traits::AnalyticalFluxType                     AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType                LocalizableFunctionType;
  typedef typename Traits::BoundaryValueType                      BoundaryValueType;
  typedef typename Traits::FVSpaceType                            FVSpaceType;

  AdvectionGodunovWithReconstruction(const AnalyticalFluxType& analytical_flux,
                         const LocalizableFunctionType& dx,
                         const double dt,
                         const BoundaryValueType& boundary_values,
                         const FVSpaceType& fv_space,
                         const bool is_linear = false)
    : OperatorBaseType()
    , analytical_flux_(analytical_flux)
    , dx_(dx)
    , dt_(dt)
    , boundary_values_(boundary_values)
    , current_boundary_values_(boundary_values_.evaluate_at_time(0.0))
    , fv_space_(fv_space)
    , is_linear_(is_linear)
  {}

  const GridViewType& grid_view() const
  {
    return fv_space_.grid_view();
  }

  template< class SourceType, class RangeType >
  void apply(const SourceType& source, RangeType& range, const double time = 0.0, const bool visualize = false, const size_t save_counter = 0) const
  {
    current_boundary_values_ = boundary_values_.evaluate_at_time(time);
    AdvectionGodunovWithReconstructionLocalizable< AnalyticalFluxType,
                                       LocalizableFunctionType,
                                       SourceType,
                                       typename BoundaryValueType::ExpressionFunctionType,
                                       RangeType,
                                       slopeLimiter > localizable_operator(analytical_flux_, dx_, dt_, source, current_boundary_values_, range, is_linear_);
    // hack to be able to visualize reconstruction in timestepper
    if (!visualize)
      localizable_operator.apply();
    else
      localizable_operator.visualize_reconstruction(save_counter);
  }

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& dx_;
  const double dt_;
  const BoundaryValueType& boundary_values_;
  mutable typename BoundaryValueType::ExpressionFunctionType current_boundary_values_;
  const FVSpaceType& fv_space_;
  const bool is_linear_;
}; // class AdvectionGodunovWithReconstruction


} // namespace Operators
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_OPERATORS_ADVECTION_HH