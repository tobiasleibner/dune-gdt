// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_OPERATORS_ADVECTION_HH
#define DUNE_GDT_OPERATORS_ADVECTION_HH

#include <type_traits>

#include <dune/stuff/aliases.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/grid/walker/apply-on.hh>
#include <dune/stuff/la/container/common.hh>
#include <dune/stuff/la/container/interfaces.hh>

#include <dune/gdt/spaces/interface.hh>
#include <dune/gdt/localevaluation/godunov.hh>
#include <dune/gdt/localevaluation/laxfriedrichs.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/discretefunction/default.hh>

#include "interfaces.hh"

namespace Dune {
namespace GDT {
namespace Operators {


// forwards
template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionLaxFriedrichsLocalizable;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionLaxFriedrichs;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class SourceImp, class BoundaryValueImp, class RangeImp >
class AdvectionGodunovLocalizable;

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionGodunov;


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
//  static_assert(is_space< FVSpaceImp >::value,    "FVSpaceImp has to be derived from SpaceInterface!");
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
    typedef AdvectionLaxFriedrichsLocalizableTraits< AnalyticalFluxImp, LocalizableFunctionImp, SourceImp, BoundaryValueImp, RangeImp > BaseType;
public:
  typedef AdvectionGodunovLocalizable< AnalyticalFluxImp,
                                             LocalizableFunctionImp,
                                             SourceImp,
                                             BoundaryValueImp,
                                             RangeImp >           derived_type;
  using typename BaseType::AnalyticalFluxType;
  using typename BaseType::LocalizableFunctionType;
  using typename BaseType::SourceType;
  using typename BaseType::BoundaryValueType;
  using typename BaseType::RangeType;
  using typename BaseType::GridViewType;
  using typename BaseType::FieldType;
}; // class AdvectionGodunovLocalizableTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class BoundaryValueImp, class FVSpaceImp >
class AdvectionGodunovTraits
    : public AdvectionLaxFriedrichsTraits< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp >
{
typedef AdvectionLaxFriedrichsTraits< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp > BaseType;
public:
  typedef AdvectionGodunov< AnalyticalFluxImp, LocalizableFunctionImp, BoundaryValueImp, FVSpaceImp > derived_type;
  using typename BaseType::AnalyticalFluxType;
  using typename BaseType::LocalizableFunctionType;
  using typename BaseType::BoundaryValueType;
  using typename BaseType::FVSpaceType;
  using typename BaseType::GridViewType;
  using typename BaseType::FieldType;
}; // class GodunovTraits

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
  typedef typename Traits::BoundaryValueType                                                  BoundaryValueType;

  typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Inner< LocalizableFunctionImp > NumericalFluxType;
  typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Dirichlet< LocalizableFunctionImp,
                                                                         BoundaryValueType >  NumericalBoundaryFluxType;
  typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType >                    LocalOperatorType;
  typedef typename Dune::GDT::LocalOperator::Codim1FVBoundary< NumericalBoundaryFluxType >    LocalBoundaryOperatorType;
  typedef typename LocalAssembler::Codim1CouplingFV< LocalOperatorType >                      InnerAssemblerType;
  typedef typename LocalAssembler::Codim1BoundaryFV< LocalBoundaryOperatorType >              BoundaryAssemblerType;

  AdvectionLaxFriedrichsLocalizable(const AnalyticalFluxType& analytical_flux,
                                    const LocalizableFunctionType& ratio_dt_dx,
                                    const SourceType& source,
                                    const BoundaryValueType& boundary_values,
                                    RangeType& range)
    : OperatorBaseType()
    , AssemblerBaseType(range.space())
    , local_operator_(analytical_flux, ratio_dt_dx)
    , local_boundary_operator_(analytical_flux, ratio_dt_dx, boundary_values)
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
                         const LocalizableFunctionType& ratio_dt_dx,
                         const BoundaryValueType& boundary_values,
                         const FVSpaceType& fv_space)
    : OperatorBaseType()
    , analytical_flux_(analytical_flux)
    , ratio_dt_dx_(ratio_dt_dx)
    , boundary_values_(boundary_values)
    , fv_space_(fv_space)
  {}

  const GridViewType& grid_view() const
  {
    return fv_space_.grid_view();
  }

  template< class SourceType, class RangeType >
  void apply(const SourceType& source, RangeType& range) const
  {
    AdvectionLaxFriedrichsLocalizable< AnalyticalFluxType,
                                       LocalizableFunctionType,
                                       SourceType,
                                       BoundaryValueType,
                                       RangeType > localizable_operator(analytical_flux_, ratio_dt_dx_, source, boundary_values_, range);
    localizable_operator.apply();
  }

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& ratio_dt_dx_;
  const BoundaryValueType&  boundary_values_;
  const FVSpaceType& fv_space_;
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
                              const LocalizableFunctionType& ratio_dt_dx,
                              const SourceType& source,
                              const BoundaryValueType& boundary_values,
                              RangeType& range,
                              const bool is_linear)
    : OperatorBaseType()
    , AssemblerBaseType(range.space())
    , local_operator_(analytical_flux, ratio_dt_dx, is_linear)
    , local_boundary_operator_(analytical_flux, ratio_dt_dx, boundary_values, is_linear)
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
                         const LocalizableFunctionType& ratio_dt_dx,
                         const BoundaryValueType& boundary_values,
                         const FVSpaceType& fv_space,
                         const bool is_linear = false)
    : OperatorBaseType()
    , analytical_flux_(analytical_flux)
    , ratio_dt_dx_(ratio_dt_dx)
    , boundary_values_(boundary_values)
    , fv_space_(fv_space)
    , is_linear_(is_linear)
  {}

  const GridViewType& grid_view() const
  {
    return fv_space_.grid_view();
  }

  template< class SourceType, class RangeType >
  void apply(const SourceType& source, RangeType& range, const double time = 0.0) const
  {
    typename BoundaryValueType::ExpressionFunctionType current_boundary_values = boundary_values_.evaluate_at_time(time);
    AdvectionGodunovLocalizable<       AnalyticalFluxType,
                                       LocalizableFunctionType,
                                       SourceType,
                                       typename BoundaryValueType::ExpressionFunctionType,
                                       RangeType > localizable_operator(analytical_flux_, ratio_dt_dx_, source, current_boundary_values, range, is_linear_);
    localizable_operator.apply();
  }

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& ratio_dt_dx_;
  const BoundaryValueType&  boundary_values_;
  const FVSpaceType& fv_space_;
  const bool is_linear_;
}; // class AdvectionGodunov

} // namespace Operators
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_OPERATORS_ADVECTION_HH
