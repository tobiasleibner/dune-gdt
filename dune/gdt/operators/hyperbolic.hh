// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_OPERATORS_HYPERBOLIC_HH
#define DUNE_GDT_OPERATORS_HYPERBOLIC_HH

#include <type_traits>

#include <dune/stuff/la/container/common.hh>
#include <dune/stuff/la/container/interfaces.hh>
#include <dune/stuff/functions/interfaces.hh>

#include <dune/gdt/spaces/interface.hh>
#include <dune/gdt/localevaluation/laxfriedrichs.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/discretefunction/default.hh>

#include "interfaces.hh"

namespace Dune {
namespace GDT {
namespace Operators {


// forward
template< class InnerAssemblerImp, class SourceImp, class RangeImp >
class HyperbolicLaxFriedrichsLocalizable;

template< class AnalyticalFluxImp, class LocalizableFunctionImp,class FVSpaceImp >
class HyperbolicLaxFriedrichs;


namespace internal {

template< class InnerAssemblerImp, class SourceImp, class RangeImp >
class HyperbolicLaxFriedrichsLocalizableTraits
{
public:
  typedef HyperbolicLaxFriedrichsLocalizable< InnerAssemblerImp, SourceImp, RangeImp >          derived_type;
  typedef SourceImp                                                                             SourceType;
  typedef RangeImp                                                                              RangeType;
  typedef InnerAssemblerImp                                                                     InnerAssemblerType;
  typedef typename RangeType::SpaceType::GridViewType                                           GridViewType;
  typedef typename GridViewType::ctype                                                          FieldType;
}; // class HyperbolicLaxFriedrichsTraits

template< class AnalyticalFluxImp, class LocalizableFunctionImp, class FVSpaceImp >
class HyperbolicLaxFriedrichsTraits
{
  static_assert(std::is_base_of< Stuff::GlobalFunctionInterface< typename AnalyticalFluxImp::EntityType,
                                                                 typename FVSpaceImp::RangeFieldType,
                                                                 FVSpaceImp::dimRange,
                                                                 typename FVSpaceImp::DomainFieldType,
                                                                 FVSpaceImp::dimDomain >,
                AnalyticalFluxImp >::value,
                "AnalyticalFluxType has to be derived from Stuff::GlobalFunctionInterface!");
  static_assert(Stuff::is_localizable_function< LocalizableFunctionImp >::value,
                "LocalizableFunctionImp has to be derived from Stuff::LocalizableFunctionInterface!");
  static_assert(is_space< FVSpaceImp >::value,    "FVSpaceImp has to be derived from SpaceInterface!");
public:
  typedef HyperbolicLaxFriedrichs< AnalyticalFluxImp, LocalizableFunctionImp, FVSpaceImp >      derived_type;
  typedef FVSpaceImp                                                                            FVSpaceType;
  typedef typename FVSpaceType::GridViewType                                                    GridViewType;
  typedef typename FVSpaceType::RangeFieldType                                                  RangeFieldType;
  typedef AnalyticalFluxImp                                                                     AnalyticalFluxType;
  typedef LocalizableFunctionImp                                                                LocalizableFunctionType;
  typedef typename FVSpaceType::DomainFieldType                                                 FieldType;
}; // class HyperbolicLaxFriedrichsTraits

} // namespace internal


template< class InnerAssemblerImp, class SourceImp, class RangeImp >
class HyperbolicLaxFriedrichsLocalizable
  : public Dune::GDT::LocalizableOperatorInterface<
                             internal::HyperbolicLaxFriedrichsLocalizableTraits< InnerAssemblerImp,
                                                                                 SourceImp,
                                                                                 RangeImp > >
  , public SystemAssembler< typename RangeImp::SpaceType >
{
  typedef Dune::GDT::LocalizableOperatorInterface<
                             internal::HyperbolicLaxFriedrichsLocalizableTraits< InnerAssemblerImp,
                                                                                 SourceImp,
                                                                                 RangeImp > > OperatorBaseType;
  typedef SystemAssembler< typename RangeImp::SpaceType >     AssemblerBaseType;

public:
  typedef internal::HyperbolicLaxFriedrichsLocalizableTraits< InnerAssemblerImp,
                                                              SourceImp,
                                                              RangeImp >         Traits;
  typedef typename Traits::GridViewType            GridViewType;
  typedef typename Traits::SourceType              SourceType;
  typedef typename Traits::RangeType               RangeType;
  typedef typename Traits::InnerAssemblerType      InnerAssemblerType;

  HyperbolicLaxFriedrichsLocalizable(const InnerAssemblerType inner_assembler,
                                     const SourceType& source,
                                     RangeType& range)
    : OperatorBaseType()
    , AssemblerBaseType(range.space())
    , inner_assembler_(inner_assembler)
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
    this->add(inner_assembler_, source_, range_);
    this->assemble();
  }

private:
  const InnerAssemblerType inner_assembler_;
  const SourceType& source_;
  RangeType& range_;
}; // class HyperbolicLaxFriedrichsLocalizable



template< class AnalyticalFluxImp, class LocalizableFunctionImp, class FVSpaceImp >
class HyperbolicLaxFriedrichs
  : public Dune::GDT::OperatorInterface< internal::HyperbolicLaxFriedrichsTraits< AnalyticalFluxImp,
                                                                                  LocalizableFunctionImp,
                                                                                  FVSpaceImp > >
{
  typedef Dune::GDT::OperatorInterface< internal::HyperbolicLaxFriedrichsTraits< AnalyticalFluxImp,
                                                                                 LocalizableFunctionImp,
                                                                                 FVSpaceImp > > OperatorBaseType;

  typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Inner< LocalizableFunctionImp >   NumericalFluxType;
  typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType >                      LocalOperatorType;
  typedef typename LocalAssembler::Codim1CouplingFV< LocalOperatorType >                        InnerAssemblerType;

public:
  typedef internal::HyperbolicLaxFriedrichsTraits< AnalyticalFluxImp, LocalizableFunctionImp, FVSpaceImp > Traits;
  typedef typename Traits::GridViewType            GridViewType;
  typedef typename Traits::RangeFieldType          RangeFieldType;
  typedef typename Traits::AnalyticalFluxType      AnalyticalFluxType;
  typedef typename Traits::LocalizableFunctionType LocalizableFunctionType;
  typedef typename Traits::FVSpaceType             FVSpaceType;

  HyperbolicLaxFriedrichs(const std::shared_ptr< const AnalyticalFluxType >& analytical_flux,
                          const LocalizableFunctionType& ratio_dt_dx,
                          const FVSpaceType& fv_space)
    : OperatorBaseType()
    , local_operator_(*analytical_flux, ratio_dt_dx)
    , inner_assembler_(local_operator_)
    , fv_space_(fv_space)
  {}

  const GridViewType& grid_view() const
  {
    return fv_space_.grid_view();
  }

  template< class SourceType, class RangeType >
  void apply(const SourceType& source, RangeType& range) const
  {
    HyperbolicLaxFriedrichsLocalizable< InnerAssemblerType, SourceType, RangeType > localizable_operator(inner_assembler_, source, range);
    localizable_operator.apply();
  }

private:
  const LocalOperatorType local_operator_;
  const InnerAssemblerType inner_assembler_;
  const FVSpaceType& fv_space_;
}; // class HyperbolicLaxFriedrichs


} // namespace Operators
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_OPERATORS_HYPERBOLIC_HH
