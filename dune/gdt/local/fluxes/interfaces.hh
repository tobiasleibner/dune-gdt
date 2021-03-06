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

#ifndef DUNE_GDT_LOCAL_FLUXES_INTERFACES_HH
#define DUNE_GDT_LOCAL_FLUXES_INTERFACES_HH

#include <dune/grid/yaspgrid.hh>

#include <dune/xt/common/crtp.hh>
#include <dune/xt/functions/interfaces.hh>

namespace Dune {
namespace GDT {
namespace internal {


class IsNumericalCouplingFlux
{
};

class IsNumericalBoundaryFlux
{
};

class IsAnalyticalFlux
{
};

class IsRHSEvaluation
{
};


} // namespace internal


template <class Traits>
class LocalNumericalCouplingFluxInterface
    : public XT::CRTPInterface<LocalNumericalCouplingFluxInterface<Traits>, Traits>,
      internal::IsNumericalCouplingFlux
{
  typedef typename Traits::LocalfunctionTupleType LocalfunctionTupleType;
  typedef typename Traits::EntityType EntityType;

public:
  LocalfunctionTupleType local_functions(const EntityType& entity) const
  {
    CHECK_CRTP(this->as_imp().local_functions(entity))
    return this->as_imp().local_functions(entity);
  }

  template <class E, class D, size_t d, class R, size_t r, size_t rC, class IntersectionType>
  auto evaluate(const LocalfunctionTupleType& local_functions_tuple_entity,
                const LocalfunctionTupleType& local_functions_tuple_neighbor,
                const XT::Functions::LocalfunctionInterface<E, D, d, R, r, rC>& local_source_entity,
                const XT::Functions::LocalfunctionInterface<E, D, d, R, r, rC>& local_source_neighbor,
                const IntersectionType& intersection,
                const Dune::FieldVector<D, d - 1>& x_intersection) const ->
      typename XT::Functions::LocalfunctionSetInterface<E, D, d, R, r, rC>::RangeType
  {
    CHECK_CRTP(this->as_imp().evaluate(local_functions_tuple_entity,
                                       local_functions_tuple_neighbor,
                                       local_source_entity,
                                       local_source_neighbor,
                                       intersection,
                                       x_intersection));
    this->as_imp().evaluate(local_functions_tuple_entity,
                            local_functions_tuple_neighbor,
                            local_source_entity,
                            local_source_neighbor,
                            intersection,
                            x_intersection);
  }
}; // class LocalNumericalCouplingFluxInterface


template <class Traits>
class LocalNumericalBoundaryFluxInterface
    : public XT::CRTPInterface<LocalNumericalBoundaryFluxInterface<Traits>, Traits>,
      internal::IsNumericalBoundaryFlux
{
  typedef typename Traits::LocalfunctionTupleType LocalfunctionTupleType;
  typedef typename Traits::EntityType EntityType;

public:
  LocalfunctionTupleType local_functions(const EntityType& entity) const
  {
    CHECK_CRTP(this->as_imp().local_functions(entity));
    return this->as_imp().local_functions(entity);
  }

  template <class E, class D, size_t d, class R, size_t r, size_t rC, class IntersectionType>
  auto evaluate(const LocalfunctionTupleType& local_functions_tuple,
                const XT::Functions::LocalfunctionInterface<E, D, d, R, r, rC>& local_source_entity,
                const IntersectionType& intersection,
                const Dune::FieldVector<D, d - 1>& x_intersection) const ->
      typename XT::Functions::LocalfunctionSetInterface<E, D, d, R, r, rC>::RangeType
  {
    CHECK_CRTP(this->as_imp().evaluate(local_functions_tuple, local_source_entity, intersection, x_intersection))
    return this->as_imp().evaluate(local_functions_tuple, local_source_entity, intersection, x_intersection);
  }
}; // class LocalNumericalBoundaryFluxInterface

template <class T>
struct is_local_numerical_coupling_flux : std::is_base_of<internal::IsNumericalCouplingFlux, T>
{
};

template <class T>
struct is_local_numerical_boundary_flux : std::is_base_of<internal::IsNumericalBoundaryFlux, T>
{
};


} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_LOCAL_FLUXES_INTERFACES_HH
