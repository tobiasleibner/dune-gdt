// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//
// Contributors: Tobias Leibner

#ifndef DUNE_GDT_EVALUATION_LAXFRIEDRICHS_HH
#define DUNE_GDT_EVALUATION_LAXFRIEDRICHS_HH

#include <tuple>
#include <memory>

#include <dune/common/dynmatrix.hh>
#include <dune/common/typetraits.hh>

#include <dune/geometry/referenceelements.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/stuff/common/fmatrix.hh>
#include <dune/stuff/functions/interfaces.hh>
#include <dune/stuff/functions/constant.hh>

#include "interface.hh"

namespace Dune {
namespace GDT {
namespace LocalEvaluation {
namespace LaxFriedrichs {


// forwards
template< class LocalizableFunctionImp, size_t domainDim >
class Inner;

template< class LocalizableFunctionImp, class BoundaryValueFunctionImp >
class Dirichlet;

template< class LocalizableFunctionImp >
class Absorbing;


namespace internal {


/**
 *  \brief  Traits for the Lax-Friedrichs flux evaluation.
 */
template< class LocalizableFunctionImp >
class InnerTraits
{
  static_assert(std::is_base_of< Dune::Stuff::IsLocalizableFunction, LocalizableFunctionImp >::value,
                "LocalizableFunctionImp has to be derived from Stuff::IsLocalizableFunction.");
public:
  typedef LocalizableFunctionImp                                    LocalizableFunctionType;
  static const unsigned int dimDomain = LocalizableFunctionType::dimDomain;
  static const unsigned int dimRange = LocalizableFunctionType::dimRange;
  typedef Inner< LocalizableFunctionType, dimDomain >               derived_type;
  typedef typename LocalizableFunctionType::EntityType              EntityType;
  typedef typename LocalizableFunctionType::DomainFieldType         DomainFieldType;
  typedef typename LocalizableFunctionType::RangeFieldType          RangeFieldType;
  typedef typename LocalizableFunctionType::LocalfunctionType       LocalfunctionType;
  typedef std::tuple< std::shared_ptr< LocalfunctionType > >        LocalfunctionTupleType;
  static_assert(LocalizableFunctionType::dimRangeCols == 1, "Not implemented for dimRangeCols > 1!");
  typedef typename Dune::YaspGrid< dimRange >::template Codim< 0 >::Entity              FluxSourceEntityType;
  typedef Dune::Stuff::GlobalFunctionInterface< FluxSourceEntityType,
                                                RangeFieldType, dimRange,
                                                RangeFieldType, dimRange, dimDomain >   AnalyticalFluxType;

  typedef typename AnalyticalFluxType::RangeType                                        FluxRangeType;
  typedef typename Dune::Stuff::LocalfunctionSetInterface< EntityType,
                                                           DomainFieldType, dimDomain,
                                                           RangeFieldType, dimRange, 1 >::RangeType  RangeType;
}; // class InnerTraits

/**
 *  \brief  Traits for the Lax-Friedrichs flux evaluation at Dirichlet boundary intersections .
 */
template< class LocalizableFunctionImp, class BoundaryValueFunctionImp >
class DirichletTraits
    : public InnerTraits< LocalizableFunctionImp >
{
  typedef InnerTraits< LocalizableFunctionImp >                            BaseType;
public:
  typedef LocalizableFunctionImp                                           LocalizableFunctionType;
  typedef BoundaryValueFunctionImp                                         BoundaryValueFunctionType;
  typedef typename BoundaryValueFunctionType::LocalfunctionType            BoundaryValueLocalfunctionType;
  typedef Dirichlet< LocalizableFunctionType, BoundaryValueFunctionType >  derived_type;
  using typename BaseType::EntityType;
  using typename BaseType::DomainFieldType;
  using typename BaseType::RangeFieldType;
  using typename BaseType::LocalfunctionType;
  typedef std::tuple< std::shared_ptr< LocalfunctionType >,
                      std::shared_ptr< BoundaryValueLocalfunctionType > >  LocalfunctionTupleType;
  using BaseType::dimDomain;
  using BaseType::dimRange;
  using typename BaseType::FluxSourceEntityType;
  using typename BaseType::AnalyticalFluxType;
  using typename BaseType::FluxRangeType;
  using typename BaseType::RangeType;
  typedef typename BoundaryValueFunctionType::DomainType  DomainType;
}; // class DirichletTraits

/**
 *  \brief  Traits for the Lax-Friedrichs flux evaluation on absorbing boundary.
 */
template< class LocalizableFunctionImp >
class AbsorbingTraits
   : public InnerTraits< LocalizableFunctionImp >
{
  typedef InnerTraits< LocalizableFunctionImp > BaseType;
public:
  typedef LocalizableFunctionImp                LocalizableFunctionType;
  typedef Absorbing< LocalizableFunctionType >  derived_type;
  using typename BaseType::EntityType;
  using typename BaseType::DomainFieldType;
  using typename BaseType::RangeFieldType;
  using typename BaseType::LocalfunctionType;
  typedef typename std::tuple< >                LocalfunctionTupleType;
  using BaseType::dimDomain;
  using BaseType::dimRange;
  using typename BaseType::FluxSourceEntityType;
  using typename BaseType::AnalyticalFluxType;
  using typename BaseType::FluxRangeType;
  using typename BaseType::RangeType;
}; // class AbsorbingTraits


} // namespace internal


/**
 *  \brief  Lax-Friedrichs flux evaluation for inner intersections and periodic boundary intersections.
 */
template< class LocalizableFunctionImp, size_t domainDim = LocalizableFunctionImp::dimDomain >
class Inner
  : public LocalEvaluation::Codim1Interface< internal::InnerTraits< LocalizableFunctionImp >, 4 >
{
public:
  typedef internal::InnerTraits< LocalizableFunctionImp >           Traits;
  typedef typename Traits::LocalizableFunctionType                  LocalizableFunctionType;
  typedef typename Traits::LocalfunctionTupleType                   LocalfunctionTupleType;
  typedef typename Traits::EntityType                               EntityType;
  typedef typename Traits::DomainFieldType                          DomainFieldType;
  typedef typename Traits::RangeFieldType                           RangeFieldType;
  typedef typename Traits::AnalyticalFluxType                       AnalyticalFluxType;
  typedef typename Traits::FluxRangeType                            FluxRangeType;
  typedef typename Traits::RangeType                                RangeType;
  typedef typename LocalizableFunctionType::DomainType              DomainType;
  static const size_t dimDomain = Traits::dimDomain;
  static const size_t dimRange = Traits::dimRange;

  explicit Inner(const AnalyticalFluxType& analytical_flux,
                 const LocalizableFunctionType& dx,
                 const double dt,
                 const bool is_linear = false,
                 const bool use_local = false,
                 const bool entity_geometries_equal = false)
    : analytical_flux_(analytical_flux)
    , dx_(dx)
    , dt_(dt)
    , is_linear_(is_linear)
    , use_local_(use_local)
    , entity_geometries_equal_(entity_geometries_equal)
  {
    if (is_linear_ && use_local_ && !max_derivative_calculated_) {
      const auto jacobian_u_i = analytical_flux_.jacobian(RangeType(0));
      // jacobian_u_i is FieldVector< FieldMatrix, ... >
      for (size_t ii = 0; ii < dimDomain; ++ii) {
        auto& derivative_i = jacobian_u_i[ii];
        if (derivative_i.infinity_norm() > max_derivative_[ii]) {
          max_derivative_[ii] = derivative_i.infinity_norm();
        }
      }
      max_derivative_calculated_ = true;
    }
    geometry_evaluated_ = false;
  }

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(dx_.local_function(entity));
  }

  size_t order(const LocalfunctionTupleType& /*localFunctionsEntity*/,
               const LocalfunctionTupleType& /*localFunctionsNeighbor*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 >& /*testBaseEntity*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 >& /*ansatzBaseEntity*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 >& /*testBaseNeighbor*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 >& /*ansatzBaseNeighbor*/) const
  {
    DUNE_THROW(NotImplemented, "Not meant to be integrated");
  }

  /**
   *  \brief  Computes a quaternary codim 1 evaluation.
   *  \tparam IntersectionType      A model of Dune::Intersection< ... >
   *  \tparam R                     RangeFieldType
   *  \tparam r{T,A}                dimRange of the {testBase*,ansatzBase*}
   *  \tparam rC{T,A}               dimRangeRows of the {testBase*,ansatzBase*}
   *  \attention entityEntityRet, entityEntityRet, entityEntityRet and neighborEntityRet are assumed to be zero!
   */
  template< class IntersectionType >
  void evaluate(const LocalfunctionTupleType& localFunctionsEntity,
                const LocalfunctionTupleType& /*localFunctionsNeighbor*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >& /*testBaseEntity*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >& ansatzBaseEntity,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >& /*testBaseNeighbor*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >& ansatzBaseNeighbor,
                const IntersectionType& intersection,
                const Dune::FieldVector< DomainFieldType, dimDomain - 1 >& localPoint,
                Dune::DynamicMatrix< RangeFieldType >& /*entityEntityRet*/,
                Dune::DynamicMatrix< RangeFieldType >& /*neighborNeighborRet*/,
                Dune::DynamicMatrix< RangeFieldType >& entityNeighborRet,
                Dune::DynamicMatrix< RangeFieldType >& /*neighborEntityRet*/) const
  {
    const auto intersection_center_entity = intersection.geometryInInside().center();
    const auto intersection_center_neighbor = intersection.geometryInOutside().center();
    const RangeType u_i = ansatzBaseEntity.evaluate(intersection_center_entity)[0];
    //std::cout << "u_i: " << DSC::toString(u_i) << std::endl;
    RangeType u_j = ansatzBaseNeighbor.evaluate(intersection_center_neighbor)[0];
    //std::cout << "u_j: " << DSC::toString(u_j) << std::endl;
    FluxRangeType f_u_i_plus_f_u_j = analytical_flux_.evaluate(u_i);
    //std::cout << "f_u_i: " << DSC::toString(f_u_i_plus_f_u_j) << std::endl;
    //std::cout << "f_u_j: " << DSC::toString(analytical_flux_.evaluate(u_j)) << std::endl;
    f_u_i_plus_f_u_j += analytical_flux_.evaluate(u_j);
    auto n_ij = intersection.unitOuterNormal(localPoint);
    // find direction of unit outer normal
    size_t coord = 0;
#ifndef NDEBUG
    size_t num_zeros = 0;
#endif //NDEBUG
    for (size_t ii = 0; ii < dimDomain; ++ii) {
      if (DSC::FloatCmp::eq(n_ij[ii], RangeFieldType(1)) || DSC::FloatCmp::eq(n_ij[ii], RangeFieldType(-1)))
        coord = ii;
      else if (DSC::FloatCmp::eq(n_ij[ii], RangeFieldType(0))) {
#ifndef NDEBUG
        ++num_zeros;
#endif //NDEBUG
      }
      else
        DUNE_THROW(Dune::NotImplemented, "Godunov flux is only implemented for axis parallel cube grids");
    }

    if (!use_local_) {
      const RangeFieldType dx = std::get< 0 >(localFunctionsEntity)->evaluate(intersection_center_entity)[0];
      max_derivative_ = dx/dt_;
    } else {
      if (!is_linear_) {
        max_derivative_ = 0;
        const auto jacobian_u_i = analytical_flux_.jacobian(u_i);
        const auto jacobian_u_j = analytical_flux_.jacobian(u_j);
        // jacobian_u_i is either a FieldMatrix or a FieldVector< FieldMatrix, ... >, so derivative_i is either a row of
        // the FieldMatrix (i.e. a FieldVector) or a FieldMatrix. In both cases, the correct infinity norm is obtained.
        for (size_t ii = 0; ii < dimDomain; ++ii) {
          auto& derivative_i = jacobian_u_i[ii];
          if (derivative_i.infinity_norm() > max_derivative_[ii]) {
            max_derivative_[ii] = derivative_i.infinity_norm();
          }
        }
        for (size_t ii = 0; ii < dimDomain; ++ii) {
          auto& derivative_j = jacobian_u_j[ii];
          if (derivative_j.infinity_norm() > max_derivative_[ii]) {
            max_derivative_[ii] = derivative_j.infinity_norm();
          }
        }
      }
    }
    if (!entity_geometries_equal_ || !geometry_evaluated_) {
      vol_intersection_ = intersection.geometry().volume();
      const auto& reference_element
        = Dune::ReferenceElements< DomainFieldType, dimDomain >::general(ansatzBaseEntity.entity().geometry().type());
      num_neighbors_ = reference_element.size(1);
      geometry_evaluated_ = true;
    }
    //entityNeighborRet[0][kk] = ((f_u_i[kk] + f_u_j[kk])*n_ij*0.5 - (u_j - u_i)[kk]*max_derivative_*1.0/num_neighbors_)*vol_intersection_
    //calculate (u_j - u_i)*max_derivative_/num_neighbors_*vol_intersection_
    u_j -= u_i;
    u_j *= max_derivative_[coord]/num_neighbors_*vol_intersection_;
    //std::cout << max_derivative_[coord]/num_neighbors_*vol_intersection_ << std::endl;
    // scale n_ij by 0.5*vol_intersection_
    n_ij[coord] *= 0.5*vol_intersection_;
    // calculate flux
    for (size_t kk = 0; kk < dimRange; ++kk)
      entityNeighborRet[0][kk] = f_u_i_plus_f_u_j[kk][coord]*n_ij[coord] - u_j[kk];
    //std::cout << "n_ij: " << DSC::toString(n_ij) << ", ret: " << DSC::toString(entityNeighborRet[0]) << ", u_i: " << DSC::toString(u_i) << ", u_j: " << DSC::toString(u_j) << ", fuiplusfuj: " << DSC::toString(f_u_i_plus_f_u_j) << std::endl;
  } // void evaluate(...) const

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& dx_;
  const double dt_;
  const bool is_linear_;
  const bool use_local_;
  const bool entity_geometries_equal_;
  static DomainType max_derivative_;
  static bool max_derivative_calculated_;
  static bool geometry_evaluated_;
  mutable RangeFieldType vol_intersection_;
  mutable int num_neighbors_;
}; // class Inner

template < class LocalizableFunctionImp, size_t dimDomain >
typename Inner< LocalizableFunctionImp, dimDomain >::DomainType
Inner< LocalizableFunctionImp, dimDomain >::max_derivative_;

template < class LocalizableFunctionImp, size_t dimDomain >
bool
Inner< LocalizableFunctionImp, dimDomain >::max_derivative_calculated_(false);

template < class LocalizableFunctionImp, size_t dimDomain >
bool
Inner< LocalizableFunctionImp, dimDomain >::geometry_evaluated_(false);

/**
 *  \brief  Lax-Friedrichs flux evaluation for inner intersections and periodic boundary intersections.
 */
template< class LocalizableFunctionImp >
class Inner< LocalizableFunctionImp, 1 >
  : public LocalEvaluation::Codim1Interface< internal::InnerTraits< LocalizableFunctionImp >, 4 >
{
public:
  typedef internal::InnerTraits< LocalizableFunctionImp >           Traits;
  typedef typename Traits::LocalizableFunctionType                  LocalizableFunctionType;
  typedef typename Traits::LocalfunctionTupleType                   LocalfunctionTupleType;
  typedef typename Traits::EntityType                               EntityType;
  typedef typename Traits::DomainFieldType                          DomainFieldType;
  typedef typename Traits::RangeFieldType                           RangeFieldType;
  typedef typename Traits::AnalyticalFluxType                       AnalyticalFluxType;
  typedef typename Traits::FluxRangeType                            FluxRangeType;
  typedef typename Traits::RangeType                                RangeType;
  static const size_t dimDomain = Traits::dimDomain;
  static const size_t dimRange = Traits::dimRange;

  explicit Inner(const AnalyticalFluxType& analytical_flux, const LocalizableFunctionType& dx, const double dt, const bool use_local = false, const bool = false, const bool = false)
    : analytical_flux_(analytical_flux)
    , dx_(dx)
    , dt_(dt)
    , use_local_(use_local)
  {}

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(dx_.local_function(entity));
  }

  size_t order(const LocalfunctionTupleType& /*localFunctionsEntity*/,
               const LocalfunctionTupleType& /*localFunctionsNeighbor*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 >& /*testBaseEntity*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 >& /*ansatzBaseEntity*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 >& /*testBaseNeighbor*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 >& /*ansatzBaseNeighbor*/) const
  {
    DUNE_THROW(NotImplemented, "Not meant to be integrated");
  }

  /**
   *  \brief  Computes a quaternary codim 1 evaluation.
   *  \tparam IntersectionType      A model of Dune::Intersection< ... >
   *  \tparam R                     RangeFieldType
   *  \tparam r{T,A}                dimRange of the {testBase*,ansatzBase*}
   *  \tparam rC{T,A}               dimRangeRows of the {testBase*,ansatzBase*}
   *  \attention entityEntityRet, entityEntityRet, entityEntityRet and neighborEntityRet are assumed to be zero!
   */
  template< class IntersectionType >
  void evaluate(const LocalfunctionTupleType& localFunctionsEntity,
                const LocalfunctionTupleType& /*localFunctionsNeighbor*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >& /*testBaseEntity*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >& ansatzBaseEntity,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >& /*testBaseNeighbor*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 >& ansatzBaseNeighbor,
                const IntersectionType& intersection,
                const Dune::FieldVector< DomainFieldType, dimDomain - 1 >& localPoint,
                Dune::DynamicMatrix< RangeFieldType >& /*entityEntityRet*/,
                Dune::DynamicMatrix< RangeFieldType >& /*neighborNeighborRet*/,
                Dune::DynamicMatrix< RangeFieldType >& entityNeighborRet,
                Dune::DynamicMatrix< RangeFieldType >& /*neighborEntityRet*/) const
  {
    const auto intersection_center_entity = intersection.geometryInInside().center();
    const auto intersection_center_neighbor = intersection.geometryInOutside().center();
    RangeType u_i = ansatzBaseEntity.evaluate(intersection_center_entity)[0];
    const RangeType u_j = ansatzBaseNeighbor.evaluate(intersection_center_neighbor)[0];

    const auto n_ij = intersection.unitOuterNormal(localPoint);
    const RangeFieldType dx = std::get< 0 >(localFunctionsEntity)->evaluate(intersection_center_entity)[0];
    RangeFieldType max_derivative = dx/dt_;
    if (use_local_) {
      max_derivative = 0;
      const auto jacobian_u_i = analytical_flux_.jacobian(u_i);
      const auto jacobian_u_j = analytical_flux_.jacobian(u_j);
      // jacobian_u_i is either a FieldMatrix or a FieldVector< FieldMatrix, ... >, so derivative_i is either a row of
      // the FieldMatrix (i.e. a FieldVector) or a FieldMatrix. In both cases, the correct infinity norm is obtained.
      for (auto& derivative_i : jacobian_u_i) {
        if (derivative_i.infinity_norm() > max_derivative) {
          max_derivative = derivative_i.infinity_norm();
        }
      }
      for (auto& derivative_j : jacobian_u_j) {
        if (derivative_j.infinity_norm() > max_derivative) {
          max_derivative = derivative_j.infinity_norm();
        }
      }
    }
    // entityNeighborRet[0] = 0.5*((f(u_i) + f(u_j))*n_ij + max_derivative*(u_i - u_j)) where max_derivative = dx/dt if
    // we dont use the local LxF method. As the FieldVector does not provide an operator+, we have to split the expression.
    // calculate n_ij*(f(u_i) + f(u_j)) first
    entityNeighborRet[0] = Dune::DynamicVector< RangeFieldType >(analytical_flux_.evaluate(u_i));
    entityNeighborRet[0] += analytical_flux_.evaluate(u_j);
    if (n_ij < 0)
      entityNeighborRet[0] *= n_ij;
    // add max_derivative*(u_i - u_j)
    u_i -= u_j;
    entityNeighborRet[0].axpy(max_derivative, u_i);
    // multiply by 0.5
    entityNeighborRet[0] *= 0.5;
  } // void evaluate(...) const

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& dx_;
  const double dt_;
  const bool use_local_;
}; // class Inner< ... , 1 >


/**
 *  \brief  Lax-Friedrichs flux evaluation for Dirichlet boundary intersections.
 */
template< class LocalizableFunctionImp, class BoundaryValueFunctionImp >
class Dirichlet
  : public LocalEvaluation::Codim1Interface
                          < internal::DirichletTraits< LocalizableFunctionImp, BoundaryValueFunctionImp >, 2 >
{
public:
  typedef internal::DirichletTraits< LocalizableFunctionImp, BoundaryValueFunctionImp >  Traits;
  typedef typename Traits::BoundaryValueFunctionType                BoundaryValueFunctionType;
  typedef typename Traits::LocalizableFunctionType                  LocalizableFunctionType;
  typedef typename Traits::LocalfunctionTupleType                   LocalfunctionTupleType;
  typedef typename Traits::EntityType                               EntityType;
  typedef typename Traits::DomainFieldType                          DomainFieldType;
  typedef typename Traits::RangeFieldType                           RangeFieldType;
  typedef typename Traits::AnalyticalFluxType                       AnalyticalFluxType;
  typedef typename Traits::FluxRangeType                            FluxRangeType;
  typedef typename Traits::DomainType                               DomainType;
  static const unsigned int dimDomain = Traits::dimDomain;
  static const unsigned int dimRange = Traits::dimRange;

  // lambda = Delta t / Delta x
  explicit Dirichlet(const AnalyticalFluxType& analytical_flux,
                     const LocalizableFunctionType& dx,
                     const double dt,
                     const BoundaryValueFunctionType& boundary_values,
                     const bool use_local = false)
    : analytical_flux_(analytical_flux)
    , dx_(dx)
    , dt_(dt)
    , boundary_values_(boundary_values)
    , use_local_(use_local)
  {}

  LocalfunctionTupleType localFunctions(const EntityType& entity) const
  {
    return std::make_tuple(dx_.local_function(entity), boundary_values_.local_function(entity));
  }

  template< class R, unsigned long rT, unsigned long rCT, unsigned long rA, unsigned long rCA >
  size_t order(const LocalfunctionTupleType /*localFuncs*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& /*testBase*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rA, rCA >& /*ansatzBase*/) const
  {
    DUNE_THROW(NotImplemented, "Not meant to be integrated");
  }

  /**
   *  \brief  Computes a binary codim 1 evaluation.
   *  \tparam IntersectionType    A model of Dune::Intersection< ... >
   *  \tparam R                   RangeFieldType
   *  \tparam r{T,A}              dimRange of the {testBase*,ansatzBase*}
   *  \tparam rC{T,A}             dimRangeRows of the {testBase*,ansatzBase*}
   *  \attention ret is assumed to be zero!
   */
  template< class IntersectionType, class R >
  void evaluate(const LocalfunctionTupleType& localFuncs,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, dimRange, 1 >& /*testBase*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, dimRange, 1 >& ansatzBase,
                const IntersectionType& intersection,
                const Dune::FieldVector< DomainFieldType, dimDomain - 1 >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    const auto intersection_center_local = intersection.geometryInInside().center();
    const auto u_i = ansatzBase.evaluate(intersection_center_local)[0];
    const auto u_j = std::get< 1 >(localFuncs)->evaluate(intersection_center_local);
    const FluxRangeType f_u_i_temp = analytical_flux_.evaluate(u_i);
    const FluxRangeType f_u_j_temp = analytical_flux_.evaluate(u_j);
    DSC::FieldMatrix< RangeFieldType, dimRange, dimDomain > f_u_i;
    DSC::FieldMatrix< RangeFieldType, dimRange, dimDomain > f_u_j;
    for (size_t ii = 0; ii < dimRange; ++ii) {
      f_u_i[ii] = f_u_i_temp[ii];
      f_u_j[ii] = f_u_j_temp[ii];
    }
    const auto n_ij = intersection.unitOuterNormal(localPoint);
    const RangeFieldType dx = std::get< 0 >(localFuncs)->evaluate(intersection_center_local)[0];
    RangeFieldType max_derivative = dx/dt_;
    if (use_local_) {
      max_derivative = 0;
      const auto jacobian_u_i = analytical_flux_.jacobian(u_i);
      const auto jacobian_u_j = analytical_flux_.jacobian(u_j);
      // jacobian_u_i is either a FieldMatrix or a FieldVector< FieldMatrix, ... >, so derivative_i is either a row of
      // the FieldMatrix (i.e. a FieldVector) or a FieldMatrix. In both cases, the correct infinity norm is obtained.
      for (auto& derivative_i : jacobian_u_i) {
        if (derivative_i.infinity_norm() > max_derivative) {
          max_derivative = derivative_i.infinity_norm();
        }
      }
      for (auto& derivative_j : jacobian_u_j) {
        if (derivative_j.infinity_norm() > max_derivative) {
          max_derivative = derivative_j.infinity_norm();
        }
      }
    }
    RangeFieldType vol_intersection = 1;
    int num_neighbors = 2;
    if (dimDomain != 1) {
      vol_intersection = intersection.geometry().volume();
      const auto& reference_element
          = Dune::ReferenceElements< DomainFieldType, dimDomain >::general(ansatzBase.entity().geometry().type());
      num_neighbors = reference_element.size(1);
    }
    for (size_t kk = 0; kk < dimRange; ++kk)
      ret[0][kk] = ((f_u_i[kk] + f_u_j[kk])*n_ij*0.5 - (u_j - u_i)[kk]*max_derivative*1.0/num_neighbors)*vol_intersection;
  } // void evaluate(...) const

private:
  const AnalyticalFluxType& analytical_flux_;
  const LocalizableFunctionType& dx_;
  const double dt_;
  const BoundaryValueFunctionType& boundary_values_;
  const bool use_local_;
}; // class Dirichlet


/**
 *  \brief  Lax-Friedrichs flux evaluation for absorbing boundary conditions on boundary intersections.
 */
template< class LocalizableFunctionImp >
class Absorbing
  : public LocalEvaluation::Codim1Interface
                          < internal::AbsorbingTraits< LocalizableFunctionImp >, 2 >
{
public:
  typedef internal::AbsorbingTraits< LocalizableFunctionImp >       Traits;
  typedef typename Traits::LocalizableFunctionType                  LocalizableFunctionType;
  typedef typename Traits::LocalfunctionTupleType                   LocalfunctionTupleType;
  typedef typename Traits::EntityType                               EntityType;
  typedef typename Traits::DomainFieldType                          DomainFieldType;
  typedef typename Traits::RangeFieldType                           RangeFieldType;
  typedef typename Traits::AnalyticalFluxType                       AnalyticalFluxType;
  typedef typename Traits::FluxRangeType                            FluxRangeType;
  static const size_t dimDomain = Traits::dimDomain;
  static const size_t dimRange = Traits::dimRange;

  explicit Absorbing(const AnalyticalFluxType& analytical_flux, const LocalizableFunctionType& /*dx*/, const double /*dt*/, const bool /*use_local*/)
    : analytical_flux_(analytical_flux)
  {}

  LocalfunctionTupleType localFunctions(const EntityType& /*entity*/) const
  {
    return std::make_tuple();
  }

  template< class R, unsigned long rT, unsigned long rCT, unsigned long rA, unsigned long rCA >
  size_t order(const LocalfunctionTupleType /*localFuncs*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rT, rCT >& /*testBase*/,
               const Stuff::LocalfunctionSetInterface
                   < EntityType, DomainFieldType, dimDomain, R, rA, rCA >& /*ansatzBase*/) const
  {
    DUNE_THROW(NotImplemented, "Not meant to be integrated");
  }

  /**
   *  \brief  Computes a binary codim 1 evaluation.
   *  \tparam IntersectionType    A model of Dune::Intersection< ... >
   *  \tparam R                   RangeFieldType
   *  \attention ret is assumed to be zero!
   */
  template< class IntersectionType, class R >
  void evaluate(const LocalfunctionTupleType& /*localFunctions*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, dimRange, 1 >& /*testBase*/,
                const Stuff::LocalfunctionSetInterface
                    < EntityType, DomainFieldType, dimDomain, R, dimRange, 1 >& ansatzBase,
                const IntersectionType& intersection,
                const Dune::FieldVector< DomainFieldType, dimDomain - 1 >& localPoint,
                Dune::DynamicMatrix< R >& ret) const
  {
    const auto& u_i = ansatzBase.evaluate(intersection.geometryInInside().center());
    assert(u_i.size() == 1);
    const FluxRangeType f_u_i_temp = analytical_flux_.evaluate(u_i[0]);
    DSC::FieldMatrix< RangeFieldType, dimRange, dimDomain > f_u_i;
    for (size_t ii = 0; ii < dimRange; ++ii) {
      f_u_i[ii] = f_u_i_temp[ii];
    }
    const auto n_ij = intersection.unitOuterNormal(localPoint);
    RangeFieldType vol_intersection = 1;
    if (dimDomain != 1) {
      vol_intersection = intersection.geometry().volume();
    }
    for (size_t kk = 0; kk < dimRange; ++kk)
      ret[0][kk] = (f_u_i[kk] + f_u_i[kk])*n_ij*0.5*vol_intersection;
  } // void evaluate(...) const

private:
  const AnalyticalFluxType& analytical_flux_;
}; // class Absorbing

} // namespace LaxFriedrichs
} // namespace Evaluation
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_EVALUATION_LAXFRIEDRICHS_HH
