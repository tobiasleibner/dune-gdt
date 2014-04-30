// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_GDT_SPACES_CONTINUOUSLAGRANGE_FEM_HH
#define DUNE_GDT_SPACES_CONTINUOUSLAGRANGE_FEM_HH

#include <memory>
#include <type_traits>

#include <dune/common/typetraits.hh>

#if HAVE_DUNE_FEM
# include <dune/fem/space/common/functionspace.hh>
# include <dune/fem/space/lagrangespace.hh>
#endif // HAVE_DUNE_FEM

#include "../../mapper/fem.hh"
#include "../../basefunctionset/fem.hh"

#include "base.hh"
#include "../constraints.hh"

namespace Dune {
namespace GDT {
namespace Spaces {
namespace ContinuousLagrange {

#if HAVE_DUNE_FEM


// forward, to be used in the traits and to allow for specialization
template< class GridPartImp, int polynomialOrder, class RangeFieldImp, int rangeDim, int rangeDimCols = 1 >
class FemBased
{
  static_assert(Dune::AlwaysFalse< GridPartImp >::value, "Untested for these dimensions!");
};


template< class GridPartImp, int polynomialOrder, class RangeFieldImp, int rangeDim, int rangeDimCols = 1 >
class FemBasedTraits
{
public:
  typedef FemBased< GridPartImp, polynomialOrder, RangeFieldImp, rangeDim, rangeDimCols > derived_type;
  typedef GridPartImp GridPartType;
  typedef typename GridPartType::GridViewType GridViewType;
  static const int                            polOrder = polynomialOrder;
  static_assert(polOrder >= 1, "Wrong polOrder given!");
  static const unsigned int             dimDomain = GridPartType::dimension;
private:
  typedef typename GridPartType::ctype  DomainFieldType;
public:
  typedef RangeFieldImp                 RangeFieldType;
  static const unsigned int             dimRange = rangeDim;
  static const unsigned int             dimRangeCols = rangeDimCols;
private:
  typedef Dune::Fem::FunctionSpace< DomainFieldType, RangeFieldType, dimDomain, dimRange > FunctionSpaceType;
public:
  typedef Dune::Fem::LagrangeDiscreteFunctionSpace< FunctionSpaceType, GridPartType, polOrder > BackendType;
  typedef Mapper::FemDofWrapper< typename BackendType::MapperType > MapperType;
  typedef typename GridPartType::template Codim< 0 >::EntityType EntityType;
  typedef BaseFunctionSet::FemWrapper
      < typename BackendType::BaseFunctionSetType, EntityType, DomainFieldType, dimDomain,
        RangeFieldType, dimRange, dimRangeCols > BaseFunctionSetType;
  static const Stuff::Grid::ChoosePartView part_view_type = Stuff::Grid::ChoosePartView::part;
  static const bool needs_grid_view = false;
}; // class SpaceWrappedFemContinuousLagrangeTraits


// untested for the vector-valued case, especially Spaces::ContinuousLagrangeBase
template< class GridPartImp, int polynomialOrder, class RangeFieldImp, int rangeDim >
class FemBased< GridPartImp, polynomialOrder, RangeFieldImp, rangeDim, 1 >
  : public Spaces::ContinuousLagrangeBase< FemBasedTraits< GridPartImp, polynomialOrder, RangeFieldImp, rangeDim, 1 >
                                      , GridPartImp::dimension, RangeFieldImp, rangeDim, 1 >
{
  typedef Spaces::ContinuousLagrangeBase< FemBasedTraits< GridPartImp, polynomialOrder, RangeFieldImp, rangeDim, 1 >
                                     , GridPartImp::dimension, RangeFieldImp, rangeDim, 1 > BaseType;
  typedef FemBased< GridPartImp, polynomialOrder, RangeFieldImp, rangeDim, 1 >                             ThisType;
public:
  typedef FemBasedTraits< GridPartImp, polynomialOrder, RangeFieldImp, rangeDim, 1 > Traits;

  typedef typename Traits::GridPartType GridPartType;
  typedef typename Traits::GridViewType GridViewType;
  static const int                      polOrder = Traits::polOrder;
  typedef typename GridPartType::ctype  DomainFieldType;
  static const unsigned int             dimDomain = GridPartType::dimension;
  typedef typename Traits::RangeFieldType RangeFieldType;
  static const unsigned int               dimRange = Traits::dimRange;
  static const unsigned int               dimRangeCols = Traits::dimRangeCols;

  typedef typename Traits::BackendType          BackendType;
  typedef typename Traits::MapperType           MapperType;
  typedef typename Traits::BaseFunctionSetType  BaseFunctionSetType;
  typedef typename Traits::EntityType           EntityType;

  typedef Dune::Stuff::LA::SparsityPatternDefault PatternType;

  FemBased(const std::shared_ptr< const GridPartType >& gridP)
    : gridPart_(gridP)
    , gridView_(std::make_shared< GridViewType >(gridPart_->gridView()))
    , backend_(std::make_shared< BackendType >(const_cast< GridPartType& >(*(gridPart_))))
    , mapper_(std::make_shared< MapperType >(backend_->mapper()))
    , tmpMappedRows_(mapper_->maxNumDofs())
    , tmpMappedCols_(mapper_->maxNumDofs())
  {}

  FemBased(const ThisType& other)
    : gridPart_(other.gridPart_)
    , gridView_(other.gridView_)
    , backend_(other.backend_)
    , mapper_(other.mapper_)
    , tmpMappedRows_(mapper_->maxNumDofs())
    , tmpMappedCols_(mapper_->maxNumDofs())
  {}

  ThisType& operator=(const ThisType& other)
  {
    if (this != &other) {
      gridPart_ = other.gridPart_;
      gridView_ = other.gridView_;
      backend_ = other.backend_;
      mapper_ = other.mapper_;
      tmpMappedRows_.resize(mapper_->maxNumDofs());
      tmpMappedCols_.resize(mapper_->maxNumDofs());
    }
    return *this;
  }

  ~FemBased() {}

  const std::shared_ptr< const GridPartType >& grid_part() const
  {
    return gridPart_;
  }

  const std::shared_ptr< const GridViewType >& grid_view() const
  {
    return gridView_;
  }

  const BackendType& backend() const
  {
    return *backend_;
  }

  const MapperType& mapper() const
  {
    return *mapper_;
  }

  BaseFunctionSetType base_function_set(const EntityType& entity) const
  {
    return BaseFunctionSetType(*backend_, entity);
  }

private:
  std::shared_ptr< const GridPartType > gridPart_;
  std::shared_ptr< const GridViewType > gridView_;
  std::shared_ptr< const BackendType > backend_;
  std::shared_ptr< const MapperType > mapper_;
  mutable Dune::DynamicVector< size_t > tmpMappedRows_;
  mutable Dune::DynamicVector< size_t > tmpMappedCols_;
}; // class FemBased< ..., 1 >


#else // HAVE_DUNE_FEM


template< class GridPartImp, int polynomialOrder, class RangeFieldImp, int rangeDim, int rangeDimCols = 1 >
class FemBased
{
  static_assert(Dune::AlwaysFalse< GridPartImp >::value, "You are missing dune-fem!");
};


#endif // HAVE_DUNE_FEM

} // namespace ContinuousLagrange
} // namespace Spaces
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_SPACES_CONTINUOUSLAGRANGE_FEM_HH