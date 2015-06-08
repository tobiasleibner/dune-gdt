// This file is part of the dune-gdt project:
//   http://users.dune-project.org/projects/dune-gdt
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//
// Contributors: Tobias Leibner

#ifndef DUNE_GDT_PLAYGROUND_SPACES_DG_PDELABPRODUCT_HH
#define DUNE_GDT_PLAYGROUND_SPACES_DG_PDELABPRODUCT_HH

#include <tuple>

#include <dune/gdt/playground/mapper/productdgpdelab.hh>

#include <dune/gdt/spaces/productinterface.hh>
#include <dune/gdt/spaces/dg/interface.hh>
#include <dune/gdt/playground/spaces/dg/pdelab.hh>

namespace Dune {
namespace GDT {
namespace Spaces {
namespace DG {


// forward, to be used in the traits and to allow for specialization
template< class GridViewImp, int polynomialOrder, class RangeFieldImp, size_t rangeDim, size_t rangeDimCols = 1 >
class PdelabBasedProduct
{
  static_assert(Dune::AlwaysFalse< GridViewImp >::value, "Untested for these dimensions!");
};


namespace internal {

// from https://stackoverflow.com/questions/16853552/how-to-create-a-type-list-for-variadic-templates-that-contains-n-times-the-sam

// in the end, we would like to have something like indices< 1, 2, 3 > for N = 3
template< std::size_t... >
struct indices {};

// we want to call this with empty Indices, i.e. create_indices< N >::type == indices< 1, 2, 3 > for N = 3
template< std::size_t N, std::size_t... Indices>
struct create_indices : create_indices< N-1, N-1, Indices...> {};

// terminating template
template< std::size_t... Indices >
struct create_indices< 0, Indices...> {
  typedef indices<Indices...> type;
};

// T_aliased< T, Index > is always the type T, no matter what Index is
template<typename T, std::size_t index>
using T_aliased = T;

// make_identical_tuple< T, N >::type is a std::tuple< T, ... , T > with a length of N
template< typename T, std::size_t N, typename I = typename create_indices< N >::type >
struct make_identical_tuple;

template< typename T, std::size_t N, std::size_t ...Indices >
struct make_identical_tuple< T, N, indices< Indices... > >
{
    using type = std::tuple<T_aliased<T, Indices>...>;
};



template< class GridViewImp, int polynomialOrder, class RangeFieldImp, size_t rangeDim, size_t rangeDimCols >
class PdelabBasedProductTraits
    : public PdelabBasedTraits< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, rangeDimCols >
{
  typedef PdelabBasedTraits< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, rangeDimCols > BaseType;
public:
  typedef PdelabBasedProduct< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, rangeDimCols > derived_type;
  using typename BaseType::GridViewType;
  static const int polOrder = BaseType::polOrder;
  static const size_t dimDomain = GridViewType::dimension;
  static const size_t dimRange = rangeDim;
  static const size_t dimRangeCols = rangeDimCols;
  using typename BaseType::BackendType;
  using typename BaseType::EntityType;
  using typename BaseType::RangeFieldType;
  typedef Mapper::ProductDG< BackendType, rangeDim, rangeDimCols >          MapperType;
  using BaseType::part_view_type;
  using BaseType::needs_grid_view;

  typedef typename Dune::GDT::Spaces::DG::PdelabBased< GridViewType, polOrder, RangeFieldType, 1, dimRangeCols >  FactorSpaceType;
  typedef typename make_identical_tuple< FactorSpaceType, dimRange >::type                                        SpaceTupleType;
  typedef MapperType                                                                                              FactorMapperType;
};


} // namespace internal


template< class GridViewImp, int polynomialOrder, class RangeFieldImp, size_t rangeDim >
class PdelabBasedProduct< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, 1 >
  : public Dune::GDT::ProductSpaceInterface< internal::PdelabBasedProductTraits< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, 1 > >
{
  typedef PdelabBasedProduct< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, 1 >                          ThisType;
  typedef typename Dune::GDT::ProductSpaceInterface< internal::PdelabBasedProductTraits< GridViewImp, polynomialOrder, RangeFieldImp, rangeDim, 1 > >  BaseType;
public:
  using typename BaseType::Traits;
  using typename BaseType::GridViewType;
  using typename BaseType::EntityType;
  using typename BaseType::BaseFunctionSetType;
  using typename BaseType::MapperType;
  using typename BaseType::CommunicatorType;
  using typename BaseType::BackendType;
  using BaseType::dimDomain;
  using BaseType::dimRange;
  using BaseType::dimRangeCols;
  typedef typename Traits::FactorMapperType FactorMapperType;
  typedef typename Traits::SpaceTupleType   SpaceTupleType;
  typedef typename Traits::FactorSpaceType  FactorSpaceType;

  PdelabBasedProduct(GridViewType gv)
    : grid_view_(gv)
    , factor_space_(grid_view_)
    , factor_mapper_(factor_space_.backend())
    , communicator_(CommunicationChooser<GridViewImp>::create(grid_view_))
    , communicator_prepared_(false)
  {}

  PdelabBasedProduct(const ThisType& other)
    : grid_view_(other.grid_view_)
    , factor_space_(other.factor_space_)
    , factor_mapper_(other.factor_mapper_)
    , communicator_(CommunicationChooser< GridViewImp >::create(grid_view_))
    , communicator_prepared_(false)
  {
    // make sure our new communicator is prepared if other's was
    if (other.communicator_prepared_)
      const auto& DUNE_UNUSED(comm) = this->communicator();
  }

  PdelabBasedProduct(ThisType&& source) = default;

  ThisType& operator=(const ThisType& other) = delete;

  ThisType& operator=(ThisType&& source) = delete;

  const GridViewType& grid_view() const
  {
    return grid_view_;
  }

  const BackendType& backend() const
  {
    return factor_space_.backend();
  }

  const MapperType& mapper() const
  {
    return factor_mapper_;
  }

  BaseFunctionSetType base_function_set(const EntityType& entity) const
  {
    return BaseFunctionSetType(backend(), entity);
  }

  CommunicatorType& communicator() const
  {
    std::lock_guard< std::mutex > DUNE_UNUSED(gg)(communicator_mutex_);
    if (!communicator_prepared_)
      communicator_prepared_ = CommunicationChooser<GridViewType>::prepare(*this, *communicator_);
    return *communicator_;
  } // ... communicator(...)

  const FactorMapperType& factor_mapper() const
  {
    return factor_mapper_;
  }

  template< size_t ii >
  const FactorSpaceType& factor() const
  {
    return factor_space_;
  }

private:
    const GridViewType grid_view_;
    const FactorSpaceType factor_space_;
    const FactorMapperType factor_mapper_;
    mutable std::unique_ptr< CommunicatorType > communicator_;
    mutable bool communicator_prepared_;
    mutable std::mutex communicator_mutex_;
}; // class DefaultProduct< ..., 1 >


} // namespace DG
} // namespace Spaces
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_PLAYGROUND_SPACES_DG_PDELABPRODUCT_HH