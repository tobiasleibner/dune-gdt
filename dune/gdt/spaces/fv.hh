// This file is part of the dune-gdt project:
//   https://github.com/dune-community/dune-gdt
// Copyright 2010-2017 dune-gdt developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   Felix Schindler (2016 - 2017)
//   Rene Milk       (2015, 2017)

#ifndef DUNE_GDT_SPACES_FV_HH
#define DUNE_GDT_SPACES_FV_HH

#include <dune/xt/grid/gridprovider/provider.hh>
#include <dune/xt/grid/layers.hh>

#include "fv/interface.hh"
#include "fv/default.hh"
#include "fv/product.hh"

#include <dune/gdt/playground/spaces/block.hh>

namespace Dune {
namespace GDT {


template <class GridType,
          XT::Grid::Layers layer_type,
          Backends backend_type,
          class RangeFieldType,
          size_t dimRange,
          size_t dimRangeCols = 1,
          XT::Grid::Backends grid_backend_type = layer_from_backend<backend_type>::type>
class FvSpaceProvider
{
public:
  static const constexpr SpaceType space_type = SpaceType::fv;
  static const constexpr Backends space_backend = backend_type;
  static const constexpr XT::Grid::Layers grid_layer = layer_type;
  static const constexpr XT::Grid::Backends layer_backend = grid_backend_type;

  typedef typename XT::Grid::Layer<GridType, layer_type, layer_backend>::type GridLayerType;

private:
  template <class G, class R, size_t r, size_t rC, GDT::Backends b>
  struct SpaceChooser
  {
    static_assert(AlwaysFalse<G>::value, "No space available for this backend!");
  };

  template <class G, class R, size_t r, size_t rC>
  struct SpaceChooser<G, R, r, rC, GDT::Backends::gdt>
  {
    typedef GDT::FvSpace<GridLayerType, R, r, rC> Type;
  };

public:
  typedef typename SpaceChooser<GridType, RangeFieldType, dimRange, dimRangeCols, backend_type>::Type Type;
  typedef Type type;

  static Type create(GridLayerType grid_layer)
  {
    return Type(grid_layer);
  }

  template <class DdGridType>
  static Type create(XT::Grid::GridProvider<GridType, DdGridType>& grid_provider, const int level = 0)
  {
    return Type(grid_provider.template layer<layer_type, layer_backend>(level));
  }
}; // class FvSpaceProvider

template <class GridType,
          XT::Grid::Layers layer_type,
          Backends backend_type,
          class RangeFieldType,
          size_t dimRange,
          size_t dimRangeCols = 1,
          XT::Grid::Backends grid_backend_type = layer_from_backend<backend_type>::type>
class FvProductSpaceProvider
{
public:
  static const constexpr SpaceType space_type = SpaceType::product_fv;
  static const constexpr Backends space_backend = backend_type;
  static const constexpr XT::Grid::Layers grid_layer = layer_type;
  static const constexpr XT::Grid::Backends layer_backend = grid_backend_type;

  typedef typename XT::Grid::Layer<GridType, layer_type, layer_backend>::type GridLayerType;

private:
  template <class G, class R, size_t r, size_t rC, GDT::Backends b>
  struct SpaceChooser
  {
    static_assert(AlwaysFalse<G>::value, "No space available for this backend!");
  };

  template <class G, class R, size_t r, size_t rC>
  struct SpaceChooser<G, R, r, rC, GDT::Backends::gdt>
  {
    typedef GDT::FvProductSpace<GridLayerType, R, r, rC> Type;
  };

public:
  typedef typename SpaceChooser<GridType, RangeFieldType, dimRange, dimRangeCols, backend_type>::Type Type;
  typedef Type type;

  static Type create(GridLayerType grid_layer)
  {
    return Type(grid_layer);
  }

  template <class DdGridType>
  static Type create(XT::Grid::GridProvider<GridType, DdGridType>& grid_provider, const int level = 0)
  {
    return Type(grid_provider.template layer<layer_type, layer_backend>(level));
  }
}; // class FvProductSpaceProvider


template <class GridType,
          XT::Grid::Layers layer_type,
          Backends backend_type,
          class RangeFieldType,
          size_t dimRange,
          size_t dimRangeCols = 1,
          XT::Grid::Backends grid_backend_type = layer_from_backend<backend_type>::type>
class BlockFvSpaceProvider
{
public:
  static const constexpr SpaceType space_type = SpaceType::block_fv;
  static const constexpr Backends space_backend = backend_type;
  static const constexpr XT::Grid::Layers grid_layer = layer_type;
  static const constexpr XT::Grid::Backends layer_backend = grid_backend_type;

  typedef typename XT::Grid::Layer<GridType, grid_layer, layer_backend>::type GridLayerType;

private:
  typedef typename FvSpaceProvider<GridType,
                                   grid_layer,
                                   space_backend,
                                   RangeFieldType,
                                   dimRange,
                                   dimRangeCols,
                                   layer_backend>::Type LocalType;

public:
  typedef GDT::BlockSpace<LocalType> Type;
  typedef Type type;

  static Type create(GridLayerType /*grid_layer*/)
  {
    DUNE_THROW(NotImplemented, "Only usable to extract the correct type");
  }

  template <class DdGridType>
  static Type create(XT::Grid::GridProvider<GridType, DdGridType>& /*grid_provider*/, const int /*level*/ = 0)
  {
    DUNE_THROW(NotImplemented, "Only usable to extract the correct type");
  }
}; // class BlockFvSpaceProvider


} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_SPACES_FV_HH
