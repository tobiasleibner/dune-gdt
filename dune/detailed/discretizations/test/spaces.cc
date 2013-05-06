#include <dune/stuff/test/test_common.hh>

#if HAVE_ALUGRID

#include <vector>

#include <dune/grid/alugrid.hh>

#include <dune/grid/part/leaf.hh>

#include <dune/stuff/grid/provider.hh>

#include <dune/detailed/discretizations/space/continuouslagrange/fem-localfunctions.hh>
#include <dune/detailed/discretizations/space/continuouslagrange/fem.hh>
#include <dune/detailed/discretizations/mapper/interface.hh>
#include <dune/detailed/discretizations/basefunctionset/interface.hh>


namespace Stuff = Dune::Stuff;
namespace DD    = Dune::Detailed::Discretizations;


static const unsigned int dimDomain = 2;
typedef Dune::ALUGrid<dimDomain, dimDomain, Dune::simplex, Dune::conforming> GridType;
typedef typename GridType::ctype DomainFieldType;
typedef double RangeFieldType;
static const unsigned int dimRange = 1;
typedef Dune::grid::Part::Leaf::Const<GridType> GridPartType;

typedef testing::Types<DD::ContinuousLagrangeSpace::FemLocalfunctionsWrapper<GridPartType, 1, RangeFieldType, dimRange>,
                       DD::ContinuousLagrangeSpace::FemWrapper<GridPartType, 1, RangeFieldType, dimRange>> SpaceTypes;

template <class T>
struct SpaceCRTPtest : public ::testing::Test
{
  typedef T SpaceType;

  void check(const GridPartType& gridPart) const
  {
    // check the space
    const SpaceType space(gridPart);
    // check for static information
    typedef typename SpaceType::Traits Traits;
    typedef typename SpaceType::GridPartType S_GridPartType;
    typedef typename SpaceType::DomainFieldType S_DomainFieldType;
    static const unsigned int DUNE_UNUSED(s_dimDomain) = SpaceType::dimDomain;
    typedef typename SpaceType::RangeFieldType S_RangeFieldType;
    static const unsigned int DUNE_UNUSED(s_dimRangeRows) = SpaceType::dimRangeRows;
    static const unsigned int DUNE_UNUSED(s_dimRangeCols) = SpaceType::dimRangeCols;
    static const int DUNE_UNUSED(s_polOrder) = SpaceType::polOrder;
    typedef typename SpaceType::BackendType S_BackendType;
    typedef typename SpaceType::MapperType MapperType;
    typedef typename SpaceType::BaseFunctionSetType BaseFunctionSetType;
    typedef typename SpaceType::EntityType EntityType;
    // check for functionality
    const auto entityIt      = gridPart.template begin<0>();
    const EntityType& entity = *entityIt;
    typedef typename DD::SpaceInterface<Traits> SpaceInterfaceType;
    const SpaceInterfaceType& spaceAsInterface = static_cast<const SpaceInterfaceType&>(space);
    const S_GridPartType& DUNE_UNUSED(s_gridPart) = spaceAsInterface.gridPart();
    const S_BackendType& DUNE_UNUSED(s_backend) = spaceAsInterface.backend();
    const bool DUNE_UNUSED(s_continuous) = spaceAsInterface.continuous();
    const MapperType& mapper                  = spaceAsInterface.mapper();
    const BaseFunctionSetType baseFunctionSet = spaceAsInterface.baseFunctionSet(entity);

    // check the mapper for static information
    typedef typename MapperType::Traits M_Traits;
    typedef typename M_Traits::BackendType M_BackendType;
    // check the mapper for functionality
    typedef DD::MapperInterface<M_Traits> MapperInterfaceType;
    const MapperInterfaceType& mapperAsInterface = static_cast<const MapperInterfaceType&>(mapper);
    const M_BackendType& DUNE_UNUSED(m_backend) = mapperAsInterface.backend();
    const size_t m_maxNumDofs = mapperAsInterface.maxNumDofs();
    const size_t DUNE_UNUSED(m_numDofs) = mapperAsInterface.numDofs(entity);
    Dune::DynamicVector<size_t> globalIndices(m_maxNumDofs, size_t(0));
    mapperAsInterface.globalIndices(entity, globalIndices);
    const size_t DUNE_UNUSED(globalIndex) = mapperAsInterface.mapToGlobal(entity, 0);

    // check the basefunctionset for static information
    typedef typename BaseFunctionSetType::Traits B_Traits;
    typedef typename B_Traits::BackendType B_BackendType;
    typedef typename B_Traits::EntityType B_EntityType;
    typedef typename B_Traits::DomainType B_DomainType;
    typedef typename B_Traits::RangeType B_RangeType;
    typedef typename B_Traits::JacobianRangeType B_JacobianRangeType;
    // check the basefunctionset for functionality
    typedef DD::BaseFunctionSetInterface<B_Traits> BaseFunctionSetInterfaceType;
    const BaseFunctionSetInterfaceType& baseFunctionSetAsInterface =
        static_cast<const BaseFunctionSetInterfaceType&>(baseFunctionSet);
    const B_EntityType& DUNE_UNUSED(b_entity) = baseFunctionSetAsInterface.entity();
    const B_BackendType& DUNE_UNUSED(b_backend) = baseFunctionSetAsInterface.backend();
    const size_t b_size = baseFunctionSetAsInterface.size();
    const unsigned int DUNE_UNUSED(b_order) = baseFunctionSetAsInterface.order();
    const B_DomainType x = entity.geometry().center();
    std::vector<B_RangeType> values(b_size, B_RangeType(0));
    std::vector<B_JacobianRangeType> jacobians(b_size, B_JacobianRangeType(0));
    baseFunctionSetAsInterface.evaluate(x, values);
    baseFunctionSetAsInterface.jacobian(x, jacobians);
  } // ... check()
}; // struct SpaceCRTPtest


TYPED_TEST_CASE(SpaceCRTPtest, SpaceTypes);
TYPED_TEST(SpaceCRTPtest, SpaceCRTP)
{
  const Stuff::GridProviderInterface<GridType>* gridProvider =
      Dune::Stuff::GridProviders<GridType>::create("gridprovider.cube");
  const Dune::grid::Part::Leaf::Const<GridType> gridPart(*(gridProvider->grid()));
  this->check(gridPart);
  delete gridProvider;
}


int main(int argc, char** argv)
{
  test_init(argc, argv);
  return RUN_ALL_TESTS();
}

#else
int main(int argc, char** argv)
{
  std::cerr << "'HAVE_ALUGRID' not set!";
  return 1;
}
#endif // HAVE_ALUGRID
