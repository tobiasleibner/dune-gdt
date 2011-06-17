#ifndef DUNE_FUNCTIONALS_ASSEMBLER_GENERIC_HH
#define DUNE_FUNCTIONALS_ASSEMBLER_GENERIC_HH

// dune-functionals includes
#include <dune/functionals/common/localmatrix.hh>
#include <dune/functionals/common/localvector.hh>

namespace Dune {

namespace Functionals {

namespace Assembler {

namespace System {

template <class AnsatzFunctionSpaceType, class TestFunctionSpaceType = AnsatzFunctionSpaceType>
class Affine
{
public:
  //! constructor
  Affine(const AnsatzFunctionSpaceType& ansatzSpace, const TestFunctionSpaceType& testSpace)
    : ansatzSpace_(ansatzSpace)
    , testSpace_(testSpace)
  {
  }

  //! constructor
  Affine(const AnsatzFunctionSpaceType& ansatzSpace)
    : ansatzSpace_(ansatzSpace)
    , testSpace_(ansatzSpace)
  {
  }

  template <class LocalMatrixAssemblerType, class MatrixType, class LocalVectorAssemblerType, class VectorType,
            class AffineShiftVectorType>
  void assembleSystem(const LocalMatrixAssemblerType& localMatrixAssembler, MatrixType& matrix,
                      const LocalVectorAssemblerType& localVectorAssembler, VectorType& vector,
                      AffineShiftVectorType& affineShiftVector)
  {
    // some types
    typedef typename AnsatzFunctionSpaceType::IteratorType EntityIteratorType;

    typedef typename AnsatzFunctionSpaceType::EntityType EntityType;

    typedef typename AnsatzFunctionSpaceType::RangeFieldType RangeFieldType;

    typedef Dune::Functionals::Common::LocalMatrix<RangeFieldType> LocalMatrixType;

    typedef Dune::Functionals::Common::LocalVector<RangeFieldType> LocalVectorType;

    typedef typename AnsatzFunctionSpaceType::ConstraintsType ConstraintsType;

    typedef typename ConstraintsType::LocalConstraintsType LocalConstraintsType;

    typedef typename AnsatzFunctionSpaceType::AffineShiftType AffineShiftType;

    typedef typename LocalMatrixAssemblerType::template LocalVectorAssembler<AffineShiftType>::Type
        LocalAffineShiftVectorAssemblerType;

    const LocalAffineShiftVectorAssemblerType localAffineShiftVectorAssembler =
        localMatrixAssembler.localVectorAssembler(ansatzSpace_.affineShift());

    // common storage for all entities
    LocalMatrixType localMatrix(ansatzSpace_.numMaxLocalDoFs(), testSpace_.numMaxLocalDoFs());
    LocalVectorType localVector(testSpace_.numMaxLocalDoFs());

    // do first gridwalk to assemble
    const EntityIteratorType behindLastEntity = ansatzSpace_.end();
    for (EntityIteratorType entityIterator = ansatzSpace_.begin(); entityIterator != behindLastEntity;
         ++entityIterator) {
      const EntityType& entity = *entityIterator;

      localMatrixAssembler.assembleLocal(ansatzSpace_, testSpace_, entity, matrix, localMatrix);

      localVectorAssembler.assembleLocal(testSpace_, entity, vector, localVector);

      localAffineShiftVectorAssembler.assembleLocal(testSpace_, entity, affineShiftVector, localVector);

    } // done first gridwalk to assemble

    const ConstraintsType constraints = ansatzSpace_.constraints();

    // do second gridwalk, to apply constraints
    for (EntityIteratorType entityIterator = ansatzSpace_.begin(); entityIterator != behindLastEntity;
         ++entityIterator) {
      const EntityType& entity = *entityIterator;

      const LocalConstraintsType& localConstraints = constraints.local(entity);

      applyLocalMatrixConstraints(localConstraints, matrix);
      applyLocalVectorConstraints(localConstraints, vector);

    } // done second gridwalk, to apply constraints

    // apply constraints

  } // end method assembleSystem

private:
  template <class LocalConstraintsType, class MatrixType>
  void applyLocalMatrixConstraints(const LocalConstraintsType& localConstraints, MatrixType& matrix)
  {
    for (unsigned int i = 0; i < localConstraints.rowDofsSize(); ++i) {
      for (unsigned int j = 0; j < localConstraints.columnDofsSize(); ++j) {
        matrix[localConstraints.rowDofs(i)][localConstraints.columnDofs(j)] = localConstraints.localMatrix(i, j);
      }
    }
  } // end applyLocalMatrixConstraints

  template <class LocalConstraintsType, class VectorType>
  void applyLocalVectorConstraints(const LocalConstraintsType& localConstraints, VectorType& vector)
  {
    for (unsigned int i = 0; i < localConstraints.rowDofsSize(); ++i) {
      vector[localConstraints.rowDofs(i)] = 0.0;
    }
  } // end applyLocalVectorConstraints

  const AnsatzFunctionSpaceType& ansatzSpace_;
  const TestFunctionSpaceType& testSpace_;

}; // end class Affine

} // end namespace System

} // end namespace Assembler

} // end namespace Functionals

} // end namespace Dune

#endif // DUNE_FUNCTIONALS_ASSEMBLER_GENERIC_HH
