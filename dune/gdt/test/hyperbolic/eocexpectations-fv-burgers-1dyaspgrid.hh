// This file is part of the dune-gdt project:
//   https://github.com/dune-community/dune-gdt
// Copyright 2010-2017 dune-gdt developers and contributors. All rights reserved.
// License: Dual licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
//      or  GPL-2.0+ (http://opensource.org/licenses/gpl-license)
//          with "runtime exception" (http://www.dune-project.org/license.html)
// Authors:
//   Felix Schindler (2017)
//   Rene Milk       (2016 - 2017)
//   Tobias Leibner  (2016)

#ifndef DUNE_GDT_TEST_HYPERBOLIC_EOCEXPECTATIONS_FV_BURGERS_1DYASPGRID_HH
#define DUNE_GDT_TEST_HYPERBOLIC_EOCEXPECTATIONS_FV_BURGERS_1DYASPGRID_HH

#include "problems/burgers.hh"
#include "eocexpectations.hh"

namespace Dune {
namespace GDT {
namespace Test {


template <>
class HyperbolicEocExpectations<Hyperbolic::BurgersTestCase<Yasp1, double, 1>,
                                Hyperbolic::ChooseDiscretizer::fv,
                                1,
                                NumericalFluxes::godunov,
                                TimeStepperMethods::explicit_euler,
                                TimeStepperMethods::explicit_euler> : public internal::HyperbolicEocExpectationsBase<1>
{
  typedef Hyperbolic::BurgersTestCase<Yasp1, double, 1> TestCaseType;

public:
  static std::vector<double> results(const TestCaseType& test_case, const std::string type); // ... results(...)
}; // HyperbolicEocExpectations

template <>
class HyperbolicEocExpectations<Hyperbolic::BurgersTestCase<Yasp1, double, 1>,
                                Hyperbolic::ChooseDiscretizer::fv,
                                1,
                                NumericalFluxes::godunov,
                                TimeStepperMethods::dormand_prince,
                                TimeStepperMethods::dormand_prince> : public internal::HyperbolicEocExpectationsBase<1>
{
  typedef Hyperbolic::BurgersTestCase<Yasp1, double, 1> TestCaseType;

public:
  static std::vector<double> results(const TestCaseType& test_case, const std::string type); // ... results(...)
}; // HyperbolicEocExpectations

template <>
class HyperbolicEocExpectations<Hyperbolic::BurgersTestCase<Yasp1, double, 1>,
                                Hyperbolic::ChooseDiscretizer::fv,
                                1,
                                NumericalFluxes::laxfriedrichs,
                                TimeStepperMethods::explicit_euler,
                                TimeStepperMethods::explicit_euler> : public internal::HyperbolicEocExpectationsBase<1>
{
  typedef Hyperbolic::BurgersTestCase<Yasp1, double, 1> TestCaseType;

public:
  static std::vector<double> results(const TestCaseType& test_case, const std::string type); // ... results(...)
}; // HyperbolicEocExpectations


} // namespace Tests
} // namespace GDT
} // namespace Dune

#endif // DUNE_GDT_TEST_HYPERBOLIC_EOCEXPECTATIONS_FV_BURGERS_1DYASPGRID_HH
