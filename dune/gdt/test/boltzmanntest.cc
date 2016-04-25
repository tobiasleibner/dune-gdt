// This file is part of the dune-gdt project:
//   https://github.com/dune-community/dune-gdt
// Copyright 2010-2016 dune-gdt developers and contributors. All rights reserved.
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
// Authors:
//   Felix Schindler (2014, 2016)
//   Rene Milk       (2014)
//   Tobias Leibner  (2016)

#include "config.h"

#include <sys/resource.h>

#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>

#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/fvector.hh>

#include <dune/stuff/common/string.hh>
#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container/common.hh>

#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/operators/fv.hh>
#include <dune/gdt/local/operators/l2-projection.hh>
#include <dune/gdt/spaces/fv/product.hh>
#include <dune/gdt/timestepper/factory.hh>

#include <dune/gdt/test/hyperbolic/problems/2dboltzmann.hh>

using namespace Dune::GDT;


int main(int argc, char* argv[])
{
  try {
    // setup MPI
    typedef Dune::MPIHelper MPIHelper;
    MPIHelper::instance(argc, argv);

    // parse options
    if (argc < 5) {
      std::cerr << "Usage: " << argv[0] << "-threading.max_count THREADS -global.datadir DIR [-gridsize GRIDSIZE]" << std::endl;
      return 1;
    }
    size_t num_threads;
    std::string output_dir;
    std::string grid_size = "100";
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "-threading.max_count") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          num_threads = DSC::fromString< size_t >(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
          DS::threadManager().set_max_threads(num_threads);
          DSC_CONFIG.set("threading.max_count", DSC::toString(num_threads), true);
        } else {
          std::cerr << "-threading.max_count option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-global.datadir") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          output_dir = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
          DSC_CONFIG.set("global.datadir", output_dir, true);
        } else {
          std::cerr << "-global.datadir option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-gridsize") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          grid_size = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
        } else {
          std::cerr << "-gridsize option requires one argument." << std::endl;
          return 1;
        }
      }
    }

    // setup threadmanager
    DSC_CONFIG.set("threading.partition_factor", 1, true);
    // set dimensions
    static const size_t dimDomain = 2;
    // for dimRange > 250, an "exceeded maximum recursive template instantiation limit" error occurs (tested with
    // clang 3.5). You need to pass -ftemplate-depth=N with N > dimRange + 10 to clang for higher dimRange.
    // for Boltzmann2D, this is not dimRange but the maximal moment order
    static const size_t momentOrder = 15;
    //choose GridType
    typedef Dune::YaspGrid< dimDomain >  GridType;
    typedef typename GridType::Codim< 0 >::Entity                                         EntityType;

    // configure Problem
    typedef Dune::GDT::Hyperbolic::Problems::Boltzmann2DCheckerboard< EntityType, double, dimDomain, double, momentOrder > ProblemType;

    static const size_t dimRange = ProblemType::dimRange;

    //create Problem
    const auto problem_ptr = ProblemType::create();
    const auto& problem = *problem_ptr;

    //get grid configuration from problem
    auto grid_config = problem.grid_config();
    grid_config["num_elements"] = "[" + grid_size;
    for (size_t ii = 1; ii < dimDomain; ++ii)
        grid_config["num_elements"] += " " + grid_size;
    grid_config["num_elements"] += "]";


    //get analytical flux, initial and boundary values
    typedef typename ProblemType::FluxType              AnalyticalFluxType;
    typedef typename ProblemType::RHSType               RHSType;
    typedef typename ProblemType::InitialValueType      InitialValueType;
    typedef typename ProblemType::BoundaryValueType     BoundaryValueType;
    typedef typename InitialValueType::DomainFieldType      DomainFieldType;
    typedef typename ProblemType::RangeFieldType        RangeFieldType;
    const std::shared_ptr< const AnalyticalFluxType > analytical_flux = problem.flux();
    const std::shared_ptr< const InitialValueType > initial_values = problem.initial_values();
    const std::shared_ptr< const BoundaryValueType > boundary_values = problem.boundary_values();
    const std::shared_ptr< const RHSType > rhs = problem.rhs();

    //create grid
    std::cout << "Creating Grid..." << std::endl;
    typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
    GridProviderType grid_provider = *(GridProviderType::create(grid_config));
    const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();

    // make a product finite volume space on the leaf grid
    std::cout << "Creating GridView..." << std::endl;
    typedef typename GridType::LeafGridView                                        GridViewType;
    const GridViewType grid_view = grid->leafGridView();
    typedef FvProductSpace< GridViewType, RangeFieldType, dimRange > FVSpaceType;
    std::cout << "Creating FiniteVolumeSpace..." << std::endl;
    const FVSpaceType fv_space(grid_view);

    // allocate a discrete function for the concentration
    std::cout << "Allocating discrete function..." << std::endl;
    typedef DiscreteFunction< FVSpaceType, Dune::Stuff::LA::CommonDenseVector< RangeFieldType > > FVFunctionType;
    FVFunctionType u(fv_space, "solution");

    //project initial values
    std::cout << "Projecting initial values..." << std::endl;
    project(*initial_values, u);

    //calculate dx and choose t_end and initial dt
    std::cout << "Calculating dx..." << std::endl;
    Dune::Stuff::Grid::Dimensions< GridViewType > dimensions(fv_space.grid_view());
    const double dx = dimensions.entity_width.max();
    const double CFL = problem.CFL();
    double dt = CFL*dx; //dx/4.0;
    const double t_end = 3.2;

    //define operator types
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 > ConstantFunctionType;
    typedef typename Dune::GDT::AdvectionGodunovOperator< AnalyticalFluxType, BoundaryValueType > OperatorType;
    typedef typename Dune::GDT::AdvectionRHSOperator< RHSType > RHSOperatorType;
    typedef typename Dune::GDT::ExplicitRungeKuttaTimeStepper<OperatorType, FVFunctionType,
        double, TimeStepperMethods::explicit_euler> FluxTimeStepperType;
    typedef typename Dune::GDT::ExplicitRungeKuttaTimeStepper<RHSOperatorType, FVFunctionType,
        double, TimeStepperMethods::explicit_euler> RHSTimeStepperType;
    typedef typename Dune::GDT::FractionalTimeStepper<FluxTimeStepperType, RHSTimeStepperType> TimeStepperType;

    const size_t num_save_steps = -1;

    //create Operators
    ConstantFunctionType dx_function(dx);
    OperatorType advection_operator(*analytical_flux, *boundary_values, true);
    RHSOperatorType rhs_operator(*rhs);

    //create timestepper
    FluxTimeStepperType flux_timestepper(advection_operator, u, -1.0);
    RHSTimeStepperType rhs_timestepper(rhs_operator, u);
    TimeStepperType timestepper(flux_timestepper, rhs_timestepper);

    // solve five times to average timings
//    for (size_t run = 0; run < 5; ++run) {
    // now do the time steps

//    boost::timer::cpu_timer timer;
    DSC_PROFILER.startTiming("fv.solve");
    timestepper.solve(t_end, dt, num_save_steps, true, true, false, DSC_CONFIG.get<std::string>("global.datadir") + "boltzmann");
    DSC_PROFILER.stopTiming("fv.solve");
//    const auto duration = timer.elapsed();
//    std::cout << "took: " << duration.wall*1e-9 << " seconds(" << duration.user*1e-9 << ", " << duration.system*1e-9 << ")" << std::endl;
    std::cout << "took: " << DSC_PROFILER.getTiming("fv.solve")/1000.0 << std::endl;
//    DSC_PROFILER.nextRun();

      // write timings to file
//      const bool file_already_exists = boost::filesystem::exists("weak_scaling_versuch_2.csv");
//      std::ofstream output_file("weak_scaling_versuch_2.csv", std::ios_base::app);
//      if (!file_already_exists) { // write header
//      output_file << "Problem: " + problem.static_id()
//                  << ", dimRange = " << dimRange
// //                  << ", number of grid cells: " << grid_config["num_elements"]
//                  << ", dt = " << DSC::toString(dt)
//                  << std::endl;
//      output_file << "num_threads, num_grid_cells, wall, user, system" << std::endl;
//      }
//      output_file << num_threads << ", " << grid_config["num_elements"] << ", " << duration.wall*1e-9 << ", " << duration.user*1e-9 << ", " << duration.system*1e-9 << std::endl;
//      output_file.close();

      // visualize solution
      timestepper.visualize_factor_of_solution< 0 >(DSC_CONFIG.get<std::string>("global.datadir") + "boltzmann");
//    }
//    for (size_t ii = 0; ii < solution_as_discrete_function.size(); ++ii) {
//      auto& pair = solution_as_discrete_function[ii];
//      pair.second.template visualize_factor< 0 >(ProblemType::short_id() + "_factor_0_" + DSC::toString(ii), true);
//    }
//}
      // write solution to *.csv file
//      write_solution_to_csv(grid_view, timestepper.solution(), problem.short_id() + "_P" + DSC::toString(dimRange - 1) + "_n" + DSC::toString(1.0/dx) + "_CFL" + DSC::toString(CFL) + "_CGLegendre_fractional_exact.csv");

//    // compute L1 error norm
//    SolutionType difference(*solution1);
//    for (size_t ii = 0; ii < difference.size(); ++ii) {
//      assert(DSC::FloatCmp::eq(difference[ii].first, solution2->operator[](ii).first) && "Time steps must be the same");
//      difference[ii].second.vector() = difference[ii].second.vector() - solution2->operator[](ii).second.vector();
//    }
//    std::cout << "error: " << DSC::toString(compute_L1_norm(grid_view, difference)) << std::endl;

//    mem_usage();
    DSC_PROFILER.setOutputdir(output_dir);
    DSC_PROFILER.outputTimings("timings_twobeams");
    std::cout << " done" << std::endl;
    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)


