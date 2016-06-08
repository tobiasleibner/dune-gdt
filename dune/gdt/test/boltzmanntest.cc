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
#include <random>

#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>
#include <boost/python.hpp>

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

typedef std::mt19937 RandomNumberGeneratorType;

DSC::FieldMatrix< double, 7, 7 > create_random_sigma_s(const double lower_bound, const double upper_bound)
{
static RandomNumberGeneratorType rng{std::random_device()()};
std::uniform_real_distribution<double> distribution(lower_bound,upper_bound);
DSC::FieldMatrix< double, 7, 7 > ret;
for (size_t ii = 0; ii < 7; ++ii) {
  for (size_t jj = 0; jj < 7; ++jj) {
    ret[ii][jj] = distribution(rng);
  }
}
return ret;
}

DSC::FieldMatrix< double, 7, 7 > create_random_sigma_t(const double lower_bound, const double upper_bound, const DSC::FieldMatrix< double, 7, 7 >& sigma_s)
{
static RandomNumberGeneratorType rng{std::random_device()()};
std::uniform_real_distribution<double> distribution(lower_bound, upper_bound);
DSC::FieldMatrix< double, 7, 7 > ret;
for (size_t ii = 0; ii < 7; ++ii) {
  for (size_t jj = 0; jj < 7; ++jj) {
    ret[ii][jj] = distribution(rng) + sigma_s[ii][jj];
  }
}
return ret;
}


class BoltzmannSolver
{
public:
  // set dimensions
  // for dimRange > 250, an "exceeded maximum recursive template instantiation limit" error occurs (tested with
  // clang 3.5). You need to pass -ftemplate-depth=N with N > dimRange + 10 to clang for higher dimRange.
  static const size_t dimDomain = 2;
  static const size_t momentOrder = 15;

  typedef Dune::YaspGrid< dimDomain >  GridType;
  typedef typename GridType::Codim< 0 >::Entity                                         EntityType;
  typedef Dune::GDT::Hyperbolic::Problems::Boltzmann2DCheckerboard< EntityType, double, dimDomain, double, momentOrder > ProblemType;
  static const size_t dimRange = ProblemType::dimRange;
  typedef Dune::Stuff::Common::Configuration ConfigType;
  typedef typename ProblemType::FluxType              AnalyticalFluxType;
  typedef typename ProblemType::RHSType               RHSType;
  typedef typename ProblemType::InitialValueType      InitialValueType;
  typedef typename ProblemType::BoundaryValueType     BoundaryValueType;
  typedef typename InitialValueType::DomainFieldType      DomainFieldType;
  typedef typename ProblemType::RangeFieldType        RangeFieldType;
  typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
  typedef typename GridType::LeafGridView                GridViewType;
  typedef FvProductSpace< GridViewType, RangeFieldType, dimRange > FVSpaceType;
  typedef typename Dune::Stuff::LA::CommonDenseVector< RangeFieldType > VectorType;
  typedef DiscreteFunction< FVSpaceType, VectorType > DiscreteFunctionType;
  typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 > ConstantFunctionType;
  typedef typename Dune::GDT::AdvectionLaxFriedrichsOperator< AnalyticalFluxType, BoundaryValueType, ConstantFunctionType > LaxFriedrichsOperatorType;
  typedef typename Dune::GDT::AdvectionGodunovOperator< AnalyticalFluxType, BoundaryValueType > GodunovOperatorType;
  typedef typename Dune::GDT::AdvectionRHSOperator< RHSType > RHSOperatorType;
  typedef typename Dune::GDT::ExplicitRungeKuttaTimeStepper<LaxFriedrichsOperatorType, DiscreteFunctionType,
      double, TimeStepperMethods::explicit_euler> FluxTimeStepperType;
  typedef typename Dune::GDT::ExplicitRungeKuttaTimeStepper<RHSOperatorType, DiscreteFunctionType,
      double > RHSTimeStepperType;
  typedef typename Dune::GDT::FractionalTimeStepper<FluxTimeStepperType, RHSTimeStepperType> TimeStepperType;
  typedef typename TimeStepperType::SolutionType SolutionType;
  typedef std::vector< VectorType > SolutionVectorsVectorType;

  BoltzmannSolver(const std::string output_dir, const size_t num_save_steps,
                  const size_t grid_size, const bool visualize_solution, const bool silent,
                  const std::string sigma_s_in, const std::string sigma_t_in)
  {
    auto num_save_steps_copy = num_save_steps;
    if (num_save_steps > 1e6) // hack to allow for size_t(-1) when called from the python bindings
        num_save_steps_copy = size_t(-1);
    auto random_sigma_s = create_random_sigma_s(0.0,10.0);
    auto random_sigma_t = create_random_sigma_t(0.0,10.0, random_sigma_s);
    auto sigma_s = sigma_s_in.empty() ? DSC::toString(random_sigma_s) : sigma_s_in;
    auto sigma_t = sigma_t_in.empty() ? DSC::toString(random_sigma_t) : sigma_t_in;

    //create problem configuration
    ConfigType checkerboard_config;
    checkerboard_config["sigma_s"] = sigma_s;
    checkerboard_config["sigma_t"] = sigma_t;
    const auto problem_config = ProblemType::default_config(checkerboard_config);

    init(output_dir, num_save_steps_copy, grid_size, visualize_solution, true, problem_config);
    silent_ = silent;
  }

  BoltzmannSolver(const std::string output_dir = "boltzmann", const size_t num_save_steps = 10,
                  const size_t grid_size = 50, const bool visualize_solution = true, const bool silent = false,
                  const RangeFieldType sigma_s_scattering = 1, const RangeFieldType sigma_s_absorbing = 0,
                  const RangeFieldType sigma_a_scattering = 0, const RangeFieldType sigma_a_absorbing = 10)
  {
    auto num_save_steps_copy = num_save_steps;
    if (num_save_steps > 1e6) // hack to allow for size_t(-1) when called from the python bindings
        num_save_steps_copy = size_t(-1);
    const auto problem_config = ProblemType::default_config(sigma_s_scattering, sigma_s_absorbing, sigma_s_scattering + sigma_a_scattering, sigma_s_absorbing + sigma_a_absorbing);
    init(output_dir, num_save_steps_copy, grid_size, visualize_solution, true, problem_config);
    silent_ = silent;
  }

  double current_time()
  {
    return timestepper_->current_time();
  }

  double t_end()
  {
    return 3.2;
  }

  void set_current_time(const double time)
  {
    timestepper_->current_time() = time;
  }

  void set_current_solution(const VectorType& vec)
  {
    timestepper_->current_solution().vector() = vec;
    rhs_timestepper_->current_solution().vector() = vec;
    flux_timestepper_->current_solution().vector() = vec;
  }

  double time_step_length()
  {
    return dt_;
  }

  SolutionVectorsVectorType solve(const bool with_half_steps = false)
  {
    if (!silent_)
      std::cout << "Solving... " <<  std::endl;
    DSC_PROFILER.startTiming("fv.solve");
    timestepper_->solve(t_end_, dt_, num_save_steps_, true, !silent_, false, with_half_steps, file_path_);
    DSC_PROFILER.stopTiming("fv.solve");
    if (!silent_)
      std::cout << "Solving took: " << DSC_PROFILER.getTiming("fv.solve")/1000.0 << "s" << std::endl;
    if (visualize_solution_) {
      if (!silent_)
        std::cout << "Visualizing... ";
      timestepper_->visualize_factor_of_solution< 0 >(file_path_);
      if (!silent_)
        std::cout << " done" << std::endl;
    }
    std::vector< VectorType > ret;
    for (const auto& pair : timestepper_->solution())
      ret.push_back(pair.second.vector());
    return ret;
  }

  SolutionVectorsVectorType next_n_time_steps(const size_t n, const bool with_half_steps = false)
  {
    if (!silent_)
      std::cout << "Calculating next " << DSC::toString(n) << " time steps... " <<  std::endl;
    DSC_PROFILER.startTiming("fv.solve");
    SolutionType solution;
    timestepper_->next_n_steps(n, t_end_, dt_, !silent_, with_half_steps, solution);
    DSC_PROFILER.stopTiming("fv.solve");
    if (!silent_)
      std::cout << "Solving took: " << DSC_PROFILER.getTiming("fv.solve")/1000.0 << "s" << std::endl;
    std::vector< VectorType > ret;
    for (const auto& pair : solution)
      ret.push_back(pair.second.vector());
    return ret;
  }

  VectorType apply_LF_operator(VectorType source, const double time)
  {
    const DiscreteFunctionType source_function(*fv_space_, source);
    VectorType ret(source);
    DiscreteFunctionType range_function(*fv_space_, ret);
    laxfriedrichs_operator_->apply(source_function, range_function, time);
    return ret;
  }

  VectorType apply_godunov_operator(VectorType source, const double time)
  {
    const DiscreteFunctionType source_function(*fv_space_, source);
    VectorType ret(source);
    DiscreteFunctionType range_function(*fv_space_, ret);
    godunov_operator_->apply(source_function, range_function, time);
    return ret;
  }

  VectorType apply_rhs_operator(VectorType source, const double time)
  {
    const DiscreteFunctionType source_function(*fv_space_, source);
    VectorType ret(source);
    DiscreteFunctionType range_function(*fv_space_, ret);
    rhs_operator_->apply(source_function, range_function, time);
    return ret;
  }

  VectorType apply_rhs_operator(VectorType source, const double time, const RangeFieldType sigma_s_scattering, const RangeFieldType sigma_s_absorbing = 0,
                                const RangeFieldType sigma_a_scattering = 1, const RangeFieldType sigma_a_absorbing = 10)
  {
    set_rhs_operator_parameters(sigma_s_scattering, sigma_s_absorbing, sigma_a_scattering, sigma_a_absorbing);
    return apply_rhs_operator(source, time);
  }

  void set_rhs_operator_parameters(const RangeFieldType sigma_s_scattering = 1, const RangeFieldType sigma_s_absorbing = 0,
                                   const RangeFieldType sigma_a_scattering = 0, const RangeFieldType sigma_a_absorbing = 10)
  {
    const auto problem_config = ProblemType::default_config(sigma_s_scattering, sigma_s_absorbing, sigma_s_scattering + sigma_a_scattering, sigma_s_absorbing + sigma_a_absorbing);
    const auto problem_ptr = ProblemType::create(problem_config);
    const auto& problem = *problem_ptr;
    rhs_ = problem.rhs();
    rhs_operator_ = std::make_shared< RHSOperatorType >(*rhs_);
    rhs_timestepper_->set_operator(*rhs_operator_);
  }

  VectorType get_initial_values()
  {
    DiscreteFunctionType ret(*fv_space_, "discrete_initial_values");
    project(*initial_values_, ret);
    return ret.vector();
  }

  bool finished()
  {
    return DSC::FloatCmp::eq(timestepper_->current_time(), 3.2);
  }

  void init(const std::string output_dir = "boltzmann", const size_t num_save_steps = 10,
            const size_t grid_size = 50, const bool visualize_solution = true, const bool silent = false,
            const ConfigType problem_config = ProblemType::default_config())
  {
#if HAVE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
      MPI_Init(NULL, NULL);
#endif
    silent_ = silent;
    if (!silent_)
      std::cout << "Setting problem parameters ...";
    visualize_solution_ = visualize_solution;
    file_path_ = output_dir;
    num_save_steps_ = num_save_steps;
    // setup threadmanager
    DSC_CONFIG.set("global.datadir", output_dir, true);
    DSC_PROFILER.setOutputdir(output_dir);
    //choose GridType

    const auto problem_ptr = ProblemType::create(problem_config);
    const auto& problem = *problem_ptr;

    //get grid configuration from problem
    auto grid_config = problem.grid_config();
    grid_config["num_elements"] = "[" + DSC::toString(grid_size);
    for (size_t ii = 1; ii < dimDomain; ++ii)
        grid_config["num_elements"] += " " + DSC::toString(grid_size);
    grid_config["num_elements"] += "]";

    //get analytical flux, initial and boundary values
    analytical_flux_ = problem.flux();
    initial_values_ = problem.initial_values();
    boundary_values_ = problem.boundary_values();
    rhs_ = problem.rhs();

    //create grid
    GridProviderType grid_provider = *(GridProviderType::create(grid_config));
    grid_ = grid_provider.grid_ptr();

    // make a product finite volume space on the leaf grid
    grid_view_ = std::make_shared< GridViewType >(grid_->leafGridView());
    fv_space_ = std::make_shared< FVSpaceType >(*grid_view_);

    // allocate a discrete function for the concentration
    u_ = std::make_shared< DiscreteFunctionType >(*fv_space_, "solution");

    //project initial values
    if (!silent_) {
      std::cout << " done " << std::endl;
      std::cout << "Projecting initial values...";
    }
    project(*initial_values_, *u_);
    if (!silent_)
      std::cout << " done " << std::endl;

    //calculate dx and choose t_end and initial dt
    Dune::Stuff::Grid::Dimensions< GridViewType > dimensions(*grid_view_);
    const double dx = dimensions.entity_width.max()/std::sqrt(2);
    const double CFL = problem.CFL();
    dt_ = CFL*dx;
    t_end_ = problem.t_end();

    //create Operators
    dx_function_ = std::make_shared< ConstantFunctionType>(dx);
    laxfriedrichs_operator_ = std::make_shared< LaxFriedrichsOperatorType >(*analytical_flux_, *boundary_values_, *dx_function_, dt_, true, false);
    godunov_operator_ = std::make_shared< GodunovOperatorType >(*analytical_flux_, *boundary_values_, *dx_function_, dt_, true, false);
    rhs_operator_ = std::make_shared< RHSOperatorType >(*rhs_);

    //create timestepper
    flux_timestepper_ = std::make_shared< FluxTimeStepperType >(*laxfriedrichs_operator_, *u_, -1.0);
    rhs_timestepper_ = std::make_shared< RHSTimeStepperType >(*rhs_operator_, *u_);
    timestepper_ = std::make_shared< TimeStepperType >(*flux_timestepper_, *rhs_timestepper_);
  } // void init()

  void reset()
  {
    u_ = std::make_shared< DiscreteFunctionType >(*fv_space_, "solution");
    project(*initial_values_, *u_);
    flux_timestepper_ = std::make_shared< FluxTimeStepperType >(*laxfriedrichs_operator_, *u_, -1.0);
    rhs_timestepper_ = std::make_shared< RHSTimeStepperType >(*rhs_operator_, *u_);
    timestepper_ = std::make_shared< TimeStepperType >(*flux_timestepper_, *rhs_timestepper_);
  }

private:
  std::shared_ptr< const GridType > grid_;
  std::shared_ptr< const GridViewType > grid_view_;
  std::shared_ptr< const FVSpaceType > fv_space_;
  std::shared_ptr< DiscreteFunctionType > u_;
  std::shared_ptr< const AnalyticalFluxType > analytical_flux_;
  std::shared_ptr< const InitialValueType > initial_values_;
  std::shared_ptr< const BoundaryValueType > boundary_values_;
  std::shared_ptr< const RHSType > rhs_;
  std::shared_ptr< LaxFriedrichsOperatorType > laxfriedrichs_operator_;
  std::shared_ptr< GodunovOperatorType > godunov_operator_;
  std::shared_ptr< RHSOperatorType > rhs_operator_;
  std::shared_ptr< FluxTimeStepperType > flux_timestepper_;
  std::shared_ptr< RHSTimeStepperType > rhs_timestepper_;
  std::shared_ptr< TimeStepperType > timestepper_;
  std::shared_ptr< ConstantFunctionType > dx_function_;
  double t_end_;
  double dt_;
  bool silent_;
  bool visualize_solution_;
  std::string file_path_;
  size_t num_save_steps_;
};


// Python bindings
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

namespace {

  // Converts a std::pair instance to a Python tuple.
  template <typename T1, typename T2>
  struct std_pair_to_tuple
  {
    static PyObject* convert(std::pair<T1, T2> const& p)
    {
      return boost::python::incref(
        boost::python::make_tuple(p.first, p.second).ptr());
    }
    static PyTypeObject const *get_pytype () {return &PyTuple_Type; }
  };

  // Helper for convenience.
  template <typename T1, typename T2>
  struct std_pair_to_python_converter
  {
    std_pair_to_python_converter()
    {
      boost::python::to_python_converter<
        std::pair<T1, T2>,
        std_pair_to_tuple<T1, T2>,
        true //std_pair_to_tuple has get_pytype
        >();
    }
  };

} // namespace anonymous


template <class Vec>
struct VectorExporter
{
  typedef typename Vec::ScalarType ScalarType;
  typedef typename Dune::Stuff::LA::VectorInterface< typename Vec::Traits, ScalarType > VectorInterfaceType;
  typedef typename VectorInterfaceType::derived_type derived_type;
  typedef typename Vec::RealType RealType;

  static object buffer(Vec& self)
  {
    PyObject* py_buf = PyBuffer_FromReadWriteMemory(&self[0], self.size() * sizeof(ScalarType));
    object retval = object(handle<>(py_buf));
    return retval;
  }

  static void export_(const std::string& classname)
  {
    std_pair_to_python_converter<size_t, RealType>();

    void (Vec::*sub_void)(const derived_type&, derived_type&) const = &Vec::sub;
    derived_type (Vec::*sub_vec)(const derived_type&) const = &Vec::sub;

    void (Vec::*add_void)(const derived_type&, derived_type&) const = &Vec::add;
    derived_type (Vec::*add_vec)(const derived_type&) const = &Vec::add;

    class_< Vec >(classname.c_str())
	.def(init<const size_t, const ScalarType>())
        .def("size", &Vec::size)
        .def("add_to_entry", &Vec::add_to_entry)
        .def("__setitem__", &Vec::set_entry)
        .def("__getitem__", &Vec::get_entry)
        .def("l1_norm", &Vec::l1_norm)
        .def("l2_norm", &Vec::l2_norm)
        .def("sup_norm", &Vec::sup_norm)
        .def("standard_deviation", &Vec::standard_deviation)
        .def("set_all", &Vec::set_all)
        .def("valid", &Vec::valid)
        .def("dim", &Vec::dim)
        .def("mean", &Vec::mean)
        .def("amax", &Vec::amax)
        .def("sub", sub_void)
        .def("sub",sub_vec)
        .def("add", add_void)
        .def("add", add_vec)
        .def("__add__", add_vec)
        .def("__sub__", sub_vec)
        .def("__iadd__", &Vec::iadd)
        .def("__isub__", &Vec::isub)
        .def("dot", &Vec::dot)
        .def("__mul__", &Vec::dot)
        .def("buffer", &buffer)
        .def("scal", &Vec::scal)
        .def("axpy", &Vec::axpy)
        .def("copy", &Vec::copy)
        ;
  }
};


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(init_overloads, BoltzmannSolver::init, 0, 6)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(apply_rhs_overloads, BoltzmannSolver::apply_rhs_operator, 3, 6)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(next_n_time_steps_overloads, BoltzmannSolver::next_n_time_steps, 1, 2)

BOOST_PYTHON_MODULE(libboltzmann)
{
  typedef typename BoltzmannSolver::VectorType VectorType;
  typedef typename BoltzmannSolver::RangeFieldType RangeFieldType;
  VectorType (BoltzmannSolver::*apply_rhs_without_params)(VectorType, const double) = &BoltzmannSolver::apply_rhs_operator;
  VectorType (BoltzmannSolver::*apply_rhs_with_params)(VectorType, const double, const RangeFieldType,
                                                          const RangeFieldType, const RangeFieldType,
                                                          const RangeFieldType) = &BoltzmannSolver::apply_rhs_operator;

  class_<BoltzmannSolver>("BoltzmannSolver",
                          init< optional< const std::string, const size_t, const size_t, const bool,
                                          const bool, const double, const double, const double, const double > >())
       .def(init< const std::string, const size_t, const size_t, const bool, const bool,
                  const std::string, const std::string >())
       .def("init", &BoltzmannSolver::init, init_overloads())
       .def("solve", &BoltzmannSolver::solve)
       .def("next_n_time_steps", &BoltzmannSolver::next_n_time_steps, next_n_time_steps_overloads())
       .def("reset", &BoltzmannSolver::reset)
       .def("finished", &BoltzmannSolver::finished)
       .def("apply_LF_operator", &BoltzmannSolver::apply_LF_operator)
       .def("apply_godunov_operator", &BoltzmannSolver::apply_godunov_operator)
       .def("apply_rhs_operator", apply_rhs_without_params)
       .def("apply_rhs_operator", apply_rhs_with_params, apply_rhs_overloads())
       .def("set_rhs_operator_parameters", &BoltzmannSolver::set_rhs_operator_parameters)
       .def("get_initial_values", &BoltzmannSolver::get_initial_values)
       .def("current_time", &BoltzmannSolver::current_time)
       .def("set_current_time", &BoltzmannSolver::set_current_time)
       .def("set_current_solution", &BoltzmannSolver::set_current_solution)
       .def("time_step_length", &BoltzmannSolver::time_step_length)
       .def("t_end", &BoltzmannSolver::t_end)
      ;

  class_<typename BoltzmannSolver::SolutionVectorsVectorType>("SolutionVectorsVectorType")
       .def(vector_indexing_suite<typename BoltzmannSolver::SolutionVectorsVectorType>())
       .def("size", &BoltzmannSolver::SolutionVectorsVectorType::size)
      ;

  VectorExporter< typename Dune::Stuff::LA::CommonDenseVector< double > >::export_("CommonDenseVector");
}


int main(int argc, char* argv[])
{
  try {
    // parse options
    if (argc == 1)
      std::cout << "The following options are available: "
                << argv[0]
                << " [-output_dir DIR -num_save_steps INT -gridsize INT "
                << "  -sigma_s_1 FLOAT -sigma_s_2 FLOAT -sigma_a_1 FLOAT -sigma_a_2 FLOAT"
                << " --no_visualization --silent --random_parameters --totally_random_parameters]"
                << std::endl;

    size_t num_save_steps = 10;
    size_t grid_size = 50;
    bool visualize = true;
    bool silent = false;
    bool random_parameters = false;
    bool totally_random_parameters = false;
    bool parameters_given = false;
    std::string output_dir;
    double sigma_s_lower = 0, sigma_s_upper = 8, sigma_a_lower = 0, sigma_a_upper = 8;
    double sigma_s_1 = 1, sigma_s_2 = 0, sigma_a_1 = 0, sigma_a_2 = 10;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "-output_dir") {
        if (i + 1 < argc) {
          output_dir = argv[++i];
        } else {
          std::cerr << "-output_dir option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-num_save_steps") {
        if (i + 1 < argc) {
          num_save_steps = DSC::fromString< size_t >(argv[++i]);
        } else {
          std::cerr << "-num_save_steps option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "--no_visualization") {
        visualize = false;
      } else if (std::string(argv[i]) == "--silent") {
        silent = true;
      } else if (std::string(argv[i]) == "--random_parameters") {
        if (totally_random_parameters) {
          std::cerr << "Options --random_parameters and --totally-random_parameters are not compatible!" << std::endl;
          return 1;
        }
        if (parameters_given) {
          std::cerr << "You specified a value for at least one parameter so you can't use --random_parameters!"
                    << std::endl;
          return 1;
        }
        random_parameters = true;
        RandomNumberGeneratorType rng{std::random_device()()};
        std::uniform_real_distribution<double> sigma_s_dist(sigma_s_lower, sigma_s_upper);
        std::uniform_real_distribution<double> sigma_a_dist(sigma_a_lower, sigma_a_upper);
        sigma_s_1 = sigma_s_dist(rng);
        sigma_s_2 = sigma_s_dist(rng);
        sigma_a_1 = sigma_a_dist(rng);
        sigma_a_2 = sigma_a_dist(rng);
      } else if (std::string(argv[i]) == "--totally_random_parameters") {
        if (random_parameters) {
          std::cerr << "Options --random_parameters and --totally-random_parameters are not compatible!" << std::endl;
          return 1;
        }
        if (parameters_given) {
          std::cerr << "You specified a value for at least one parameter so you can't use --totally_random_parameters!"
                    << std::endl;
          return 1;
        }
        totally_random_parameters = true;
      } else if (std::string(argv[i]) == "-gridsize") {
        if (i + 1 < argc) {
          grid_size = DSC::fromString< size_t >(argv[++i]);
        } else {
          std::cerr << "-gridsize option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-sigma_s_1") {
        if (random_parameters || totally_random_parameters) {
          std::cerr << "You specified a value for at least one parameter on the command line so you can't use "
                    << "--random_parameters or --totally_random_parameters!" << std::endl;
          return 1;
        }
        if (i + 1 < argc) {
          sigma_s_1 = DSC::fromString< double >(argv[++i]);
          parameters_given = true;
        } else {
          std::cerr << "-sigma_s_1 option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-sigma_s_2") {
        if (random_parameters || totally_random_parameters) {
          std::cerr << "You specified a value for at least one parameter on the command line so you can't use "
                    << "--random_parameters or --totally_random_parameters!" << std::endl;
          return 1;
        }
        if (i + 1 < argc) {
          sigma_s_2 = DSC::fromString< double >(argv[++i]);
          parameters_given = true;
        } else {
          std::cerr << "-sigma_s_2 option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-sigma_a_1") {
        if (random_parameters || totally_random_parameters) {
          std::cerr << "You specified a value for at least one parameter on the command line so you can't use "
                    << "--random_parameters or --totally_random_parameters!" << std::endl;
          return 1;
        }
        if (i + 1 < argc) {
          sigma_a_1 = DSC::fromString< double >(argv[++i]);
          parameters_given = true;
        } else {
          std::cerr << "-sigma_a_1 option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-sigma_a_2") {
        if (random_parameters || totally_random_parameters) {
          std::cerr << "You specified a value for at least one parameter on the command line so you can't use "
                    << "--random_parameters or --totally_random_parameters!" << std::endl;
          return 1;
        }
        if (i + 1 < argc) {
          sigma_a_2 = DSC::fromString< double >(argv[++i]);
          parameters_given = true;
        } else {
          std::cerr << "-sigma_a_2 option requires one argument." << std::endl;
          return 1;
        }
      } else {
        std::cerr << "Unknown option " << std::string(argv[i]) << std::endl;
        return 1;
      }
    }

    std::ofstream parameterfile;
    parameterfile.open(output_dir + "_parameters.txt");
    parameterfile << "Gridsize: " << DSC::toString(grid_size) + " x " + DSC::toString(grid_size) << std::endl;

    // run solver
    std::shared_ptr< BoltzmannSolver > solver;
    if (totally_random_parameters) {
      const auto sigma_s_matrix = create_random_sigma_s(sigma_s_lower, sigma_s_upper);
      const auto sigma_t_matrix = create_random_sigma_t(sigma_a_lower, sigma_a_upper, sigma_s_matrix);
      auto sigma_a_matrix = sigma_t_matrix;
      sigma_a_matrix -= sigma_s_matrix;
      parameterfile << "Random parameters chosen on each square of the 7x7 checkerboard domain were: " << std::endl
                    << "sigma_s: " << DSC::toString(sigma_s_matrix) << std::endl
                    << "sigma_a: " << DSC::toString(sigma_a_matrix) << std::endl;
      solver = std::make_shared< BoltzmannSolver >(output_dir, num_save_steps, grid_size, visualize,
                                                   silent, DSC::toString(sigma_s_matrix), DSC::toString(sigma_t_matrix));
    } else {
      parameterfile << "Domain was composed of two materials, parameters were: " << std::endl
                    << "First material: sigma_s = " + DSC::toString(sigma_s_1)
                       + ", sigma_a = " + DSC::toString(sigma_a_1) << std::endl
                    << "Second material: sigma_s = " + DSC::toString(sigma_s_2)
                       + ", sigma_a = " + DSC::toString(sigma_a_2) << std::endl;

      solver = std::make_shared< BoltzmannSolver >(output_dir, num_save_steps, grid_size, visualize,
                                                   silent, sigma_s_1, sigma_s_2, sigma_a_1, sigma_a_2);
    }

    DSC_PROFILER.startTiming("solve_all");
    solver->solve();
    DSC_PROFILER.stopTiming("solve_all");
    parameterfile << "Elapsed time: " << DSC_PROFILER.getTiming("solve_all")/1000.0 << " s" << std::endl;
    parameterfile.close();

    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)





