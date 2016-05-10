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

DSC::FieldMatrix< double, 7, 7 > create_random_sigma_s(const double lower_bound, const double upper_bound)
{
typedef std::mt19937 RandomNumberGeneratorType;  // the Mersenne Twister with a popular choice of parameters
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
typedef std::mt19937 RandomNumberGeneratorType;  // the Mersenne Twister with a popular choice of parameters
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
  static const size_t dimDomain = 2;
  // for dimRange > 250, an "exceeded maximum recursive template instantiation limit" error occurs (tested with
  // clang 3.5). You need to pass -ftemplate-depth=N with N > dimRange + 10 to clang for higher dimRange.
  // for Boltzmann2D, this is not dimRange but the maximal moment order
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
  typedef typename Dune::GDT::AdvectionLaxFriedrichsOperator< AnalyticalFluxType, BoundaryValueType, ConstantFunctionType > OperatorType;
  typedef typename Dune::GDT::AdvectionRHSOperator< RHSType > RHSOperatorType;
  typedef typename Dune::GDT::ExplicitRungeKuttaTimeStepper<OperatorType, DiscreteFunctionType,
      double, TimeStepperMethods::explicit_euler> FluxTimeStepperType;
  typedef typename Dune::GDT::ExplicitRungeKuttaTimeStepper<RHSOperatorType, DiscreteFunctionType,
      double > RHSTimeStepperType;
  typedef typename Dune::GDT::FractionalTimeStepper<FluxTimeStepperType, RHSTimeStepperType> TimeStepperType;
  typedef typename TimeStepperType::SolutionType SolutionType;
  typedef std::vector< VectorType > SolutionVectorsVectorType;

  BoltzmannSolver(const size_t num_threads, const std::string output_dir, const size_t num_save_steps,
                  const size_t grid_size, const bool visualize_solution, const bool silent,
                  const std::string sigma_s_in, const std::string sigma_t_in)
  {
    auto num_save_steps_copy = num_save_steps;
    if (num_save_steps > 1e6)
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

    init(num_threads, output_dir, num_save_steps_copy, grid_size, visualize_solution, true, problem_config);
    silent_ = silent;
  }

  BoltzmannSolver(const size_t num_threads = 1, const std::string output_dir = "boltzmann", const size_t num_save_steps = 10,
                  const size_t grid_size = 50, const bool visualize_solution = true, const bool silent = false,
                  const RangeFieldType sigma_s_scattering = 1, const RangeFieldType sigma_s_absorbing = 0,
                  const RangeFieldType sigma_t_scattering = 1, const RangeFieldType sigma_t_absorbing = 10)
  {
    auto num_save_steps_copy = num_save_steps;
    if (num_save_steps > 1e6)
        num_save_steps_copy = size_t(-1);
    const auto problem_config = ProblemType::default_config(sigma_s_scattering, sigma_s_absorbing, sigma_t_scattering, sigma_t_absorbing);
    init(num_threads, output_dir, num_save_steps_copy, grid_size, visualize_solution, true, problem_config);
    silent_ = silent;
  }

  SolutionVectorsVectorType solve()
  {
    if (!silent_)
      std::cout << "Solving... " <<  std::endl;
    DSC_PROFILER.startTiming("fv.solve");
    timestepper_->solve(t_end_, dt_, num_save_steps_, true, !silent_, false, file_path_);
    DSC_PROFILER.stopTiming("fv.solve");
    if (!silent_)
      std::cout << "Solving took: " << DSC_PROFILER.getTiming("fv.solve")/1000.0 << "s" << std::endl;
    // visualize solution
    if (!silent_)
      std::cout << "Visualizing... ";
    if (visualize_solution_)
      timestepper_->visualize_factor_of_solution< 0 >(file_path_);
    if (!silent_)
      std::cout << " done" << std::endl;
    std::vector< VectorType > ret;
    for (const auto& pair : timestepper_->solution())
      ret.push_back(pair.second.vector());
    return ret;
  }

  SolutionVectorsVectorType next_n_time_steps(const size_t n)
  {
    if (!silent_)
      std::cout << "Calculating next " << DSC::toString(n) << " time steps... " <<  std::endl;
    DSC_PROFILER.startTiming("fv.solve");
    SolutionType solution;
    timestepper_->next_n_steps(n, t_end_, dt_, !silent_, solution);
    DSC_PROFILER.stopTiming("fv.solve");
    if (!silent_)
      std::cout << "Solving took: " << DSC_PROFILER.getTiming("fv.solve")/1000.0 << "s" << std::endl;
    // visualize solution
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
    advection_operator_->apply(source_function, range_function, time);
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

  void init(const size_t num_threads = 1, const std::string output_dir = "boltzmann", const size_t num_save_steps = 10,
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
    DSC_CONFIG.set("threading.partition_factor", 1, true);
    DS::threadManager().set_max_threads(num_threads);
    DSC_CONFIG.set("threading.max_count", DSC::toString(num_threads), true);
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
    advection_operator_ = std::make_shared< OperatorType >(*analytical_flux_, *boundary_values_, *dx_function_, dt_, true, false);
    rhs_operator_ = std::make_shared< RHSOperatorType >(*rhs_);

    //create timestepper
    flux_timestepper_ = std::make_shared< FluxTimeStepperType >(*advection_operator_, *u_, -1.0);
    rhs_timestepper_ = std::make_shared< RHSTimeStepperType >(*rhs_operator_, *u_);
    timestepper_ = std::make_shared< TimeStepperType >(*flux_timestepper_, *rhs_timestepper_);
  } // void init()

  void reset()
  {
    u_ = std::make_shared< DiscreteFunctionType >(*fv_space_, "solution");
    project(*initial_values_, *u_);
    flux_timestepper_ = std::make_shared< FluxTimeStepperType >(*advection_operator_, *u_, -1.0);
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
  std::shared_ptr< OperatorType > advection_operator_;
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


int main(int argc, char* argv[])
{
  try {
    // parse options
    if (argc == 1)
      std::cout << "Usage: " << argv[0] << "[-threading.max_count THREADS -global.datadir DIR -num_save_steps NUM -gridsize GRIDSIZE -no_visualization -silent -sigma_s SIGMA_S_MATRIX -sigma_t SIGMA_T_MATRIX]" << std::endl;

    size_t num_threads = 1;
    size_t num_save_steps = 10;
    size_t grid_size = 50;
    bool visualize = true;
    bool silent = false;
    std::string output_dir, sigma_s, sigma_t;
    double sigma_s_lower = 0, sigma_s_upper = 1, sigma_a_lower = 0, sigma_a_upper = 10;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "-threading.max_count") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          num_threads = DSC::fromString< size_t >(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
        } else {
          std::cerr << "-threading.max_count option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-global.datadir") {
        if (i + 1 < argc) {
          output_dir = argv[++i];
        } else {
          std::cerr << "-global.datadir option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-num_save_steps") {
        if (i + 1 < argc) {
          num_save_steps = DSC::fromString< size_t >(argv[++i]);
        } else {
          std::cerr << "-num_save_steps option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-no_visualization") {
        visualize = false;
        i++;
      } else if (std::string(argv[i]) == "-silent") {
        silent = true;
        i++;
      } else if (std::string(argv[i]) == "-num_save_steps") {
        if (i + 1 < argc) {
          num_save_steps = DSC::fromString< size_t >(argv[++i]);
        } else {
          std::cerr << "-num_save_steps option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-gridsize") {
        if (i + 1 < argc) {
          grid_size = DSC::fromString< size_t >(argv[++i]);
        } else {
          std::cerr << "-gridsize option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-sigma_s") {
        if (i + 1 < argc) {
          sigma_s = argv[++i];
        } else {
          std::cerr << "-sigma_s option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-sigma_t") {
        if (i + 1 < argc) {
          sigma_t = argv[++i];
        } else {
          std::cerr << "-sigma_t option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-sigma_s_lower") {
        if (i + 1 < argc) {
          sigma_s_lower = DSC::fromString<double>(argv[++i]);
        } else {
            std::cerr << "-sigma_s_lower option requires one argument." << std::endl;
            return 1;
          }
      } else if (std::string(argv[i]) == "-sigma_s_upper") {
        if (i + 1 < argc) {
          sigma_s_upper = DSC::fromString<double>(argv[++i]);
        } else {
            std::cerr << "-sigma_s_upper option requires one argument." << std::endl;
            return 1;
          }
      } else if (std::string(argv[i]) == "-sigma_a_lower") {
        if (i + 1 < argc) {
          sigma_a_lower = DSC::fromString<double>(argv[++i]);
        } else {
            std::cerr << "-sigma_a_lower option requires one argument." << std::endl;
            return 1;
          }
      } else if (std::string(argv[i]) == "-sigma_a_upper") {
        if (i + 1 < argc) {
          sigma_a_upper = DSC::fromString<double>(argv[++i]);
        } else {
            std::cerr << "-sigma_a_upper option requires one argument." << std::endl;
            return 1;
          }
      } else {
        std::cerr << "Unknown option " << std::string(argv[i]) << std::endl;
        return 1;
      }
    }

    BoltzmannSolver solver(num_threads, output_dir, 5, grid_size, true, false);
    solver.solve();

    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

//------------------------------------
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

//template< class Traits, class ScalarImp = typename Traits::ScalarType >
//class VectorInterface
//  : public ContainerInterface< Traits, ScalarImp >
//  , public Tags::VectorInterface
//{
//public:
//  typedef typename Traits::derived_type                       derived_type;
//  typedef typename Dune::FieldTraits< ScalarImp >::field_type ScalarType;
//  typedef typename Dune::FieldTraits< ScalarImp >::real_type  RealType;
//  static const constexpr ChooseBackend                        dense_matrix_type  = Traits::dense_matrix_type;
//  static const constexpr ChooseBackend                        sparse_matrix_type = Traits::sparse_matrix_type;

//  typedef internal::VectorInputIterator< Traits, ScalarType >  const_iterator;
//  typedef internal::VectorOutputIterator< Traits, ScalarType > iterator;

//  inline ScalarType& get_entry_ref(const size_t ii)

//  inline const ScalarType& get_entry_ref(const size_t ii) const

//  inline ScalarType& operator[](const size_t ii)

//  inline const ScalarType& operator[](const size_t ii) const

//  virtual bool almost_equal(const derived_type& other,
//                            const RealType epsilon = Stuff::Common::FloatCmp::DefaultEpsilon< RealType >::value()) const

//  template< class T >
//  bool almost_equal(const VectorInterface< T >& other,
//                    const RealType epsilon = Stuff::Common::FloatCmp::DefaultEpsilon< RealType >::value()) const

//  virtual derived_type operator*(const ScalarType& alpha) const

//  virtual ScalarType operator*(const derived_type& other) const

//  virtual derived_type& operator+=(const ScalarType& scalar)

//  virtual derived_type& operator-=(const ScalarType& scalar)

//  virtual derived_type& operator/=(const ScalarType& scalar)

//  virtual bool operator==(const derived_type& other) const

//  virtual bool operator!=(const derived_type& other) const


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(init_overloads, BoltzmannSolver::init, 0, 7)

BOOST_PYTHON_MODULE(libboltzmann)
{
  class_<BoltzmannSolver>("BoltzmannSolver", init< optional< const size_t, const std::string, const size_t, const size_t, const bool, const bool, const double, const double, const double, const double > >())
       .def(init< const size_t, const std::string, const size_t, const size_t, const bool, const bool, const std::string, const std::string >())
       .def("init", &BoltzmannSolver::init, init_overloads())
       .def("solve", &BoltzmannSolver::solve)
       .def("next_n_time_steps", &BoltzmannSolver::next_n_time_steps)
       .def("reset", &BoltzmannSolver::reset)
       .def("finished", &BoltzmannSolver::finished)
       .def("apply_LF_operator", &BoltzmannSolver::apply_LF_operator)
       .def("apply_rhs_operator", &BoltzmannSolver::apply_rhs_operator)
       .def("get_initial_values", &BoltzmannSolver::get_initial_values)
      ;

  class_<typename BoltzmannSolver::SolutionVectorsVectorType>("SolutionVectorsVectorType")
       .def(vector_indexing_suite<typename BoltzmannSolver::SolutionVectorsVectorType>())
       .def("size", &BoltzmannSolver::SolutionVectorsVectorType::size)
      ;

  VectorExporter< typename Dune::Stuff::LA::CommonDenseVector< double > >::export_("CommonDenseVector");
}




