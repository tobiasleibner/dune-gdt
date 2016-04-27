# This file is part of the dune-gdt project:
#   https://github.com/dune-community/dune-gdt
# Copyright 2010-2016 dune-gdt developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Felix Schindler (2010, 2013 - 2016)

add_subdirectory(test EXCLUDE_FROM_ALL)

set(lib_boltzmann_sources test/boltzmanntest.cc)

add_library("boltzmann" SHARED ${lib_boltzmann_sources})
target_link_libraries(boltzmann ${DUNE_LIBS})
add_dune_mpi_flags(boltzmann)
if(TBB_FOUND)
  add_dune_tbb_flags(boltzmann)
endif(TBB_FOUND)

if(dune-grid_FOUND)
  add_dune_superlu_flags(boltzmann)
  add_dune_alugrid_flags(boltzmann)
endif(dune-grid_FOUND)
target_link_dune_default_libraries(boltzmann)

find_package(Boost COMPONENTS python)
find_package(PythonLibs 2.7)
include_directories (${PYTHON_INCLUDE_DIRS})
include_directories (${Boost_INCLUDE_DIRS})
target_link_libraries(boltzmann boost_python ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
