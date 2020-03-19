set(TEST_DATA_DIR "" CACHE PATH
  "Path to a local copy of the pumipic-data repo.")
if(NOT EXISTS ${TEST_DATA_DIR})
  message(FATAL_ERROR "TEST_DATA_DIR \"${TEST_DATA_DIR}\" is not accessible")
endif()
#get the mpirun binary/script
get_filename_component(COMPILER_DIR "${CMAKE_CXX_COMPILER}" PATH)
find_program(MPIRUN NAMES mpirun PATHS "${COMPILER_DIR}")
set(MPIRUN_PROCFLAG "-np" CACHE STRING
  "the command line flag to give process count to MPIRUN")

function(make_test exename srcname)
  add_executable(${exename} ${srcname})
  target_link_libraries(${exename} pumipic)
endfunction(make_test)

make_test(barycentric test_barycentric.cpp)
make_test(linetri_intersection test_linetri_intersection.cpp)
make_test(test_dist2bdry_preprocess test_dist2Bdry_preprocess.cpp)
make_test(test_gitrm_ps test_gitrm_ps.cpp)
make_test(test_interp2d test_interpolate2d.cpp)
include(testing.cmake)

bob_end_subdir()
