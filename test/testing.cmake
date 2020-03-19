function(mpi_test TESTNAME PROCS EXE)
  add_test(
    NAME ${TESTNAME}
    COMMAND ${MPIRUN} ${MPIRUN_PROCFLAG} ${PROCS} ${VALGRIND} ${VALGRIND_ARGS} ${EXE} ${ARGN}
  )
endfunction(mpi_test)

mpi_test(barycentric_3 1  ./barycentric test1)

mpi_test(barycentric_4 1  ./barycentric test2)

mpi_test(linetri_intersection_2 1
  ./linetri_intersection  0.0,1.0,0.0:0.5,0.0,0.0:1.0,1.0,0.0  0.5,0.6,-2  0.5,0.6,2 )

mpi_test(test_interp2d 1 ./test_interp2d --kokkos-threads=1)
mpi_test(test_dist2bdry_preprocess 1  ./test_dist2ddry_preprocess --kokkos-threads=1)

