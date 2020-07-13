#include <Omega_h_mesh.hpp>
#include <particle_structs.hpp>
#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_profiling.hpp"
#include "pseudoXGCmTypes.hpp"
#include <fstream>
#include <random>

o::Mesh readMesh(char* meshFile, o::Library& lib) {
  const auto rank = lib.world()->rank();
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    if(!rank)
      std::cout << "reading gmsh mesh " << meshFile << "\n";
    return Omega_h::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    if(!rank)
      std::cout << "reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self(), true);
  } else {
    if(!rank)
      std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  printf("ok1\n");
  auto full_mesh = readMesh(argv[1], lib);

  MPI_Barrier(MPI_COMM_WORLD);
  printf("ok2\n");

  if(!comm_rank)
    printf("Mesh loaded with <v e f> %d %d %d\n",full_mesh.nverts(),full_mesh.nedges(),
        full_mesh.nfaces());

  const auto vtx_to_elm = full_mesh.ask_up(0, 2);
  const auto edge_to_elm = full_mesh.ask_up(1, 2);
  printf("ok3\n");
  const auto bufferMethod = pumipic::Input::getMethod(argv[3]);
  const auto safeMethod = pumipic::Input::getMethod(argv[4]);
  assert(bufferMethod>=0);
  assert(safeMethod>=0);
  printf("ok4\n");
  p::Input input(full_mesh, argv[2], bufferMethod, safeMethod);
  printf("ok5\n");
  if(!comm_rank)
    input.printInfo();
  MPI_Barrier(MPI_COMM_WORLD);
  p::Mesh picparts(input);
  printf("ok6\n");
  o::Mesh* mesh = picparts.mesh();
  mesh->ask_elem_verts();

  auto match_partTag = mesh->get_array<o::LO>(0, "matches");
  o::Write<o::LO> field = picparts.createCommArray(0, 1, 0);

  picparts.reduceCommArray(0, pumipic::Mesh::SUM_OP, field);

  /* Pseudo field-sync in serial */
  auto pseudo_sync = OMEGA_H_LAMBDA(o::LO i) {
    if (match_partTag[i] > 0) field[i] = field[match_partTag[i]];
  };
  o::parallel_for(field.size(), pseudo_sync);

  if (!comm_rank)
    fprintf(stderr, "done\n");
  return 0;
}
