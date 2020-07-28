#include "Omega_h_mesh.hpp"
#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pseudoXGCmTypes.hpp"

o::Mesh readMesh(char* meshFile, o::Library& lib) {
  const auto rank = lib.world()->rank();
  (void)lib;
  std::string fn(meshFile);
  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if( ext == "msh") {
    if(!rank)
      std::cout << "reading gmsh mesh " << meshFile << "\n";
    return o::gmsh::read(meshFile, lib.self());
  } else if( ext == "osh" ) {
    if(!rank)
      std::cout << "reading omegah mesh " << meshFile << "\n";
    return o::binary::read(meshFile, lib.self(), true);
  } else {
    if(!rank)
      std::cout << "error: unrecognized mesh extension \'" << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  pumipic::Library pic_lib(&argc, &argv);
  o::Library& lib = pic_lib.omega_h_lib();
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  auto full_mesh = readMesh(argv[1], lib);

  MPI_Barrier(MPI_COMM_WORLD);

  if(!rank)
    printf("Mesh loaded with <v e f> %d %d %d\n",full_mesh.nverts(),full_mesh.nedges(),
        full_mesh.nfaces());

  const auto bufferMethod = pumipic::Input::getMethod(argv[3]);
  const auto safeMethod = pumipic::Input::getMethod(argv[4]);
  assert(bufferMethod>=0);
  assert(safeMethod>=0);
  p::Input input(full_mesh, argv[2], bufferMethod, safeMethod);
  if(!rank)
    input.printInfo();
  MPI_Barrier(MPI_COMM_WORLD);
  p::Mesh picparts(input);
  o::Mesh* mesh = picparts.mesh();

  auto matches = mesh->get_array<o::LO>(0, "matches");

  o::Write<o::LO> field = picparts.createCommArray(0, 1, 0);
  picparts.reduceCommArray(0, pumipic::Mesh::SUM_OP, field);

  auto vert_globals = picparts.globalIds(0);
  //printf("vert_glosize=%d, nverts in mesh=%d\n", vert_globals.size(), full_mesh.nverts());
  auto max_globalVert = o::get_max(vert_globals);
  auto vert_owners = picparts.entOwners(0);
  //printf("vert_ownersN=%d, nverts in part=%d\n", vert_owners.size(), mesh->nverts());
  printf("matches size is %d, owners size is %d\n", matches.size(), vert_owners.size());

  if (rank == 1) {
    auto print_owners = OMEGA_H_LAMBDA(o::LO i) {
      printf("owner is %d\n", vert_owners[i]);
      //printf("owner of vert with gID %d is %d\n", vert_globals[i], vert_owners[i]);
    };
    o::parallel_for(vert_owners.size(), print_owners);
  }

  /*Implementing algo 1 to collect matchOwner
  auto matchowners = picparts.matchOwner(0);
  LOs picparts.matchOwner(Int edim) {
    assert has_tag(matches);
    assert matches.size = nents(edim)
    .
    .
    .
  } 
  */

  /* Pseudo field-sync in serial */
  if (size == 1) {
    auto pseudo_sync = OMEGA_H_LAMBDA(o::LO i) {
      if (matches[i] > 0) field[i] = field[matches[i]];
    };
    o::parallel_for(field.size(), pseudo_sync);
  }

  o::vtk::write_parallel("/lore/joshia5/develop/pumi-pic/periodic_data/periodicZ_5k_3d_tuto_4.vtk", mesh);
}
