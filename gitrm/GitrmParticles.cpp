#include <fstream>
#include <cstdlib>
#include <vector>
#include <set>
#include <Omega_h_int_scan.hpp>
#include "GitrmMesh.hpp"
#include "GitrmParticles.hpp"
#include "GitrmInputOutput.hpp"

int iTimePlusOne = 0;

namespace gitrm {
o::Reals getConstEField() {
  o::HostWrite<o::Real> ef(3);
  ef[0] = CONSTANT_EFIELD0;
  ef[1] = CONSTANT_EFIELD1;
  ef[2] = CONSTANT_EFIELD2;
  return o::Reals(ef.write());
}

bool checkIfRankZero() {
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  return (comm_rank==0);
}
}//ns 

GitrmParticles::GitrmParticles(p::Mesh& picparts, long int nPtcls, int nIter, double dT):
  ptcls(nullptr), picparts(picparts), mesh(*picparts.mesh()) {
  //move to where input is handled
  totalPtcls = nPtcls;
  numIterations = nIter;
  timeStep = dT;
  initPtclDetectionData();
}

GitrmParticles::~GitrmParticles() {
  delete ptcls;
}

void GitrmParticles::resetPtclWallCollisionData() {
  auto capacity = ptcls->capacity();
  wallCollisionPts_w = o::Write<o::Real>(3*capacity, 0, "wallXpts");
  wallCollisionFaceIds_w = o::Write<o::LO>(capacity, -1, "wallXfids");
}

void GitrmParticles::initPtclWallCollisionData() {
  resetPtclWallCollisionData();
}

void GitrmParticles::convertPtclWallCollisionData() {
  wallCollisionPts = o::Reals(wallCollisionPts_w);
  wallCollisionFaceIds = o::LOs(wallCollisionFaceIds_w);
}

void GitrmParticles::defineParticles(const o::LOs& ptclsInElem, int elId) {
  o::Int ne = mesh.nelems();
  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const o::LO& i) {
    element_gids(i) = mesh_element_gids[i];
  });
  if(elId>=0) {
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const o::LO& i) {
      ptcls_per_elem(i) = 0;
      if (i == elId) {
        ptcls_per_elem(i) = numInitPtcls;
        printf(" Ptcls in elId %d\n", elId);
      }
    });
  } else {
    Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const o::LO& i) {
      ptcls_per_elem(i) = ptclsInElem[i];
    });
  }

  printf(" ptcls/elem: \n");
  Omega_h::parallel_for(ne, OMEGA_H_LAMBDA(const o::LO& i) {
    const int np = ptcls_per_elem(i);
    if (np > 0)
      printf("%d , ", np);
  });
  printf("\n");
  //'sigma', 'V', and the 'policy' control the layout of the PS structure
  const int sigma = 1; //INT_MAX; // full sorting
  const int V = 128;//1024;
  Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(10000, 32);
  printf("Constructing Particles\n");
  //Create the particle structure
  ptcls = new SellCSigma<Particle>(policy, sigma, V, ne, numInitPtcls,
                   ptcls_per_elem, element_gids);
}

// TODO update,  this works only for full-buffer
void GitrmParticles::assignParticles(const o::Reals& data, const o::LOs& elemIdOfPtclsAll, 
   o::LOs& numPtclsInElems, o::LOs& elemIdOfPtcls, o::LOs& ptclDataInds) {
  int comm_size;
  int comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  OMEGA_H_CHECK(totalPtcls > 0);
  int rest = totalPtcls%comm_size;
  numInitPtcls = totalPtcls / comm_size;
  //last one gets the rest !
  if(comm_rank == comm_size-1)
    numInitPtcls += rest;
  auto pBegin = comm_rank* totalPtcls/comm_size;
 
  //verify
  const long int totalPtcls_ = totalPtcls;
  const long int numInitPtcls_ = numInitPtcls;
  long int totalPtcls = 0;
  MPI_Reduce(&numInitPtcls_, &totalPtcls, 1, MPI_LONG, MPI_SUM, 0,
    MPI_COMM_WORLD);
  if(!comm_rank)
    printf("Particles per rank %ld total %ld \n", numInitPtcls, totalPtcls);
  if(totalPtcls_)
    OMEGA_H_CHECK(totalPtcls_ == totalPtcls);

  o::Write<o::LO> elemIdOfPtcls_w(numInitPtcls, -1);
  o::Write<o::LO> ptclDataInds_w(numInitPtcls, -1);
  o::Write<o::LO> numPtclsInElems_w(mesh.nelems(), 0);
  o::LO offset = pBegin;  
  auto lambda = OMEGA_H_LAMBDA(const o::LO& i) {
    auto el = elemIdOfPtclsAll[offset+i];//only for equal distribution
    elemIdOfPtcls_w[i] = el;
    ptclDataInds_w[i] = offset+i;
    Kokkos::atomic_increment(&numPtclsInElems_w[el]);
  };
  o::parallel_for(numInitPtcls, lambda, "assign_init_ptcls");
  elemIdOfPtcls = o::LOs(elemIdOfPtcls_w);
  ptclDataInds = o::LOs(ptclDataInds_w);
  numPtclsInElems = o::LOs(numPtclsInElems_w);
}


void GitrmParticles::initPtclsFromFile(const std::string& fName,
   o::LO maxLoops, bool printSource) {
  std::cout << "Loading initial particle data from file: " << fName << " \n";
  o::HostWrite<o::Real> readInData_h;
  // TODO piscesLowFlux/updated/input/particleSource.cfg has r,z,angles, CDF, cylSymm=1
  //read total particles by all picparts, since parent element not known

  //Reading all simulating particles in all ranks
  auto stat = readParticleSourceNcFile(fName, readInData_h, numPtclsRead, true);
  OMEGA_H_CHECK( stat && (numPtclsRead >= totalPtcls));  
  o::Reals readInData_r(readInData_h.write());
  o::LOs elemIdOfPtcls;
  o::LOs ptclDataInds;
  o::LOs numPtclsInElems;
  std::cout << "find ElemIds of Ptcls \n";
  findElemIdsOfPtclCoordsByAdjSearch(readInData_r, elemIdOfPtcls, ptclDataInds, 
    numPtclsInElems);

  printf("Constructing PS particles\n");
  defineParticles(numPtclsInElems, -1);
  initPtclWallCollisionData();
  //note:rebuild to get mask if elem_ids changed
  printf("Setting Ptcl InitCoords \n");
  auto ptclIdPtrsOfElem = o::offset_scan(o::LOs(numPtclsInElems));
  auto totalPtcls = o::get_sum(o::LOs(numPtclsInElems));
  OMEGA_H_CHECK(numInitPtcls == totalPtcls);

  printf("converting  to CSR\n");
  o::LOs ptclIdsInElem;
  convertInitPtclElemIdsToCSR(numPtclsInElems, ptclIdPtrsOfElem, ptclIdsInElem,
   elemIdOfPtcls);
  printf("setting ptcl Ids \n");
  setPidsOfPtclsLoadedFromFile(ptclIdPtrsOfElem, ptclIdsInElem, elemIdOfPtcls, 
    ptclDataInds);
  printf("setting ptcl init data \n");
  setPtclInitData(readInData_r);
  printf("setting ionization recombination init data \n");
  initPtclChargeIoniRecombData();
  initPtclSurfaceModelData();
  //if(printSource)
  //  printPtclSource(readInData_r, numInitPtcls, 6); //nptcl=0(all), dof=6
}

bool GitrmParticles::searchPtclInAllElems(const o::Reals& data, const o::LO pind, 
   o::LO& parentElem) {
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MESHDATA(mesh);
  auto nPtclsRead = numPtclsRead;
  o::Write<o::LO> elemDet(1, -1);
  auto lamb = OMEGA_H_LAMBDA(const o::LO& elem) {
    auto pos = o::zero_vector<3>();
    auto bcc = o::zero_vector<4>();
    for(int j=0; j<3; ++j)
      pos[j] = data[j*nPtclsRead+pind];
    if(p::isPointWithinElemTet(mesh2verts, coords, pos, elem, bcc)) {
      auto prev = Kokkos::atomic_exchange(&(elemDet[0]), elem);
      if(prev >= 0) 
        printf("InitParticle %d in rank %d was found in multiple elements %d %d \n",
          pind, comm_rank, prev, elem);
    }
  };
  o::parallel_for(nel, lamb, "search_parent_of_ptcl");
  Kokkos::fence();
  parentElem = (o::HostRead<o::LO>(elemDet))[0];
  return parentElem >= 0;
}

o::LO GitrmParticles::searchAllPtclsInAllElems(const o::Reals& data, 
   o::Write<o::LO>& elemIdOfPtcls, o::Write<o::LO>& numPtclsInElems) {
  MESHDATA(mesh);
  auto nPtclsRead = numPtclsRead;
  auto totPtcls = totalPtcls;
  auto lamb = OMEGA_H_LAMBDA(const o::LO& elem) {
    auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
    auto tet = p::gatherVectors4x3(coords, tetv2v);
    auto pos = o::zero_vector<3>();
    for(auto pind=0; pind< totPtcls; ++pind) {
      for(int j=0; j<3; ++j)
        pos[j] = data[j*nPtclsRead+pind];
      auto bcc = o::zero_vector<4>();
      p::find_barycentric_tet(tet, pos, bcc);
      if(p::all_positive(bcc, 1.0e-20)) {
        Kokkos::atomic_increment(&numPtclsInElems[elem]); 
        Kokkos::atomic_exchange(&(elemIdOfPtcls[pind]), elem);
      }
    }
  };
  o::parallel_for(nel, lamb, "search_parents_of_ptcls");
  return o::get_min(o::LOs(elemIdOfPtcls));
}

o::LO GitrmParticles::searchPtclsByAdjSearchFromParent(const o::Reals& data, 
   const o::LO parentElem, o::Write<o::LO>& numPtclsInElemsAll,
   o::Write<o::LO>& elemIdOfPtclsAll) {
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MESHDATA(mesh);
  auto nPtclsRead = numPtclsRead;
  int maxSearch = 1000;
  //search all particles starting with this element
  auto lambda = OMEGA_H_LAMBDA(const o::LO& ip) {
    bool found = false;
    auto pos = o::zero_vector<3>();
    auto bcc = o::zero_vector<4>();
    o::LO elem = parentElem;
    int isearch=0;
    while(!found) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto tet = p::gatherVectors4x3(coords, tetv2v);
      for(int j=0; j<3; ++j)
        pos[j] = data[j*nPtclsRead+ip];
      p::find_barycentric_tet(tet, pos, bcc);
      if(p::all_positive(bcc, 0)) {
        Kokkos::atomic_exchange(&(elemIdOfPtclsAll[ip]), elem);
        //not useful
        Kokkos::atomic_increment(&numPtclsInElemsAll[elem]);
        found = true;
      } else {
        o::LO minInd = p::min_index(bcc, 4);
        auto dual_elem_id = dual_faces[elem];
        o::LO findex = 0;
        for(auto iface = elem*4; iface < (elem+1)*4; ++iface) {
          auto face_id = down_r2fs[iface];
          bool exposed = side_is_exposed[face_id];
          if(!exposed) {
            if(findex == minInd)
              elem = dual_elems[dual_elem_id];
            ++dual_elem_id;
          }
          ++findex;
        }//for
      }
      if(isearch > maxSearch)
        break;
      ++isearch;
    }
  };
  o::parallel_for(totalPtcls, lambda, "init_impurity_ptcl2");
  return o::get_min(o::LOs(elemIdOfPtclsAll));
}


// search all particles read. TODO update this for partitioning
void GitrmParticles::findElemIdsOfPtclCoordsByAdjSearch(const o::Reals& data, 
   o::LOs& elemIdOfPtcls, o::LOs& ptclDataInds, o::LOs& numPtclsInElems) {
  bool debug = true;
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  //try one in first few particles
  o::LO maxLoop = 100;
  MESHDATA(mesh);
  o::Write<o::LO> elemDet(1, -1);
  o::LO parentElem = -1;
  o::LO pstart = 0;
  o::LO ptcl = -1;
  for(auto ip = pstart; ip < maxLoop; ++ip) {
    searchPtclInAllElems(data, ip, parentElem);
    if(parentElem >= 0) {
      ptcl = ip;
      break;
    }
  }
  if(debug && parentElem >=0)
   printf("rank %d : Beginning Ptcl search: ptcl %d found in elem %d\n", 
    comm_rank, ptcl, parentElem);
  if(debug && parentElem < 0)
   printf("rank %d : Parent elements not found for first %d particles\n", maxLoop);

  //find all particles starting from parent
  o::Write<o::LO> numPtclsInElemsAll(nel, 0);
  o::Write<o::LO> elemIdOfPtclsAll(totalPtcls, -1);
  o::LO min = -1;
  if(parentElem >= 0)
    min = searchPtclsByAdjSearchFromParent(data, parentElem, numPtclsInElemsAll, 
      elemIdOfPtclsAll);

  //if not all found, do brute-force search for all
  if(min < 0)
    min = searchAllPtclsInAllElems(data, numPtclsInElemsAll, elemIdOfPtclsAll);

  if(debug && min < 0) {
    o::parallel_for(totalPtcls, OMEGA_H_LAMBDA(const o::LO& i) {
      if(elemIdOfPtclsAll[i] < 0) {
        double v[6];
        for(int j=0; j<6; ++j)
          v[j] = data[j*numPtclsRead+i];      
        printf("rank %d : NOTdet i %d %g %g %g :vel: %g %g %g\n", 
          comm_rank, i, v[0], v[1], v[2], v[3], v[4], v[5] );
      }
    });
  }
  //works only for full-buffer
  //OMEGA_H_CHECK(min >=0);
  assignParticles(data, o::LOs(elemIdOfPtclsAll), numPtclsInElems, elemIdOfPtcls, 
    ptclDataInds);
}


// using ptcl sequential numbers 0..numPtcls
void GitrmParticles::convertInitPtclElemIdsToCSR(const o::LOs& numPtclsInElems,
   o::LOs& ptclIdPtrsOfElem, o::LOs& ptclIdsInElem, o::LOs& elemIdOfPtcls) {
  o::LO debug = 0;
  auto nel = mesh.nelems();
  // csr data
  o::Write<o::LO> ptclIdsInElem_w(numInitPtcls, -1);
  o::Write<o::LO> ptclsFilledInElem(nel, 0); 
  auto lambda = OMEGA_H_LAMBDA(const o::LO& id) {
    auto el = elemIdOfPtcls[id];
    auto old = Kokkos::atomic_fetch_add(&(ptclsFilledInElem[el]), 1);
    //TODO FIXME invalid device function error with OMEGA_H_CHECK in lambda
    auto nLimit = numPtclsInElems[el];
    OMEGA_H_CHECK(old < nLimit); 
    //elemId is sequential from 0 .. nel
    auto beg = ptclIdPtrsOfElem[el];
    auto pos = beg + old;
    auto idLimit = ptclIdPtrsOfElem[el+1];
    OMEGA_H_CHECK(pos < idLimit);
    auto prev = Kokkos::atomic_exchange(&(ptclIdsInElem_w[pos]), id);
    if(debug)
      printf("id:el %d %d old %d beg %d pos %d previd %d maxPtcls %d \n",
        id, el, old, beg, pos, prev, numPtclsInElems[el] );
  };
  o::parallel_for(numInitPtcls, lambda, "Convert to CSR write");
  ptclIdsInElem = o::LOs(ptclIdsInElem_w);   
}


void GitrmParticles::setPidsOfPtclsLoadedFromFile(const o::LOs& ptclIdPtrsOfElem,
  const o::LOs& ptclIdsInElem,  const o::LOs& elemIdOfPtcls, const o::LOs& ptclDataInds) {
  int debug = 1;
  if(debug) printf("numInitPtcls %d numPtclsRead %d \n", numInitPtcls, numPtclsRead);
  auto nInitPtcls = numInitPtcls;
  //TODO for full-buffer only
  int comm_size;
  int comm_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  auto pBegin = comm_rank* totalPtcls/comm_size;

  auto nel = mesh.nelems();
  o::Write<o::LO> nextPtclInd(nel, 0);
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      auto thisInd = Kokkos::atomic_fetch_add(&(nextPtclInd[elem]), 1);
      auto ind = ptclIdPtrsOfElem[elem] + thisInd;
      auto limit = ptclIdPtrsOfElem[elem+1];
      //Set checks separately to avoid possible Error 
      OMEGA_H_CHECK(ind >= 0);
      OMEGA_H_CHECK(ind < limit);
      const auto ip = ptclIdsInElem[ind] + pBegin; // TODO
      if(debug || (ind < 0 || ind>= limit || ip<0 || ip>= nInitPtcls))
        printf("**** elem %d pid %d ind %d thisInd %d nextInd %d indlim %d ip %d\n", 
          elem, pid, ind, thisInd, nextPtclInd[elem], ptclIdPtrsOfElem[elem+1], ip);
      OMEGA_H_CHECK(ip >= 0);
      //OMEGA_H_CHECK(ip < nInitPtcls);
      //OMEGA_H_CHECK(elemIdOfPtcls[ip] == elem);
      pid_ps(pid) = ip;
    }
  };
  ps::parallel_for(ptcls, lambda, "setPidsOfPtcls");
}

//To use ptcls, PS_LAMBDA is required, not parallel_for(#ptcls,OMEGA_H_LAMBDA
// since ptcls in each element is iterated in groups. Construct PS with 
// #particles in each elem passed in, otherwise newly added particles in 
// originally empty elements won't show up in PS_LAMBDA iterations. 
// ie. their mask will be 0. If mask is not used, invalid particles may 
// show up from other threads in the launch group.
void GitrmParticles::setPtclInitData(const o::Reals& data) {
  o::LO debug = 1;
  if(debug) printf("numPtclsRead %d \n", numPtclsRead);
  auto npRead = numPtclsRead;
  const auto coords = mesh.coords(); 
  const auto mesh2verts = mesh.ask_elem_verts(); 
  auto next_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto pos_ps_d = ptcls->get<PTCL_POS>();
  auto vel_d = ptcls->get<PTCL_VEL>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);
      auto bcc = o::zero_vector<4>();
      auto vel = o::zero_vector<3>();
      auto pos = o::zero_vector<3>();
      auto ip = pid_ps(pid); 
      for(int i=0; i<3; ++i){
        pos[i] = data[i*npRead+ip];
      }
      for(int i=0; i<3; ++i)
        vel[i] = data[(3+i)*npRead+ip];
      if(debug)
        printf("ip %d pos %g %g %g vel %g %g %g\n", ip, pos[0], pos[1], pos[2], 
          vel[0], vel[1], vel[2]);
      p::find_barycentric_tet(M, pos, bcc);
      OMEGA_H_CHECK(p::all_positive(bcc, 0));
      for(int i=0; i<3; i++) {
        pos_ps_d(pid,i) = pos[i];
        vel_d(pid, i) = vel[i];
        next_ps_d(pid,i) = 0;
      }
    }
  };
  ps::parallel_for(ptcls, lambda, "setPtclInitData");
}

void GitrmParticles::initPtclChargeIoniRecombData() {
  auto charge_ps = ptcls->get<PTCL_CHARGE>();
  auto first_ionizeZ_ps = ptcls->get<PTCL_FIRST_IONIZEZ>();
  auto prev_ionize_ps = ptcls->get<PTCL_PREV_IONIZE>();
  auto first_ionizeT_ps = ptcls->get<PTCL_FIRST_IONIZET>();
  auto prev_recomb_ps = ptcls->get<PTCL_PREV_RECOMBINE>();

  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask > 0) {
      charge_ps(pid) = 0;
      first_ionizeZ_ps(pid) = 0;
      prev_ionize_ps(pid) = 0;
      first_ionizeT_ps(pid) = 0;
      prev_recomb_ps(pid) = 0;
    }
  };
  ps::parallel_for(ptcls, lambda, "initPtclChargeIoniRecombData");
}

void GitrmParticles::initPtclSurfaceModelData() {
  auto ps_weight = ptcls->get<PTCL_WEIGHT>();
  auto ps_hitNum = ptcls->get<PTCL_HIT_NUM>();
  auto ps_newVelMag = ptcls->get<PTCL_VMAG_NEW>();
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask > 0) {
      ps_weight(pid) = 1;
      ps_hitNum(pid) = 0;
      ps_newVelMag(pid) = 0;
    }
  };
  ps::parallel_for(ptcls, lambda, "initPtclSurfaceModelData");
}

void GitrmParticles::initPtclsInADirection(o::Real theta, o::Real phi, o::Real r, 
   o::LO maxLoops, o::Real outer) {
  o::Write<o::LO> elemAndFace(3, -1); 
  o::LO initEl = -1;
  findInitialBdryElemIdInADir(theta, phi, r, initEl, elemAndFace, maxLoops, outer);
  o::LOs temp;
  defineParticles(temp, initEl);
  printf("Constructed Particles\n");

  //note:rebuild if particles to be added in new elems, or after emptying any elem.
  printf("\n Setting ImpurityPtcl InitCoords \n");
  setPtclInitRndDistribution(elemAndFace);
}

void GitrmParticles::setPtclInitRndDistribution(o::Write<o::LO> &elemAndFace) {
  MESHDATA(mesh);

  //Set particle coordinates. Initialized only on one face. TODO confirm this ? 
  auto x_ps_d = ptcls->get<PTCL_NEXT_POS>();
  auto x_ps_prev_d = ptcls->get<PTCL_POS>();
  auto vel_d = ptcls->get<PTCL_VEL>();
  int psCapacity = ptcls->capacity();
  o::HostWrite<o::Real> rnd1(psCapacity, 0);
  o::HostWrite<o::Real> rnd2(psCapacity, 0);
  o::HostWrite<o::Real> rnd3(psCapacity, 0);
  std::srand(time(NULL));
  for(auto i=0; i<psCapacity; ++i) {
    rnd1[i] = (o::Real)(std::rand())/RAND_MAX;
    rnd2[i] = (o::Real)(std::rand())/RAND_MAX;
    rnd3[i] = (o::Real)(std::rand())/RAND_MAX;
  }
  o::Reals rand1 = o::Reals(o::Write<o::Real>(rnd1));
  o::Reals rand2 = o::Reals(o::Write<o::Real>(rnd2));
  o::Reals rand3 = o::Reals(o::Write<o::Real>(rnd3));

  o::Write<o::LO> elem_ids(psCapacity,-1);
  auto lambda = PS_LAMBDA(const int &elem, const int &pid, const int &mask) {
    if(mask > 0) {
    //if(elemAndFace[1] >=0 && elem == elemAndFace[1]) {  
      o::LO verbose =1;
      // TODO if more than an element  ?
      const auto faceId = elemAndFace[2];
      const auto fv2v = o::gather_verts<3>(face_verts, faceId);
      const auto face = p::gatherVectors3x3(coords, fv2v);
      auto fcent = p::face_centroid_of_tet(faceId, coords, face_verts);
      auto tcent = p::centroid_of_tet(elem, mesh2verts, coords); 
      auto diff = tcent - fcent;
      if(verbose >3)
        printf(" elemAndFace[1]:%d, elem:%d face%d beg%d\n", 
          elemAndFace[1], elem, elemAndFace[2], elemAndFace[0]);

      o::Vector<4> bcc;
      o::Vector<3> pos;
      auto rn1 = rand1[pid];
      auto rn2 = rand2[pid];
      do { 
        o::Real bc1 = (rn1 > rn2) ? rn2: rn1;
        o::Real bc2 = std::abs(rn1 - rn2);
        o::Real bc3 = 1.0 - bc1 - bc2;
        o::Vector<3> fpos = bc1*face[0] + bc2*face[1] + bc3*face[2];
        auto fnorm = p::face_normal_of_tet(faceId, elem, coords, mesh2verts, 
                                        face_verts, down_r2fs);
        pos = fpos - 1.0e-6*fnorm;
        auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
        auto M = p::gatherVectors4x3(coords, tetv2v);
        p::find_barycentric_tet(M, pos, bcc);
        rn1 /= 2.0;
      } while(!p::all_positive(bcc, 0));

      double amu = 2.0; //TODO
      double energy[] = {4.0, 4, 4}; //TODO actual [4,0,0]
      double vel[] = {0,0,0};
      for(int i=0; i<3; i++) {
        x_ps_prev_d(pid,i) = pos[i];
        x_ps_d(pid,i) = pos[i];
        auto en = energy[i];   
        //if(! o::are_close(energy[i], 0))
        vel[i] = std::sqrt(2.0 * abs(en) * 1.60217662e-19 / (amu * 1.6737236e-27));
        vel[i] *= rand3[pid];  //TODO
      }

      for(int i=0; i<3; i++)
        vel_d(pid, i) = vel[i];
 
      elem_ids[pid] = elem;

      if(verbose >2)
        printf("elm %d : pos %.4f %.4f %.4f : vel %.1f %.1f %.1f Mask%d\n",
          elem, x_ps_prev_d(pid,0), x_ps_prev_d(pid,1), x_ps_prev_d(pid,2),
          vel[0], vel[1], vel[2], mask);
    }
  };
  ps::parallel_for(ptcls, lambda, "setPtclInitRndDistribution");
}

// spherical coordinates (wikipedia), radius r=1.5m, inclination theta[0,pi] from the z dir,
// azimuth angle phi[0, 2π) from the Cartesian x-axis (so that the y-axis has phi = +90°).
void GitrmParticles::findInitialBdryElemIdInADir(o::Real theta, o::Real phi, o::Real r,
     o::LO &initEl, o::Write<o::LO> &elemAndFace, o::LO maxLoops, o::Real outer){

  o::LO debug = 4;
  MESHDATA(mesh);

  theta = theta * o::PI / 180.0;
  phi = phi * o::PI / 180.0;
  
  const o::Real x = r * sin(theta) * cos(phi);
  const o::Real y = r * sin(theta) * sin(phi);
  const o::Real z = r * cos(theta);

  o::Real endR = r + outer; //meter, to be outside of the domain
  const o::Real xe = endR * sin(theta) * cos(phi);
  const o::Real ye = endR * sin(theta) * sin(phi);
  const o::Real ze = endR * cos(theta);

  printf("\nDirection:x,y,z: %f %f %f\n xe,ye,ze: %f %f %f\n", x,y,z, xe,ye,ze);

  // Beginning element id of this x,y,z
  auto lamb = OMEGA_H_LAMBDA(const o::LO& elem) {
    auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
    auto M = p::gatherVectors4x3(coords, tetv2v);

    o::Vector<3> orig;
    orig[0] = x;
    orig[1] = y;
    orig[2] = z;
    o::Vector<4> bcc;
    p::find_barycentric_tet(M, orig, bcc);
    if(p::all_positive(bcc, 0)) {
      elemAndFace[0] = elem;
      if(debug > 3)
        printf(" ORIGIN detected in elem %d \n", elem);
    }
  };
  o::parallel_for(mesh.nelems(), lamb, "init_impurity_ptcl1");
  Kokkos::fence();
  o::HostRead<o::LO> elemId_bh(elemAndFace);
  printf(" ELEM_beg %d \n", elemId_bh[0]);
  
  if(elemId_bh[0] < 0) {
    Omega_h_fail("Failed finding initial element in given direction\n");
  }

  // Search final elemAndFace on bdry, on 1 thread on device(issue [] on host) 
  o::Write<o::Real> xpt(3, -1); 
  auto lamb2 = OMEGA_H_LAMBDA(const o::LO& e) {
    auto elem = elemAndFace[0];
    o::Vector<3> dest;
    dest[0] = xe;
    dest[1] = ye;
    dest[2] = ze;
    o::Vector<3> orig;
    orig[0] = x;
    orig[1] = y;
    orig[2] = z;    
    o::Vector<4> bcc;
    bool found = false;
    o::LO loops = 0;

    while (!found) {

      if(debug > 4)
        printf("\n****ELEM %d : ", elem);

      // Destination should be outisde domain
      auto tetv2v = o::gather_verts<4>(mesh2verts, elem);
      auto M = p::gatherVectors4x3(coords, tetv2v);

      p::find_barycentric_tet(M, dest, bcc);
      if(p::all_positive(bcc, 0)) {
        printf("Wrong guess of destination in initImpurityPtcls");
        OMEGA_H_CHECK(false);
      }

      // Start search
      auto dual_elem_id = dual_faces[elem];
      const auto beg_face = elem *4;
      const auto end_face = beg_face +4;
      o::LO fIndex = 0;

      for(auto iface = beg_face; iface < end_face; ++iface) {
        const auto face_id = down_r2fs[iface];

        o::Vector<3> xpoint = o::zero_vector<3>();
        const auto face = p::get_face_coords_of_tet(mesh2verts, coords, elem, fIndex);
        o::Real dproj = 0;
        bool detected = p::line_triangle_intx_simple(face, orig, dest, xpoint, dproj);
        if(debug > 4) {
          printf("iface %d faceid %d detected %d\n", iface, face_id, detected);             
        }

        if(detected && side_is_exposed[face_id]) {
          found = true;
          elemAndFace[1] = elem;
          elemAndFace[2] = face_id;

          for(o::LO i=0; i<3; ++i)
            xpt[i] = xpoint[i];

          if(debug) {
            printf(" faceid %d detected on exposed\n",  face_id);
          }
          break;
        } else if(detected && !side_is_exposed[face_id]) {
          auto adj_elem  = dual_elems[dual_elem_id];
          elem = adj_elem;
          if(debug >4) {
            printf(" faceid %d detected on interior; next elm %d\n", face_id, elem);
          }
          break;
        }
        if(!side_is_exposed[face_id]){
          ++dual_elem_id;
        }
        ++fIndex;
      } // faces

      if(loops > maxLoops) {
          printf("Tried maxLoops iterations in initImpurityPtcls");
          OMEGA_H_CHECK(false);
      }
      ++loops;
    }
  };
  o::parallel_for(1, lamb2, "init_impurity_ptcl2");
  Kokkos::fence();
  o::HostRead<o::Real> xpt_h(xpt);

  o::HostRead<o::LO> elemId_fh(elemAndFace);
  initEl = elemId_fh[1];
  printf(" ELEM_final %d xpt: %.3f %.3f %.3f\n\n", elemId_fh[1], xpt_h[0], xpt_h[1], xpt_h[2]);
  OMEGA_H_CHECK((initEl>=0) && (elemId_fh[0]>=0));
}

// Read GITR particle step data of all time steps; eg: rand numbers.
int GitrmParticles::readGITRPtclStepDataNcFile(const std::string& ncFileName,
  int& maxNPtcls, bool debug) {
  assert(USE_GITR_RND_NUMS == 1);
  std::cout << "Reading Test GITR step data " << ncFileName << "\n";
  // re-order the list in its constructor to leave out empty {}
  Field3StructInput fs({"intermediate"}, {}, {"nP", "nTHist", "dof"}, 0,
    {"RndIoni_at", "RndRecomb_at", "RndCollision_n1_at", "RndCollision_n2_at", 
     "RndCollision_xsi_at", "RndCrossField_at", "RndReflection_at",
     "Opt_IoniRecomb", "Opt_Diffusion", "Opt_Collision", "Opt_SurfaceModel"}); 
  auto stat = readInputDataNcFileFS3(ncFileName, fs, maxNPtcls, numPtclsRead, 
      "nP", debug);
  testGitrPtclStepData = o::Reals(fs.data);
  testGitrDataIoniRandInd = fs.getIntValueOf("RndIoni_at");
  testGitrDataRecRandInd = fs.getIntValueOf("RndRecomb_at");
  testGitrStepDataDof = fs.getIntValueOf("dof"); // or fs.getNumGrids(2);
  testGitrStepDataNumTsteps = fs.getIntValueOf("nTHist") - 1; // NOTE
  testGitrStepDataNumPtcls = fs.getIntValueOf("nP");
  testGitrCrossFieldDiffRndInd = fs.getIntValueOf("RndCrossField_at");
  testGitrCollisionRndn1Ind = fs.getIntValueOf("RndCollision_n1_at"); 
  testGitrCollisionRndn2Ind = fs.getIntValueOf("RndCollision_n2_at");
  testGitrCollisionRndxsiInd = fs.getIntValueOf("RndCollision_xsi_at");
  testGitrReflectionRndInd = fs.getIntValueOf("RndReflection_at");
  testGitrOptIoniRec = fs.getIntValueOf("Opt_IoniRecomb");
  testGitrOptDiffusion =  fs.getIntValueOf("Opt_Diffusion");
  testGitrOptCollision =  fs.getIntValueOf("Opt_Collision");
  testGitrOptSurfaceModel = fs.getIntValueOf("Opt_SurfaceModel");

  //if(debug)
    printf(" TestGITRdata: dof %d nT %d nP %d Index: rndIoni %d rndRec %d \n"
      " rndCrossFieldDiff %d rndColl_n1 %d rndColl_n2 %d rndColl_xsi %d "
      " rndReflection %d\n GITR run Flags: ioniRec %d diffusion %d "
      " collision %d surfmodel %d \n", testGitrStepDataDof, testGitrStepDataNumTsteps,
      testGitrStepDataNumPtcls, testGitrDataIoniRandInd, testGitrDataRecRandInd, 
      testGitrCrossFieldDiffRndInd, testGitrCollisionRndn1Ind, 
      testGitrCollisionRndn2Ind, testGitrCollisionRndxsiInd, 
      testGitrReflectionRndInd, testGitrOptIoniRec, testGitrOptDiffusion,
      testGitrOptCollision, testGitrOptSurfaceModel);

  return stat;
}

//timestep >0
void GitrmParticles::checkCompatibilityWithGITRflags(int timestep) {
  if(timestep==0) 
    printf("ERROR: random number GITR data doesn't match with flags \n");
  OMEGA_H_CHECK(timestep>0);
  if(ranIonization||ranRecombination)
    OMEGA_H_CHECK(testGitrOptIoniRec);
  else
    OMEGA_H_CHECK(!testGitrOptIoniRec);

  if(ranCoulombCollision)
    OMEGA_H_CHECK(testGitrOptCollision);
  else
    OMEGA_H_CHECK(!testGitrOptCollision);

  if(ranDiffusion)
    OMEGA_H_CHECK(testGitrOptDiffusion);
  else
    OMEGA_H_CHECK(!testGitrOptDiffusion);
  if(ranSurfaceReflection)
    OMEGA_H_CHECK(testGitrOptSurfaceModel);
  else
    OMEGA_H_CHECK(!testGitrOptSurfaceModel);
}

//allocated for totalPtcls in each picpart
void GitrmParticles::initPtclHistoryData(int hstep) {
  OMEGA_H_CHECK(hstep > 0);
  histInterval = hstep;
  if(histInterval > numIterations)
    histInterval = numIterations;
  dofHistory = 6; //x,y,z,vx,vy,vz
  histStep = 0; 
  //1 extra for initial
  nThistory = 1 + numIterations/histInterval;
  auto rem = numIterations%histInterval;
  if(rem)
    ++nThistory;
  nFilledPtclsInHistory = 0;
  int size = totalPtcls*dofHistory*nThistory;
  int nph = totalPtcls*nThistory;
  ptclHistoryData = o::Write<o::Real>(size, 0);
  ptclIdsOfHistoryData = o::Write<o::LO>(nph, -1);
}

//all are collected by rank=0 
void GitrmParticles::writePtclStepHistoryFile(std::string ncfile, bool debug) {
  o::HostWrite<o::Real> histData(ptclHistoryData);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  auto size = totalPtcls*dofHistory*nThistory;
  if(debug)
    printf("MPI gather history size %d totalPtcls %d  nThistory %d "
      " allocated history size %d\n", size, totalPtcls, nThistory, ptclHistoryData.size());
  MPI_Reduce(MPI_IN_PLACE, &histData[0], histData.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  //TODO use MPI_Gatherv
  if(comm_rank > 0)
    return;

  if(debug)
    for(int i =0 ; i < 200 && i< histData.size() && i<size; ++i) {
      printf(" %g", histData[i]);
      if(i%6==5) printf("\n");
    }
  int numPtcls = totalPtcls;
  OutputNcFileFieldStruct outStruct({"nP", "nT"}, {"x", "y", "z", "vx", "vy", "vz"},
                                     {numPtcls, nThistory});
  writeOutputNcFile(histData, numPtcls, dofHistory, outStruct, ncfile);
}

//if elem_ids invalid then store xpoints, else store final position. 
//if iter is before timestep starts, store init position, which makes
//total history step +1.
void GitrmParticles::updatePtclHistoryData(int iter, const o::LOs& elem_ids) {
  bool debug = 1;
  if(debug)
    printf(" histupdate iter %d interval %d \n", iter, histInterval);
  if(!histInterval || iter%histInterval)
    return;

  int iThistory = 1 + iter/histInterval; 
  auto size = ptclIdsOfHistoryData.size();
  auto ndi = dofHistory*iThistory;
  auto nh = nThistory;
  auto ptclIds = ptclIdsOfHistoryData;
  auto historyData = ptclHistoryData;
  auto vel_ps = ptcls->get<PTCL_VEL>();
  auto pos_ps = ptcls->get<PTCL_POS>();
  auto pos_next = ptcls->get<PTCL_NEXT_POS>();
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto xfids = wallCollisionFaceIds; 
  auto xpts = wallCollisionPts;
  auto lambda = PS_LAMBDA(const int& elem, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto vel = p::makeVector3(pid, vel_ps);
      auto pos = p::makeVector3(pid, pos_next);
      if(iter < 0)
        pos = p::makeVector3(pid, pos_ps);
      OMEGA_H_CHECK(elem_ids[pid] >= 0 || xfids[pid] >=0);
      if(elem_ids[pid] <0)
        for(int i=0; i<3; ++i)
          pos[i] = xpts[pid*3+i]; 
      int beg = nh*ptcl + ndi;
      for(int i=0; i<3; ++i) {
          historyData[beg+i] = pos[i];
          historyData[beg+3+i] = vel[i];
          if(debug)
            printf("iter %d ptcl %d pid %d i %d beg %d beg+i %d pos %g vel %g\n", 
             iter,ptcl, pid, i, beg, beg+i, pos[i], vel[i]);
      }
      Kokkos::atomic_exchange(&(ptclIds[ptcl]), ptcl);
    }// mask
  };
  ps::parallel_for(ptcls, lambda, "updateStepData");
}

void GitrmParticles::writeDetectedParticles(std::string fname, std::string header) {
  auto data_d = collectedPtcls;
  o::HostWrite<o::LO> dataTot(data_d);
  //TODO handle if different size in ranks
  MPI_Reduce(MPI_IN_PLACE, &dataTot[0], dataTot.size(), MPI_DOUBLE, MPI_SUM,
    0, MPI_COMM_WORLD);
  if(!gitrm::checkIfRankZero()) {
    printf("call to writeDetectedParticles from rank >0 ingnored \n");
    return;
  }
  if(fname=="")
    fname = "result.txt";
  std::ofstream outf(fname);
  auto total = 0;
  auto dataTot_d = dataTot.write();
  Kokkos::parallel_reduce(data_d.size(), OMEGA_H_LAMBDA(const int i, o::LO& lsum) {
    lsum += dataTot_d[i];
  }, total);
  Kokkos::fence();
  outf << header << "\n";
  outf << "total collected " <<  total << "\n";
  outf << "index   total\n";  
  for(int i=0; i<dataTot.size(); ++i)
    outf <<  i << " " << dataTot[i] << "\n";
}

//detector grids, for pisces
void GitrmParticles::initPtclDetectionData(int numGrid) {
  //o::LO numGrid = 14;
 collectedPtcls = o::Write<o::LO>(numGrid, 0);
}

//TODO dimensions set for pisces to be removed
//call this before re-building, since mask of exiting ptcl removed from origin elem
void GitrmParticles::updateParticleDetection(const o::LOs& elem_ids, o::LO iter, bool debug) {
  // test TODO move test part to separate unit test
  double radMax = 0.05; //m 0.0446+0.005
  double zMin = 0; //m height min
  double zMax = 0.15; //m height max 0.14275
  double htBead1 =  0.01275; //m ht of 1st bead
  double dz = 0.01; //m ht of beads 2..14
  auto& data_d = collectedPtcls;
  auto pisces_ids = mesh.get_array<o::LO>(o::FACE, "DetectorSurfaceIndex");
  auto pid_ps = ptcls->get<PTCL_ID>();
  //based on ptcl id or ptcls pid ?
  const auto& xpoints = wallCollisionPts;
  auto& xpointFids = wallCollisionFaceIds;

  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto ptcl = pid_ps(pid);
      auto fid = xpointFids[pid];
      auto elm = elem_ids[pid];
      if(elm < 0 && fid>=0) {
        // test
        auto xpt = o::zero_vector<3>();
        for(o::LO i=0; i<3; ++i)
          xpt[i] = xpoints[pid*3+i]; //ptcl = 0..numPtcls
        auto x = xpt[0], y = xpt[1], z = xpt[2];
        o::Real rad = sqrt(x*x + y*y);
        o::LO zInd = -1;
        if(rad < radMax && z <= zMax && z >= zMin)
          zInd = (z > htBead1) ? (1+(o::LO)((z-htBead1)/dz)) : 0;
        
        auto detId = pisces_ids[fid];
        if(detId > -1) { //TODO
          if(debug)
            printf("ptclID %d zInd %d detId %d pos %.5f %.5f %.5f iter %d\n", 
              pid_ps(pid), zInd, detId, x, y, z, iter);
          Kokkos::atomic_increment(&data_d[detId]);
        }
      }
    }
  };
  ps::parallel_for(ptcls, lamb, "StoreDetectedData");
}

// gridsR gets preference. If >1 then gridsZ not taken
void GitrmParticles::storePtclDataInGridsRZ(o::LO iter, o::Write<o::LO>& data_d,
   int gridsR, int gridsZ, double radMax, double zMax, double radMin, double zMin) {
  auto dz = (zMax - zMin)/gridsZ;
  auto dr = (radMax - radMin)/gridsR;
  auto pid_ps = ptcls->get<PTCL_ID>();
  auto tgt_ps = ptcls->get<PTCL_NEXT_POS>();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if(mask >0) {
      auto x = tgt_ps(pid, 0);
      auto y = tgt_ps(pid, 1);
      auto z = tgt_ps(pid, 2);
      auto rad = sqrt(x*x + y*y);
      int ind = -1;
      if(rad < radMax && radMin >= radMin && z <= zMax && z >= zMin)
        if(gridsR >1) //prefer radial
          ind = (int)((rad - radMin)/dr);
        else if(gridsZ >1)
          ind = (int)((z - zMin)/dz);
      //int dir = (gridsR >1)?1:0;
      if(ind >=0) {
        Kokkos::atomic_increment(&data_d[ind]);
      }
    } //mask
  };
  ps::parallel_for(ptcls, lamb);
}
