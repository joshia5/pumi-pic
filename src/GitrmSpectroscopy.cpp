#include <mpi.h>
#include <netcdf>
#include <vector>
#include "GitrmSpectroscopy.hpp"

//TODO pass i/p obj, or use static obj
GitrmSpectroscopy::GitrmSpectroscopy() {
  initSpectroscopy();
}

void makeSpectroscopyGrid(const double net0, const double range, const int netN, 
   o::Write<o::Real>& gridBins) {
  o::parallel_for(netN, OMEGA_H_LAMBDA(const int& i) {
    gridBins[i] = net0 + 1.0/(netN-1) *i*range;   //netN-1 in gitr ?
  });
}

//pass input/config object that has the input values
void GitrmSpectroscopy::initSpectroscopy() {
  //TODO read from file. These are for pisces
  netX0 = -0.11;
  netXn = 0.11;
  netY0 = -0.11;
  netYn = 0.11;
  netZ0 = 0;
  netZn = 0.06;
  nX = 220;
  nY = 220;
  nZ = 60;
  nBins = 6;
  
  nSpec = (nBins+1)*nX*nY*nZ;
  netBins = o::Write<o::Real>(nSpec,0);
  gridX0 = netX0;
  gridY0 = netY0;
  gridZ0 = netZ0;        
  dX = (netXn-netX0)/(nX-1); // nX-1 in gitr ?
  gridXn = netX0 + (nX-1)*dX;  //?
  dY = (netYn-netY0)/nY;
  gridYn = netY0 + (nY-1)*dY; //?
  dZ = (netZn-netZ0)/nZ;
  gridZn = netZ0 + (nZ-1)*dZ; //?
  //grids not stored
}

//This won't work for partitioned mesh of non-full-buffer
//TODO move to IO file
//Call at the end of simulation
void GitrmSpectroscopy::writeSpectroscopyFile(const std::string& file) {
  o::HostWrite<o::Real> netBins_h(netBins);

  //FIXME this is only for full buffer partitioning.
  //MPI_IN_PLACE  wont work for variable size, use MPI GATHERV 
  MPI_Reduce(MPI_IN_PLACE, &netBins_h[0], 
    netBins_h.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(!gitrm::checkIfRankZero()) {
    return;
  }

  netCDF::NcFile ncf(file, netCDF::NcFile::replace);
  netCDF::NcDim ncnR = ncf.addDim("nR", nX);
  netCDF::NcDim ncnY = ncf.addDim("nY", nY);
  netCDF::NcDim ncnZ = ncf.addDim("nZ", nZ);
  netCDF::NcDim ncnBins = ncf.addDim("nBins", nBins+1); //TODO ?
  std::vector<netCDF::NcDim> dims;
  dims.push_back(ncnR);
  dims.push_back(ncnY);
  dims.push_back(ncnZ);
  dims.push_back(ncnBins);
  netCDF::NcVar ncn = ncf.addVar("n", netCDF::ncDouble, dims);

  //move to hpp if needed
  auto gridX = o::Write<o::Real>(nX);
  auto gridY = o::Write<o::Real>(nY);
  auto gridZ = o::Write<o::Real>(nZ);
  makeSpectroscopyGrid(netX0, netXn-netX0, nX, gridX);
  makeSpectroscopyGrid(netY0, netYn-netY0, nY, gridY);
  makeSpectroscopyGrid(netZ0, netZn-netZ0, nZ, gridZ);
  auto gridX_h = o::HostRead<o::Real>(gridX);
  auto gridY_h = o::HostRead<o::Real>(gridY);
  auto gridZ_h = o::HostRead<o::Real>(gridZ);

  netCDF::NcVar ncgridR = ncf.addVar("gridR", netCDF::ncDouble, ncnR);
  netCDF::NcVar ncgridY = ncf.addVar("gridY", netCDF::ncDouble, ncnY);
  netCDF::NcVar ncgridZ = ncf.addVar("gridZ", netCDF::ncDouble, ncnZ);
  ncgridR.putVar(&gridX_h[0]);
  ncgridY.putVar(&gridY_h[0]);
  ncgridZ.putVar(&gridZ_h[0]);
  ncn.putVar(&netBins_h[0]);
}
