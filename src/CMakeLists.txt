
set(CD ${CMAKE_CURRENT_SOURCE_DIR})
set(HEADERS
  ${CD}/GitrmIonizeRecombine.hpp
  ${CD}/GitrmPush.hpp
  ${CD}/GitrmMesh.hpp
  ${CD}/GitrmParticles.hpp
  ${CD}/GitrmInputOutput.hpp
  ${CD}/GitrmSurfaceModel.hpp
  ${CD}/GitrmCoulombCollision.hpp
  ${CD}/GitrmThermalForce.hpp
  ${CD}/GitrmCrossDiffusion.hpp
  ${CD}/GitrmSpectroscopy.hpp
  PARENT_SCOPE
)

set(SOURCES
  ${CD}/GitrmMesh.cpp
  ${CD}/GitrmParticles.cpp
  ${CD}/GitrmIonizeRecombine.cpp
  ${CD}/GitrmInputOutput.cpp
  ${CD}/GitrmSurfaceModel.cpp
  ${CD}/GitrmSpectroscopy.cpp
  PARENT_SCOPE
)

bob_end_subdir()
