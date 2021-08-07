DEBUG      = FALSE
DIM        = 2
COMP	   = gnu
USE_MPI    = TRUE
USE_OMP    = FALSE

TINY_PROFILE = FALSE
PROFILE      = FALSE # TRUE overrides TINY_PROFILE


# define the location of the ml-reactions home directory
MLREACTIONS_HOME  := ../..


# Set the EOS, conductivity, and network directories
# We use the version in $(MICROPHYSICS_HOME)/Microphysics/(EOS/conductivity/networks)
EOS_DIR          := helmholtz
CONDUCTIVITY_DIR := stellar
NETWORK_DIR      := aprox13
INTEGRATOR_DIR   := VODE

Bpack   := ./Make.package
Blocs   := .


include $(MLREACTIONS_HOME)/cpp/Make.ml-starkiller
