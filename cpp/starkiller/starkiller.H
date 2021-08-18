#ifndef STARKILLER_H_
#define STARKILLER_H_

#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>

#include <burner.H>
#include <conductivity.H>
#include <eos.H>
#include <network.H>

// state indices
constexpr int NIN = 3 + NumSpec;
constexpr int DT = 0;
constexpr int FS = 1;  // FirstSpec
constexpr int RHO = NumSpec + 1;
constexpr int TEMP = NumSpec + 2;

constexpr int NOUT = 1 + NumSpec;
constexpr int ENUC = NumSpec;

class ReactionSystem
{
  public:
  
    // constructor
    ReactionSystem();
    ReactionSystem(const ReactionSystem& src);

    // destructor
    virtual ~ReactionSystem();

    // initialize class variables
    void init(const amrex::BoxArray& ba,
              const amrex::DistributionMapping& dm);

    // initialize extern parameters
    void init_extern();

    // initialize state
    void init_state(const amrex::Real dens, const amrex::Real temp,
                    const amrex::Real xhe, const amrex::Real end_time,
                    bool const_state = true);

    // compute the solution
    void sol(amrex::MultiFab& y);

    // compute the rhs
    void rhs(const amrex::MultiFab& y,
             amrex::MultiFab& dydt);

    
    /* PRIVATE VARIABLES */
    
    amrex::MultiFab state;
    amrex::Real dens_norm, temp_norm;
    
    // these are defined in 'starkiller.cpp'
    static std::string probin_file;
};

#endif
