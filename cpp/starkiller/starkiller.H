#ifndef STARKILLER_H_
#define STARKILLER_H_

#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>

#include <burner.H>
#include <conductivity.H>
#include <eos.H>
#include <network.H>

// state indices
constexpr int NSCAL = 4 + NumSpec;
constexpr int RHO = 0;
constexpr int DT = 1;
constexpr int TEMP = 2;
constexpr int RHOE = 3;
constexpr int FS = 4;  // FirstSpec


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
    
    // these are defined in 'starkiller.cpp'
    static std::string probin_file;
};

#endif
