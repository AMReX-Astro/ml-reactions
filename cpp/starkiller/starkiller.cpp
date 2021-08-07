#include <starkiller.H>
#include <extern_parameters.H>

#ifdef MICROPHYSICS_FORT
#include <extern_parameters_F.H>
#endif

#include <AMReX_VisMF.H>

using namespace amrex;

std::string ReactionSystem::probin_file = "probin";

// constructor
ReactionSystem::ReactionSystem() = default;

ReactionSystem::ReactionSystem(const ReactionSystem& src)
{
    size = src.size;
    state.resize(size);
    for (int i = 0; i < size; i++) {
        state[i].define(src.state[i].boxArray(), src.state[i].DistributionMap(), NSCAL, 0);
        MultiFab::Copy(state[i], src.state[i], 0, 0, NSCAL, 0);
    }
}

// destructor
ReactionSystem::~ReactionSystem() = default;

// initialize variables
void ReactionSystem::init(const int train_size, const amrex::BoxArray& ba,
                          const amrex::DistributionMapping& dm)
{
    // initialize multifabs
    size = train_size;
    state.resize(size);
    for (int i = 0; i < size; i++){
        state[i].define(ba, dm, NSCAL, 0);
        state[i].setVal(0.0);
    }

    static bool firstCall = true;

    if (firstCall) {
        // initialize the external runtime parameters
        init_extern();

        // initialize network, eos, conductivity
        network_init();   // includes actual_rhs_init()
        eos_init();
        conductivity_init();

        firstCall = false;
    }
}

// initialize extern parameters
void ReactionSystem::init_extern()
{
    // initialize the external runtime parameters -- these will
    // live in the probin the probin
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "reading extern runtime parameters ..." << std::endl;
    }

    const int probin_file_length = probin_file.length();
    Vector<int> probin_file_name(probin_file_length);

    for (int i = 0; i < probin_file_length; i++) {
        probin_file_name[i] = probin_file[i];
    }

#ifdef MICROPHYSICS_FORT
    // read them in in Fortran from the probin file
    runtime_init(probin_file_name.dataPtr(),&probin_file_length);
#endif
    
    // now sync with C++ and read in the C++ parameters
    init_extern_parameters();
    
#ifdef MICROPHYSICS_FORT
    // sync any C++ changes with Fortran
    update_fortran_extern_after_cxx();
#endif
}

// initialize state
void ReactionSystem::init_state(const Real dens, const Real temp,
                                const Real xhe, const Real end_time,
                                bool const_state)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "initializing initial conditions ..." << std::endl;
    }

    const bool const_flag = const_state;

    // find index of He4
    int he_species = 0;
    for (int i = 0; i < NumSpec; ++i) {
        std::string spec_string = short_spec_names_cxx[i];
        if (spec_string == "He4") {
            he_species = i + FS;
            break;
        }
    }
    if (he_species == 0) {
        Abort("ERROR: he4 not found in network!");
    }

    ResetRandomSeed(time(0));

    for (int l = 0; l < size; l++) {
        for (MFIter mfi(state[l], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto tileBox = mfi.tilebox();

            const Array4<Real> state_arr = state[l].array(mfi);

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (const_flag) {
                    // state is constant, time varies
                    state_arr(i,j,k,DT) = amrex::Random()*end_time;

                    // set density and temperature
                    state_arr(i,j,k,RHO) = dens;
                    state_arr(i,j,k,TEMP) = temp;

                    // mass fractions
                    for (int n = 0; n < NumSpec; ++n) {
                        state_arr(i,j,k,FS+n) = (1.0-xhe) / (NumSpec-1);
                    }
                    state_arr(i,j,k,he_species) = xhe;
                } else {
                    // time is constant / state varies
                    state_arr(i,j,k,DT) = end_time;
                }
            });
        }
    }
    VisMF::Write(state[0], "plt_x0");
    //WriteSingleLevelPlotfile("plt_train", state[0], {"rho"}, geom, 0.0, 0);

}

// Get the solutions at times dt (stored in state)
void ReactionSystem::sol(Vector<MultiFab>& y)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "computing exact solution ..." << std::endl;
    }

    // initialize y
    y.resize(size);

    for (int i = 0; i < size; i++){
        y[i].define(state[i].boxArray(), state[i].DistributionMap(), NSCAL, 0);
        MultiFab::Copy(y[i], state[i], DT, DT, 1, 0);
    }

    // evaluate the system solution
    for (int l = 0; l < size; l++) {
        for (MFIter mfi(state[l], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto tileBox = mfi.tilebox();

            const Array4<Real> state_arr = state[l].array(mfi);
            const Array4<Real> y_arr = y[l].array(mfi);

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // construct a burn type
                burn_t state_out;

                // set density & temperature
                state_out.rho = state_arr(i,j,k,RHO);
                state_out.T = state_arr(i,j,k,TEMP);

                // mass fractions
                for (int n = 0; n < NumSpec; ++n) {
                    state_out.xn[n] = state_arr(i,j,k,FS+n);
                }

                // integrate to get the output state
                Real dt = state_arr(i,j,k,DT);
                integrator(state_out, dt);

                // pass the solution values
                y_arr(i,j,k,TEMP) = state_out.T;
                y_arr(i,j,k,RHOE) = state_out.e;
                for (int n = 0; n < NumSpec; ++n) {
                    y_arr(i,j,k,FS+n) = state_out.xn[n];
                }
                y_arr(i,j,k,RHO) = state_out.rho;
            });
        }
    }
    VisMF::Write(y[0], "plt_y0");
}

// Get the solution rhs given state y
void ReactionSystem::rhs(const Vector<MultiFab>& y,
                         Vector<MultiFab>& dydt)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "computing rhs ..." << std::endl;
    }

    // initialize dydt
    dydt.resize(size);

    for (int i = 0; i < size; i++){
        dydt[i].define(y[i].boxArray(), y[i].DistributionMap(), NSCAL, 0);
        dydt[i].setVal(0.0);
    }

    // evaluate the system solution
    for (int l = 0; l < size; l++) {
        for (MFIter mfi(y[l], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const auto tileBox = mfi.tilebox();

            const Array4<const Real> y_arr = y[l].array(mfi);
            const Array4<const Real> state_arr = state[l].array(mfi);
            const Array4<Real> dydt_arr = dydt[l].array(mfi);

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // construct a burn type
                burn_t state_in;

                // set density & temperature
                state_in.rho = y_arr(i,j,k,RHO);
                state_in.T = amrex::max(y_arr(i,j,k,TEMP), 0.0);

                // mass fractions
                for (int n = 0; n < NumSpec; ++n) {
                    state_in.xn[n] = max(y_arr(i,j,k,FS+n), 0.0);
                }

                // evaluate the rhs
                Array1D<Real, 1, neqs> ydot;
                actual_rhs(state_in, ydot);
                // note ydot is 1-based

                // pass the solution values
                for (int n = 0; n < NumSpec; ++n) {
                    dydt_arr(i,j,k,FS+n) = aion[n]*ydot(1+n);
                }
                dydt_arr(i,j,k,RHOE) = ydot(net_ienuc);
                dydt_arr(i,j,k,TEMP) = state_in.T;
                dydt_arr(i,j,k,RHO) = state_in.rho;
            });
        }
    }
    VisMF::Write(dydt[0], "plt_dydt0");
}
