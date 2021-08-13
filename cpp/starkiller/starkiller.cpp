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
    state.define(src.state.boxArray(), src.state.DistributionMap(), NIN, 0);
    MultiFab::Copy(state, src.state, 0, 0, NIN, 0);
}

// destructor
ReactionSystem::~ReactionSystem() = default;

// initialize variables
void ReactionSystem::init(const amrex::BoxArray& ba,
                          const amrex::DistributionMapping& dm)
{
    // initialize multifabs
    state.define(ba, dm, NIN, 0);
    state.setVal(0.0);
    
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

    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	const auto tileBox = mfi.tilebox();

	const Array4<Real> state_arr = state.array(mfi);

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
    VisMF::Write(state, "plt_x0");
    //WriteSingleLevelPlotfile("plt_train", state[0], {"rho"}, geom, 0.0, 0);

}

// Get the solutions at times dt (stored in state)
void ReactionSystem::sol(MultiFab& y)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "computing exact solution ..." << std::endl;
    }

    y.define(state.boxArray(), state.DistributionMap(), NOUT, 0);
    
    // evaluate the system solution
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	const auto tileBox = mfi.tilebox();

	const Array4<Real> state_arr = state.array(mfi);
	const Array4<Real> y_arr = y.array(mfi);

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
	    y_arr(i,j,k,ENUC) = state_out.e;
	    for (int n = 0; n < NumSpec; ++n) {
		y_arr(i,j,k,n) = state_out.xn[n];
	    }
	});
    }
    VisMF::Write(y, "plt_y0");
}

// Get the solution rhs given state y
void ReactionSystem::rhs(const MultiFab& y,
                         MultiFab& dydt)
{
    if (ParallelDescriptor::IOProcessor()) {
        std::cout << "computing rhs ..." << std::endl;
    }

    // initialize dydt
    dydt.define(y.boxArray(), y.DistributionMap(), NOUT, 0);
    dydt.setVal(0.0);

    // evaluate the system solution
    for (MFIter mfi(y, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	const auto tileBox = mfi.tilebox();

	const Array4<const Real> y_arr = y.array(mfi);
	const Array4<const Real> state_arr = state.array(mfi);
	const Array4<Real> dydt_arr = dydt.array(mfi);

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
		dydt_arr(i,j,k,n) = aion[n]*ydot(1+n);
	    }
	    dydt_arr(i,j,k,ENUC) = ydot(net_ienuc);
        });
    }
    VisMF::Write(dydt, "plt_dydt0");

}
