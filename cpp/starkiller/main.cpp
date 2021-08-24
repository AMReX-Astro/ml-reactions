#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <starkiller.H>

#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>

#include <iostream>
#include <memory>

using namespace amrex;


int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        int n_cell = 128;
        int max_grid_size = 32;

	std::string model_filename = "ts_model.pt";
	
        Real dens_norm = 5.0e7;
        Real temp_norm = 3.0e9;
	Real enuc_norm = 1.5e12;
	Real end_time = 1.0e-6;

        // read parameters
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
	    pp.query("model_file", model_filename);
            pp.query("density", dens_norm);
            pp.query("temperature", temp_norm);
	    pp.query("energy_nuc", enuc_norm);
	    pp.query("end_time", end_time);
        }

        // Initial mass fraction
        Real xhe = 1.0;
	
        /////////// LOAD PYTORCH MODEL  ///////////////////////////////////////

        // Load pytorch module via torch script
	Print() << "Reading " << model_filename << " file ... ";
        torch::jit::script::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load(model_filename);
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model.\n";
            return -1;
        }

        std::cout << "Model loaded.\n";
	
        /////////// GENERATING MULTIFAB DATASET ///////////////////////////////////////

        // initialize arbitrary grid
        Geometry geom;
        {
            RealBox rb({AMREX_D_DECL(0.0,0.0,0.0)}, {AMREX_D_DECL(1.0,1.0,1.0)}); // physical domain
            Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(false, false, false)};
            Geometry::Setup(&rb, 0, is_periodic.data());
            Box domain(IntVect(0), IntVect(n_cell-1));
            geom.define(domain);
        }
        BoxArray ba(geom.Domain());
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};
	
        // initialize input multifabs
        ReactionSystem system;
        system.init(ba, dm);
        system.init_state(dens_norm, temp_norm, enuc_norm, xhe, end_time/*,true*/);

        // Make a copy 
	MultiFab input(ba, dm, NIN, 0);
	MultiFab::Copy(input, system.state, 0, 0, NIN, 0);
	// DEBUG 
	// VisMF::Write(input, "model_input");
	
        Print() << "Initializing input multifab complete." << std::endl;

        // retrieve size of multifab
        const auto nbox = geom.Domain().bigEnd();
        
        // // Copy input multifab to torch tensor
#if AMREX_SPACEDIM == 2
	at::Tensor t1 = torch::zeros({(nbox[0]+1)*(nbox[1]+1), NIN});
#elif AMREX_SPACEDIM == 3
        at::Tensor t1 = torch::zeros({(nbox[0]+1)*(nbox[1]+1)*(nbox[2]+1), NIN});
#endif
        
#ifdef USE_AMREX_CUDA
        t1 = t1.to(torch::kCUDA);
#endif

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(input, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& tileBox = mfi.tilebox();
            auto const& input_arr = input.array(mfi);

            ParallelFor(tileBox, NIN,
			[=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                const int index = AMREX_SPACEDIM == 2 ?
                    i*(nbox[1]+1)+j : (i*(nbox[1]+1)+j)*(nbox[2]+1)+k;
                t1[index][n] = input_arr(i, j, k, n);
            });
        }

        // Evaluate torch data
        std::vector<torch::jit::IValue> inputs_torch{t1};
        at::Tensor outputs_torch = module.forward(inputs_torch).toTensor();
	std::cout << "example input: \n"
		  << t1.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';
        std::cout << "example output: \n"
                  << outputs_torch.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';
#ifdef USE_AMREX_CUDA
        outputs_torch = outputs_torch.to(torch::kCUDA);
#endif

        // Copy torch tensor to output multifab
        MultiFab output(ba, dm, NOUT, 0);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(output, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& tileBox = mfi.tilebox();
            auto const& output_arr = output.array(mfi);

            ParallelFor(tileBox, NOUT,
			[=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                const int index = AMREX_SPACEDIM == 2 ?
                    i*(nbox[1]+1)+j : (i*(nbox[1]+1)+j)*(nbox[2]+1)+k;
                output_arr(i, j, k, n) = outputs_torch[index][n].item<double>();
            });
        }
	// DEBUG
        // VisMF::Write(output, "model_output");

        Print() << "Model evaluation complete." << std::endl;

	
        // truth solutions
        MultiFab y;
        MultiFab ydot;
        system.sol(y);
        system.rhs(y, ydot);

	// compute error between output and truth solutions
	MultiFab diff(ba, dm, NOUT, 0);
	MultiFab::LinComb(diff, 1.0, output, 0, -1.0, y, 0, 0, NOUT, 0);

	// plot error
	int cnt = 0;
	Vector<std::string> varnames(NOUT);
	for (int i = 0; i < NumSpec; i++) {
            std::string spec_string = "X(";
            spec_string += short_spec_names_cxx[i];
            spec_string += ')';
            varnames[cnt++] = spec_string;
        }
        varnames[cnt++] = "enuc";

	WriteSingleLevelPlotfile("model_error", diff, varnames, geom, 0.0, 0);
    }

    amrex::Finalize();
}
