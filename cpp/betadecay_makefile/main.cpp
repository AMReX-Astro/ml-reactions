#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#include <iostream>
#include <memory>

using namespace amrex;


int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        // input parameters
        int n_cell = 128;
        int max_grid_size = 32;
        std::string model_filename = "betadecay_model.pt";
        std::string test_filename = "test_data.txt";

        // Read parameters
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
            pp.query("model_file", model_filename);
            pp.query("test_data_file", test_filename);
        }

        /////////// INITIALIZE MULTIFAB INPUT ///////////////////////////////////////

        // Create grid
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

        // Read data from file
        std::ifstream FileStream;
        FileStream.open(test_filename.c_str(), std::istringstream::in);
        int data_size;
        if (FileStream.good()) {
            FileStream >> data_size;
        } else {
            Abort("ERROR: test_data_file is not valid!");
        }

#ifdef AMREX_USE_CUDA
        amrex::Gpu::ManagedVector<amrex::Real> input_data(data_size);
#else
        amrex::Vector<amrex::Real> input_data(data_size);
#endif

        for (int i = 0; i < data_size; ++i) {
            FileStream >> input_data[i];
        }

        Print() << "Reading of test data file complete." << std::endl;
        FileStream.close();

        // Put input data into multifab
        MultiFab input(ba, dm, 1, 0);

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(input, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            // Get the index space of the valid region
            const Box& tileBox = mfi.tilebox();

            const Array4<Real> input_arr = input.array(mfi);

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const int index = AMREX_SPACEDIM == 2 ?
                    i*n_cell+j : (i*n_cell+j)*n_cell+k;
                // Print() << index << ",  ";
                input_arr(i, j, k, 0) = input_data[amrex::min(index, data_size-1)];
            });
        }
        VisMF::Write(input, "test_data_mf");
        Print() << "Copying of input multifab complete." << std::endl;

        // retrieve size of multifab
        const auto nbox = geom.Domain().bigEnd();
        // std::printf("AMREX_SPACEDIM = %d, nx = %d, ny = %d, nz = %d \n",
        //             AMREX_SPACEDIM, nbox[0], nbox[1], nbox[2]);

        // // Copy input multifab to torch tensor
#if AMREX_SPACEDIM == 2
	at::Tensor t1 = torch::zeros({(nbox[0]+1)*(nbox[1]+1), 1});
#elif AMREX_SPACEDIM == 3
        at::Tensor t1 = torch::zeros({(nbox[0]+1)*(nbox[1]+1)*(nbox[2]+1), 1});
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

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const int index = AMREX_SPACEDIM == 2 ?
                    i*(nbox[1]+1)+j : (i*(nbox[1]+1)+j)*(nbox[2]+1)+k;
                t1[index][0] = input_arr(i, j, k, 0);
          });
        }

        // Load pytorch module via torch script
        torch::jit::script::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load(model_filename);
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return -1;
        }

        std::cout << "Model loaded.\n";

        // Evaluate torch data
        std::vector<torch::jit::IValue> inputs_torch{t1};
        at::Tensor outputs_torch = module.forward(inputs_torch).toTensor();
        std::cout << "example output: "
                  << outputs_torch.slice(/*dim=*/0, /*start=*/0, /*end=*/5) << '\n';
#ifdef USE_AMREX_CUDA
        outputs_torch = outputs_torch.to(torch::kCUDA);
#endif

        // Copy torch tensor to output multifab
        MultiFab output(ba, dm, 2, 0);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(output, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& tileBox = mfi.tilebox();
            auto const& output_arr = output.array(mfi);

            ParallelFor(tileBox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const int index = AMREX_SPACEDIM == 2 ?
                    i*(nbox[1]+1)+j : (i*(nbox[1]+1)+j)*(nbox[2]+1)+k;
                output_arr(i, j, k, 0) = outputs_torch[index][0].item<double>();
                output_arr(i, j, k, 1) = outputs_torch[index][1].item<double>();
          });
        }
        VisMF::Write(output, "output_mf");

        Print() << "Model evaluation complete." << std::endl;
    }

    amrex::Finalize();
}
