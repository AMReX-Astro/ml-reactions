#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <AMReX_ParmParse.H>

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

        std::cout << "ok\n";

    }

    amrex::Finalize();
}
