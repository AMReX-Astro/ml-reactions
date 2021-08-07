#include <torch/torch.h>

#include <starkiller.H>

#include <AMReX_ParmParse.H>
#include <AMReX_VisMF.H>

#include <memory>

using namespace amrex;


int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        int n_cell = 128;
        int max_grid_size = 32;
        int size_train = 32;
	
        Real dens = 1.0e8;
        Real temp = 4.0e8;
        Real end_time = 1.0;

        // read parameters
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
            pp.query("max_grid_size", max_grid_size);
            pp.query("size_train", size_train);
            pp.query("density" , dens);
            pp.query("temperature" , temp);
            pp.query("end_time", end_time);
        }

        // Initial mass fraction
        Real xhe = 1.0;
	
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

        // initialize training multifabs
        ReactionSystem system;
        system.init(size_train, ba, dm);
        system.init_state(dens, temp, xhe, end_time/*,true*/);

        // compute training solutions
        Vector<MultiFab> y;
        Vector<MultiFab> ydot;
        system.sol(y);
        system.rhs(y, ydot);

	//std::cout<<"Tensor example from PYTORCH!"<<std::endl;
        //torch::Tensor tensor = torch::rand({2, 3});
        //std::cout << tensor << std::endl;


#if 0
        /////////// MULTIFAB PREDICTION ///////////////////////////////

        MultiFab mfp;
        {
            // BoxArray ba(geom.Domain());
            // ba.maxSize(max_grid_size);
            // DistributionMapping dm{ba};

            // mfp.define(ba, dm, 3, 0, MFInfo(), *factory);
            // mfp.setVal(0.0);
        }

        VisMF::Write(mfp,"plt_pred");

        torch::Tensor tensorp;
        tensorp = tensorp.permute({0, 3, 1, 2}); // convert to CxHxW
        tensorp = tensorp.to(torch::kF32);
    
        // Predict the probabilities for the classes.
        torch::Tensor log_prob = model(tensorp);
        torch::Tensor prob = torch::exp(log_prob);
    
        printf("Probability of being\n\
        a circle = %.2f percent\n\
        a square = %.2f percent\n", prob[0][0].item<float>()*100., prob[0][1].item<float>()*100.); 

        std::cout<<"NN example from PYTORCH!"<<std::endl;

        // Create a new Net.
        auto net = std::make_shared<Net>();

        // Create a multi-threaded data loader for the MNIST dataset.
        auto data_loader = torch::data::make_data_loader(
                                                         torch::data::datasets::MNIST("../data").map(
                                                                                                     torch::data::transforms::Stack<>()),
                                                         /*batch_size=*/64);

        // Instantiate an SGD optimization algorithm to update our Net's parameters.
        torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

        for (size_t epoch = 1; epoch <= 10; ++epoch) {
            size_t batch_index = 0;
            // Iterate the data loader to yield batches from the dataset.
            for (auto& batch : *data_loader) {
                // Reset gradients.
                optimizer.zero_grad();
                // Execute the model on the input data.
                torch::Tensor prediction = net->forward(batch.data);
                // Compute a loss value to judge the prediction of our model.
                torch::Tensor loss = torch::nll_loss(prediction, batch.target);
                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();
                // Update the parameters based on the calculated gradients.
                optimizer.step();
                // Output the loss and checkpoint every 100 batches.
                if (++batch_index % 100 == 0) {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                              << " | Loss: " << loss.item<double>() << std::endl;
                    // Serialize your model periodically as a checkpoint.
                    torch::save(net, "net.pt");
                }
            }
        }
#endif

    }

    amrex::Finalize();
}
