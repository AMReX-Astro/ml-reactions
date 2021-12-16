using Pkg
using DelimitedFiles

using Base
#Pkg.activate("julia_env")
#Pkg.add("GR")



# using Plots
#
#println(pwd())


#cut off pred_v_sol.jl
source_dir = chop(Base.source_path(), tail = 13)


pred = readdlm(source_dir * "python_to_julia_data/pred.txt")
labels = readdlm(source_dir * "python_to_julia_data/labels.txt")
targets = readdlm(source_dir * "python_to_julia_data/targets.txt")

save_dir = labels[length(labels)]
#package_dir = labels[length(labels)]
labels = labels[1:length(labels)-1]

#Pkg.activate(package_dir * "/julia_env")
#ENV["GRDIR"]=""
#Pkg.build("GR")
using Plots


labels = string.(labels)
labels = reshape(labels, (1, size(labels)[1]) )


plotd = scatter(pred, targets, labels=labels, dpi=600)
savefig(plotd, save_dir * "julia_m2_prediction_vs_solution.png")
#we now cut out data below 0. Julia plotting can't plot log plots if there are negative
# or zero data i guess
#inds = (targets .> 0) .& (pred .> 0)

#we now cut out data below 0. Julia plotting can't plot log plots if there are negative
# or zero data i guess. but to preserve shape, we just put all these points at (1,1)
inds = (targets .<= 0) .| (pred .<= 0)
targets[inds] .= 1
pred[inds] .= 1


plotd = scatter(targets, pred, xaxis=:log, yaxis=:log, labels=labels, dpi=600)
savefig(plotd, save_dir * "julia_m2_log_prediction_vs_solution.png")
