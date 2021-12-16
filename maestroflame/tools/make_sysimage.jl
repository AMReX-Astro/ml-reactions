import Pkg;
Pkg.add("PackageCompiler")
Pkg.add("Plots")
Pkg.add("PyCall")
using PackageCompiler, Plots, PyCall
create_sysimage(:Plots, sysimage_path="sys_plots.so")

