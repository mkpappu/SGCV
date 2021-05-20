using JLD,Plots,PGFPlotsX,SparseArrays, CSV, DataFrames,LaTeXStrings
pgfplotsx()
push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")

df = CSV.File("data/AAPL.csv") |> DataFrame
AAPL_stocks = df[!, :Open]

shgf3l_results = JLD.load("dump/results_3shgf_stocks_mixture.jld")
shgf2l_results = JLD.load("dump/results_2shgf_stocks_mixture.jld")
hgf2l_results  = JLD.load("dump/results_hgf_stocks_mixture.jld")

fe_shgf3l = shgf3l_results["results"]["fe"]
fe_shgf2l = shgf2l_results["results"]["fe"]
fe_hgf2l  = hgf2l_results["results"]["fe"]
fe_max_iter = 60
free_energy_axis = @pgf Axis({xlabel="iteration",
           ylabel="free-enery [nats]",
        legend_pos = "north east",
        grid = "major",
        mark_options = {scale=0.3},
    },
    Plot({mark = "square*", "blue"},Coordinates(collect(1:fe_max_iter),fe_shgf3l[1:fe_max_iter])),
    LegendEntry("3L-SHGF"),
    Plot({mark = "triangle*", "red"},Coordinates(collect(1:fe_max_iter),fe_shgf2l[1:fe_max_iter])),
    LegendEntry("2L-SHGF"),
    Plot({mark = "o", "black"},Coordinates(collect(1:fe_max_iter),fe_hgf2l[1:fe_max_iter])),
    LegendEntry("2L-HGF"),
)
pgfsave("figures/validation_results_free_energy.tikz", free_energy_axis)


mz_shgf3l = shgf3l_results["results"]["mz1"]
mz_shgf2l = shgf2l_results["results"]["mz"]
mz_hgf2l  = hgf2l_results["results"]["mz"]
vz_shgf3l = shgf3l_results["results"]["vz1"]
vz_shgf2l = shgf2l_results["results"]["vz"]
vz_hgf2l  = hgf2l_results["results"]["vz"]
n_samples = length(AAPL_stocks)
axis3 = @pgf Axis({xlabel=L"t",
           ylabel=L"x_t^{(2)}",
        legend_pos = "north west",
        mark_options = {scale=0.3},
        grid="major",
        style = {thick}
    },
    Plot({no_marks,color="blue"},Coordinates(collect(1:n_samples), mz_shgf2l)),
    LegendEntry("2L-SHGF"),
    Plot({ "name path=f", no_marks,color="blue",opacity=0.2 }, Coordinates(collect(1:n_samples), mz_shgf2l .+  sqrt.(vz_shgf2l))),
    Plot({ "name path=g", no_marks, color="blue",opacity=0.2}, Coordinates(collect(1:n_samples), mz_shgf2l .-  sqrt.(vz_shgf2l))),
    Plot({ thick, color = "blue", fill = "blue", opacity = 0.2 },
               raw"fill between [of=f and g]"),
    Plot({no_marks,color="red"},Coordinates(collect(1:n_samples), mz_shgf3l)),
    LegendEntry("3L-SHGF"),
    Plot({ "name path=f1", no_marks,color="red",opacity=0.2 }, Coordinates(collect(1:n_samples), mz_shgf3l .+  sqrt.(vz_shgf3l))),
    Plot({ "name path=g1", no_marks, color="red",opacity=0.2}, Coordinates(collect(1:n_samples), mz_shgf3l .-  sqrt.(vz_shgf3l))),
    Plot({ thick, color = "blue", fill = "red", opacity = 0.2 },
              raw"fill between [of=f1 and g1]"),
    Plot({no_marks,color="black"},Coordinates(collect(1:n_samples), mz_hgf2l)),
    LegendEntry("2L-HGF"),
    Plot({ "name path=f2", no_marks,color="black",opacity=0.2 }, Coordinates(collect(1:n_samples), mz_hgf2l .+  sqrt.(vz_hgf2l))),
    Plot({ "name path=g2", no_marks, color="black",opacity=0.2}, Coordinates(collect(1:n_samples), mz_hgf2l .-  sqrt.(vz_hgf2l))),
    Plot({ thick, color = "blue", fill = "black", opacity = 0.2 },
            raw"fill between [of=f2 and g2]"),
)

pgfsave("figures/validation_results_vol.tikz", axis3)

mx_shgf3l = shgf3l_results["results"]["mx1"]
mx_shgf2l = shgf2l_results["results"]["mx"]
mx_hgf2l  = hgf2l_results["results"]["mx"]
vx_shgf3l = shgf3l_results["results"]["vx"]
vx_shgf2l = shgf2l_results["results"]["vx"]
vx_hgf2l  = hgf2l_results["results"]["vx"]


mz2_shgf3l = shgf3l_results["results"]["mz2"]
plot(mz2_shgf3l)
ms = shgf2l_results["results"]["ms"]
categories = [x[2] for x in findmax.(ms)]
maxup = maximum(AAPL_stocks) + 3.0
mindown = minimum(AAPL_stocks) - 3.0
axis4 = @pgf Axis({xlabel=L"t",
           ylabel=L"x_t^{(1)}",
        legend_pos = "north west",
        mark_options = {scale=0.3},
        grid="major",
        style = {thick}
    },
    Plot({no_marks,color="magenta"},Coordinates(collect(1:n_samples), mx_shgf2l)),
    LegendEntry("2L-SHGF"),
    Plot({ "name path=f", no_marks,color="magenta",opacity=0.2 }, Coordinates(collect(1:n_samples), mx_shgf2l .+  3*sqrt.(vx_shgf2l))),
    Plot({ "name path=g", no_marks, color="magenta",opacity=0.2}, Coordinates(collect(1:n_samples), mx_shgf2l .-  3*sqrt.(vx_shgf2l))),
    Plot({ thick, color = "magenta", fill = "magenta", opacity = 0.2 },
               raw"fill between [of=f and g]"),
    Plot({only_marks,color="black",opacity=0.3}, Coordinates(collect(1:n_samples),AAPL_stocks)),
    Plot(
       {only_marks, scatter, scatter_src = "explicit"},
       Table(
           {x = "x", y = "y", meta = "col"},
            x = collect(1:n_samples), y = mindown*ones(n_samples), col = categories
       ),
    ),
)

pgfsave("figures/validation_prices2.tikz", axis4)
