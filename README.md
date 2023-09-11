# The Switching Hierarchical Gaussian Filter (i.e. Switching Gaussian Controlled Variance)

The purpose of the switching HGF is to track a non-stationary (volatile) signal. The HGF has been used to model behavior within a changing environment, and the SHGF is an extension of this model that considers
behavioral outcomes over a set period of time as resulting from dynamics of several systems, rather than a single system. For our purposes, we are interested in the Probabilistic Reversal Learning task and have 
previously used the HGF to track inferred beliefs of individuals who experience psychosis.

The motivation for extending the HGF model is that we think the reward_tally, or the total number of points at each of 160 trials, might be a signal that submits to regime-switching dynamics.

## Getting Started (+Overview) and where did we get these scripts?

We use the scripts and packages from the [BIAS Lab](https://github.com/biaslab/SGCV).

After cloning this repository to your local machine, open the project in VS Code (or any preferred IDE that can work with Julia code and Jupyter notebooks). You will need to download Julia - any version after 1.3, although the most current version is recommended. After downloading Julia, type <code>julia</code> into the terminal. You will need to install the following packages:


*ProgressMeter
*CSV
*DataFrames
*Plots
*SparseArrays
*HCubature
*FastGaussQuadrature
*ForwardDiff
*StatsPlots
*Distributions
*JLD
*ForneyLab@v0.11.3

Do this by typing <code>[</code> in the terminal, followed by: <code>add ProgressMeter CSV DataFrames Plots SparseArrays HCubature FastGaussQuadrature ForwardDiff StatsPlots Distributions JLD ForneyLab@v0.11.3</code>. Note that ForneyLab version must be 0.11.3 which is stable and compatible with Julia v1.3+.

## Implementation (and key functions)

## Inference (how do we know the number of regimes?)

## What do results and figures mean?
