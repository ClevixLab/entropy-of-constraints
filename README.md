# Entropy of Constraints (EoC) Simulations

Code for numerical validation in the paper "Entropy of Constraints..." (Feb 2025).

## Dependencies
Python 3.10+
pip install numpy scipy matplotlib

## Installation
git clone https://github.com/ClevixLab/entropy-of-constraints.git
cd entropy-of-constraints
# No pip install needed if using above libs

## Reproducing Paper Results (Section 11)
- Critical threshold & error 0%: python replicator_model.py --mode threshold --alpha_range 0.1:0.4 --beta 0.003
- Transfer term T(t) & R²=0.92: python EoC_fig1_8.py --figure 4 --params default
- Attractor dimension dimF≈2.1: python eoc_system.py --dim_calc --n=4
- Entropy evolution plots: python EoC_fig1_8.py --figures 1-3

Expected outputs: See paper Section 11 for matches.
