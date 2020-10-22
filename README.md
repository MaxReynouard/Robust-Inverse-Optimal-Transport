# RIOT: Robust Inverse Optimal Transport project

In this repository, you will find an implementation of the RIOT algorithm presented by Ruilin Li in Learning to Match via Inverse Optimal Transport : https://jmlr.org/papers/volume20/18-700/18-700.pdf.

This implementation has also been improved to strengthen the convergence. 

## In riot.py : 
- You'll find implementations of the Sinkhorn-Knopp algorithm to solve the Regularized Optimal Transport (ROT) problem, 
- You'll also find its log-stabilized version. This version prevents overflow issues that were present in Ruilin Li's version of the algorithm. 
- Finally, you'll find the RIOT algorithm. In addition to some simplification, you'll find the replacement of the use of standard Sinkhorn-knopp algorithm by its log-stabilisation version, this leads to further simplifications. And you'll also find a stabilisation of the inner loop. This stabilisation makes the algorithm extremely more robust

## In Sinkhorn_vs_log_stabilized_sinkhorn.ipynb :
- There is a comparison of the results we get from both approaches, this aims to check that we converge to similar solutions
- There also is a test of the speed of those algorithms. They seem to have comparable speeds

## In Synthetic_dataset.ipynb :
- There is a generator for the synthetic data, similar to what is explained in Ruilin Li's paper 
- There is an exemple of the use of riot on those data,
- And there is a quick noise sensibility study, to test the robustness of the convergence
