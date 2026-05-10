# Rao and Ballard's Hierarchical Predictive Coding:
This repository contains files for implementing an HPC model with Gaussian priors and sparse priors over responses. The main file through which functions are called, and models are initialized and implemented, is the run_predictivecoding notebook. 
<img width="567" height="310" alt="image" src="https://github.com/user-attachments/assets/e6351826-bd3a-4100-8cbd-2beb2b849b15" /> 

model.py contains the model definition for a 2-level HPC, dataset.py contains the preprocessing steps for the natural images, and helperFunctions.py contains additional functions for post-training analysis and endstopping simulations. 

Additionally, there is an attempt to shift MAP inference to sampling to obtain a distribution over the posterior response values.
