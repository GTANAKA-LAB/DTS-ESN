# DTS-ESN
The echo state network (ESN) is a special recurrent neural network model proposed by H. Jaeger in 2001, which is a representative model for reservoir computing.
The diverse-timescale ESN (DTS-ESN) ([Tanaka et al., Phys. Rev. Res. 2022](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.L032014)) is an extended ESN model with diverse timescales for prediction of multiscale dynamics.
The python codes for the model and demonstrations are provided.  

  ## Files
  * esn_dts.py: the main DTS-ESN engine
  * pred_tcLorenz.py: the source file for demonstrations of time series prediction with the two-coupled Lorenz system
  * demo_pred_tcLorenz.ipynb: a demonstration of "pred_tcLorenz.py" with jupyter notebook

  ## Usage
  Some python modules, such as numpy, scipy, matplotlib, and networkx, are required to run the codes.
  
  ## Developer
  Gouhei Tanaka, International Research Center for Neurointelligence (IRCN), The University of Tokyo
  
  ## Citation
  G. Tanaka, T. Matsumori, H. Yoshida, K. Aihara, "Reservoir computing with diverse timescales for prediction of multiscale dynamics," arXiv:2108.09446 (Under review for journal publication)
