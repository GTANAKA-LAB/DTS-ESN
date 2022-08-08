# DTS-ESN
The echo state network (ESN) is a special recurrent neural network model proposed by H. Jaeger in 2001, which is a representative model for reservoir computing.
The diverse-timescale ESN (DTS-ESN) ([Tanaka et al., Phys. Rev. Res. 2022](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.L032014)) is an extended ESN model with a rich variety of timescales for prediction of multiscale dynamics.
The python codes for the model and demonstrations are provided.  

  ## Folders
  * 1step_ahead_prediction: this folder contains a sample code and a demo code for reproducing Fig. S2(e) in the supplementary material of the reference paper.
  * longterm_prediction: this folder contains a sample code and a demo code for reproducing Fig. 5(b) of the reference paper.

  ## Usage
  Some python modules, such as numpy, scipy, matplotlib, and networkx, are required to run the codes.
  
  ## Developer
  Gouhei Tanaka, International Research Center for Neurointelligence (IRCN), The University of Tokyo
  
  ## Citation
  G. Tanaka, T. Matsumori, H. Yoshida, K. Aihara, "Reservoir computing with diverse timescales for prediction of multiscale dynamics," Physical Review Research, vol.4, L032014 (2022)
