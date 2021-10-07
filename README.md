# DTS-ESN
The echo state network (ESN) is a special recurrent neural network model proposed by H. Jaeger in 2001, which is a representative model for reservoir computing.
The DTS-ESN ([Tanaka et al., arXiv:2108.09446](https://arxiv.org/abs/2108.09446)) is an extended ESN model with diverse timescales for prediction of multiscale dynamics.
The python codes for the model and demonstrations are provided.  

  ## Usage
  * esn_dts.py: the main DTS-ESN engine
  * pred_tcLorenz.py: the source file for demonstrations of time series prediction with the two-coupled Lorenz system
  * demo_pred_tcLorenz.ipynb: a demonstration of "pred_tcLorenz.py" with jupyter notebook
