# 1step_ahead_prediction
In [Tanaka et al., Phys. Rev. Res. 2022](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.L032014), two tasks are performed using the proposed DTS-ESN. The first task is a one-step-ahead prediction with an open-loop DTS-ESN model (see Fig.1(b)). The example here gives a code and a demo for reproducing Fig.S2(e), where the dynamics of the whole two-coupled Lorenz model is predicted only from its fast dynamics.

  ## Files
  * demo_tcLorenz_1step.ipynb: the demo.
  * esn_dts_openloop.py: the DTS-ESN engine. 
  * pred_tcLorenz_1step.py: the code.

  ## Required modules
  * Numpy, Scipy, Matplotlib, Networkx
  
  ## How to use
  * Run the code ```pred_tcLorenz_1step.py``` in the jupyter notebook.
  * The prediction result can be seen in ```demo_tcLorenz_1step.ipynb```.
