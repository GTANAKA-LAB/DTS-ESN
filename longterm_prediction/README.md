# longterm_prediction
In [Tanaka et al., Phys. Rev. Res. 2022](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.4.L032014), two tasks are performed. The second task is a long-term prediction with a closed-loop DTS-ESN model (see Fig.1(c)). The example here gives a code and a demo for reproducing Fig.5(b), where the dynamics of the Hindmarsh-Rose model is predicted. Due to the chaotic nature of the system, the prediction error eventually diverges.

  ## Files
  * demo_HR_longterm.ipynb: the demo.
  * esn_dts_closedloop.py: the DTS-ESN engine. 
  * pred_HR_longterm.py: the code.

  ## How to use
  * The code can be run with the following command:
  ```
  > python pred_HR_longterm.py
  ```
  
  The prediction result is also seen in the demo file ```demo_HR_longterm.ipynb```.
