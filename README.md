# Efficient Time Sampling Strategy for TAS

This repository is the official implementation of "Efficient Time Sampling Strategy for Transient Absorption Spectroscopy" by Juhyeon Kim, Joshua Multhaup, Mahima Sneha, Adithya Pediredla (ICCP 2024).


## Run simulation
To run simulation, please go to `src` folder and run following code.
```
python main_simulation.py expnumber
```

Expnumber could be one of the followings.

- 3 (or fig3) : Generate plots for figure 3. (comprehensive comparison) 
- 4 (or fig4) : Generate plots for figure 4. (using different initial taus)
- 5 (or fig5) : Generate plots for figure 5. (using heteroscedastic noise)
- 6 (or fig6) : Generate plots for figure 6. (using linear curve fitting)
 

You can find output in `result` folder.
We already included the plots, so please check them.

 ## Real-data
TBA

## Citation
If you find this useful for your research, please consider to cite:
```
TBA
```