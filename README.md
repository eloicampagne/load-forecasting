# Electricity Demand Forecasting during COVID-19 Lockdowns

The coronavirus disease 2019 (COVID-19) pandemic has caused significant changes in electricity consumption patterns due to the strict lockdown measures implemented by many governments worldwide. 

## Project Description

This project aims to reproduce a solution developed in [this paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9382417) to the issue of poor performances exhibited by traditional electricity load forecasting models since the beginning of the pandemic of COVID-19. These models are trained on historical data and rely on calendar or meteorological information.

The solution involves two methods: Kalman filters and fine-tuning. Kalman filters are used to model the dynamic behavior of the electricity load during lockdown periods, while fine-tuning allows the model to quickly adapt to new consumption patterns without requiring exogenous information.

The developed models are applied to forecast the electricity demand during the French lockdown period, and expert aggregation is used to leverage the specificities of each prediction and enhance results even further.

## Data

The data used in this project is publicly available and can be found in the data directory. 

## References

* D. Obst, J. de Vilmarest and Y. Goude, "Adaptive Methods for Short-Term Electricity Load Forecasting During COVID-19 Lockdown in France," in IEEE Transactions on Power Systems, vol. 36, no. 5, pp. 4754-4763, Sept. 2021, doi: 10.1109/TPWRS.2021.3067551.