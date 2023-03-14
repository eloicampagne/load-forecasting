# Electricity Demand Forecasting during COVID-19 Lockdowns

The coronavirus disease 2019 (COVID-19) pandemic has caused significant changes in electricity consumption patterns due to the strict lockdown measures implemented by many governments worldwide. 

## Project Description

This project aims to reproduce a solution developed in [this paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9382417) to the issue of poor performances exhibited by traditional electricity load forecasting models since the beginning of the pandemic of COVID-19. These models are trained on historical data and rely on calendar or meteorological information.

We are only focusing on fine-tuning which allows the model to quickly adapt to new consumption patterns without requiring exogenous information.

The developed models are applied to forecast the electricity demand during the French lockdown period, and expert aggregation is used to leverage the specificities of each prediction and enhance results even further.

## Data

The data used in this project is publicly available and can be found in the data directory. 


* `Température` : temperature in Celcius
* `Temp95` and `Temp99` : exponentially smoothed temperatures of factors .95 and .99
* `TempMin99` and `TempMax99` : minimal and maximal value of `Temp99` at the current day
* `Consommation` : electricity consumption in MW
* `Consommation1` and `Consommation7` : consumptions of the day before and the week before
* `DateN` : number of the day since the beginning of the dataset
* `TimeOfYear` : time of year (0 = 1st of January at 00:00, 1 = 31st of December at 23:30)
* `DayType` : categorical variable indicating the type of the day (0 = Monday, 6 = Sunday)
* `DLS` : binary variable indicating whether it is summertime or not

## Getting started

*We chose to develop this project in both Python and R to get the best out of the already developed libraries.*

### Prerequisites

Before you start, make sure you have the following packages installed on Python :

* `numpy`
* `pandas`
* `torch`
* `tqdm`

and on R :
* `opera`
* `mgcv`
* `caret`
* `riem`

### Cloning the repository

To clone the repository, run the following command in your terminal 

```
git clone git@github.com:Exion35/load-forecasting.git
```

or

```
git clone https://github.com/Exion35/load-forecasting.git
```

## Navigating through the repository

1. Get the Italian weather data with `get_it_weather.ipynb`
2. Process both French and Italian data with `preprocessing.ipynb`
3. Build the experts (GAM, GAM Saturday, GBM) with `build_experts.ipynb`
4. Fine-tune the GAM and display the results with `fine_tuning_gam.ipynb`
5. Aggregate the experts with `aggregate_experts.ipynb` (you can come back to `fine_tuning_gam.ipynb` to display the aggregation plot)

## Results

Numerical Performance In MAPE (%) and RMSE (MW).

| Method                       | Test 1               | Test 2                |
|:----------------------------:|:--------------------:|:---------------------:|
| GAM                          |5.40%, 3076 MW        |3.77%, 2030 MW         |
| GBM                          |6.34%, 3483 MW        |5.04%, 2607 MW         |
| Fine-tuned                   |3.96%, 2417 MW        |3.78%, 2024 MW         |
| GAM $\delta$                 |20.68%, 10833 MW (!)  |5.06%, 2706 MW         |
| GAM $\delta$ - Fine-tuned    |-                     |-                      |   
| GAM Saturday                 |4.02%, 2520 MW        |5.78%, 3227 MW         |
| Aggregation with GAM Saturday|**2.80%**, **1661 MW**|**3.15%**, **1762 MW** |


## References

* D. Obst, J. de Vilmarest and Y. Goude, "Adaptive Methods for Short-Term Electricity Load Forecasting During COVID-19 Lockdown in France," in IEEE Transactions on Power Systems, vol. 36, no. 5, pp. 4754-4763, Sept. 2021, doi: 10.1109/TPWRS.2021.3067551.
