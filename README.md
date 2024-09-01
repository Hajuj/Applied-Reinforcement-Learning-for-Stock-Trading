# Automatic Stock Trading - Applied Reinforcement Learning Sommer Term 2023

Welcome to the repository for the practical course on Applied Reinforcement Learning, offered in the summer term of 2023 at the university. In this course, we explore the fascinating field of Reinforcement Learning and its application to stock trading. The course will be undertaken by a team of four talented students: Viktoria, Mohamad, Lorena, and Jonas.

## Project Overview

The primary focus of our project is to develop an intelligent agent that leverages historical data to make informed decisions regarding stock trading. Specifically, our agent will predict whether to **sell**, **hold**, or **buy a particular stock** based on the analysis of the available data. To accomplish this, we plan to implement advanced RL algorithms such as **Advantage Actor-Critic** (A2C) and **Proximal Policy Optimization** (PPO).

For more information on the project please visit our [**Wiki**](https://github.com/Hajuj/Applied-Reinforcement-Learning-for-Stock-Trading/wiki), also check our report for more detailed information [**Report**](https://github.com/Hajuj/Applied-Reinforcement-Learning-for-Stock-Trading/blob/main/paper/ARL_Stock_Trading.pdf).

FinRL Library Installation Guide: [Installation Link](https://finrl.readthedocs.io/en/latest/start/installation.html).

## Datasets and APIs 

To facilitate our research and development process, we will utilize the following datasets and APIs:

1. **Yahoo!Finance:** This dataset will serve as our primary source of historical stock market data. It provides comprehensive and reliable information about stock prices, volumes, and various other relevant metrics. [Tutorial](https://algotrading101.com/learn/yahoo-finance-api-guide/)
2. **Alpha Vantage:** This API will supplement our analysis by providing additional financial data and metrics. We can leverage this data to enrich our models and potentially incorporate alternative data sources such as Twitter sentiment analysis. [Alpha Vantage Academy Tutorial](https://www.alphavantage.co/academy/)

For further information on these datasets and APIs, you can refer to the following resources:
- [Best Stock API - RapidAPI Blog](https://rapidapi.com/blog/best-stock-api/)

## Add-Ons Ideas

In addition to the core implementation, we have identified several exciting add-on ideas that can further enhance our project. These include:

### Robo-advisor

To expand the capabilities of our agent, we propose integrating a robo-advisor component. This would involve combining historical stock market data with alternative data sources such as **Twitter sentiment analysis**. By considering these additional factors, our agent can better understand market dynamics and make more informed trading decisions. Additionally, we can **interact with the client to determine their risk preferences** and tailor the agent's behavior accordingly.

### Market Making

Incorporating the concept of market making, we can leverage the robo-advisor as a market maker. By **utilizing the Limit Order Book**, our agent can dynamically adjust buy and sell prices to create liquidity in the market. This approach can contribute to a more efficient and stable trading environment.

### Option Pricing and Hedging

To further extend the scope of our project, we propose exploring complex derivatives, such as options, to hedge risks associated with vanilla stock trading. By incorporating **option pricing and hedging strategies**, our agent can handle more sophisticated trading scenarios and provide a more comprehensive trading solution.

## Team Members

- Lorena
- Viktoria
- Mohamad
- Jonas

Feel free to reach out to any team member if you have any questions, suggestions, or collaboration opportunities related to this project.

Let's embark on this exciting journey into the world of Reinforcement Learning for Stock Trading!

## Code - Where to find the code for the steps in our pipeline

The project requirements can be found in requirements.txt 

### Data Generation and Pre-Processing

To download the financial data use the Jupyter Notebook *Data_AlphaVantage* . It can be found in the folder named *Genereated Sentiment Data in FinRL*. 

For downloading sentiment data and interpolating the missing data points use *Data_FMP* which can be found in *Generated Sentiment Data in FinRL* as well.

The required code for smoothing the data can be found, too,  in the folder *Generated Sentiment Data in FinRL*. There it is written in the JUpyter Notebook *Data Preprocessing*.

The last step to the finished dataset is adding a randomly generatedd number to each dataset and split the finished dataset into a training- and a testing-dataset. This is done using the Jupyter Notebook *Add_Random_to_dataframe_and_split_into_training_and_testing*.

In the same directory the Jupyter Notebook *turbulenceTreshold* can be found. It contains a small piece of code for calculating the 0.8 percentile of the turbulence column of  a given dataset. This value is used to set the turbulence threshold for training and evaluation of the agents. 

### Training
The directory *Training_and_Testing* contains the directory *Envs* where the adapted Environment can be found. 

To train the agent *main.py* is used. By setting the variables like hourly, smoothed, sentiment and the used agents to either True or False the right dataframe is loaded, the correct turbulence threshold (if used) is set and the paths for logging and saving the models is set. 

The trained models can then be found in the directory *Evaluation/model* in the respective folder. 

### Evaluation

In the *Evaluation* directory the file *main_backtest* can be found. It is used for evaluating the trained models by testing them on the testing dataset. The results are saved as a csv-file and can be found in the *results_data* directory.
There the file *plots.py* can be found. It is used to combine the different datasets gained from backtesting for the different seeds and prepare thmen in a way that can be used as input for generating plots.

For the plots comparing only several agents the file *comparison plots* can be used. Depending on which feature and which agent is used for the comparison, the relevant line has to be commented in or out repectively. 

The results of the classical trading strategies and the associated plots generated in the Jupyter Notebook *Plot_Jonas-TechIndi*. 

### Sensitivity Analysis

The code for the sensitivity analysis can be found in the directory *Training_and_Testing*. 

there, the Notebook *Sensitivity_Analysis_hourly* is used for exporting the onnx model and transfering it to tensorflow and the calculation of the gradients. 

The file *daily_sensitivity_PPO* is used for creating the sensitivity plots of the PPO agent trained on daily data.

The plots for visualizing the other sensitivity results are created using the file *sensitivity_plots*.



