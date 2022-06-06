# Chocolate Segmentation (because Customer Segmentation is boring)

## Content

1. Gathering the data, [Notebook](src/request_and_save_data.ipynb)
2. Make an exploratory data analysis, [Notebook](src/EDA_and_cleansing.ipynb)
3. Cleanse and prepare the data, [Notebook](src/EDA_and_cleansing.ipynb)
4. Cluster the data, [Notebook](src/Clustering_and_Interpretation.ipynb)
5. Interpret the data, [Notebook](src/Clustering_and_Interpretation.ipynb)

## Introduction

The goal of this project is to go though an end-to-end clustering project from getting the data to use the generated insights further. An imaginary usage could be to use the clusters to build a recommender system for chocolate: "If you like _this_ chocolate, you might like _that_ ones as well." I've chosen the chocolate context because I have much more contact to chocolates than to customers or end users.

## Overview

The project is built upon data from the public API of the U.S. Department of Agriculture (https://fdc.nal.usda.gov/api-guide.html). The API documentation can be found [here](https://app.swaggerhub.com/apis/fdcnal/food-data_central_api/1.0.0). The data is accessed via REST API and stored in a local PostgreSQL database. But you can see the raw data from the API [here](data/api_raw_data.csv).  
Next steps are performing an exploratory data analysis as a foundation for the data cleansing step. And the data cleansing itself, including a wide range of adjustments, e.g. extracting data in lists. The cleansed and fully prepared data is stored again in the database. A cleansed but no-encoded version of the data for visualization or other projects can be found [here as csv](data/cleaned_data.csv).  
Last part is the clustering itself. Two algorithms, KMeans and DBSCAN were used on different subsets of the dataset based on the first clustering results. For both a comprehensive hyperparamter tuning was implemented in order to get a optimal result.

## Results

Unfortunately, both clustering algorithms were not able to define meaningful clusters. In summary, there is one big cluster aka. no cluster. For more information, please see the clustering notebook.  
Nevertheless, I still believe one can find interesting information through visualization.
