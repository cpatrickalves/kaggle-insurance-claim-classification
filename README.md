# Data Science Academy Kaggle Competition

This project presents a code/kernel used in a Kaggle competition promoted by [Data Science Academy](https://www.datascienceacademy.com.br/) in **December of 2019**.

The aim of this competition is to build a predictive model that can predict the probability that a particular claim will be approved immediately or not based on the resources available at the beginning of the process, helping the insurance company to accelerate the payment release process and thus provide better service to the client.

#### About the project: 
Claims should be carefully evaluated by the insurer, which may take time. As a result, I need to build a predictive model that can predict the probability that a particular claim will be approved immediately or not based on historical and anonymous data.

Historical data is classified into two classes, 0 and 1.

Class 0 indicates that the claim was not approved immediately (probably because it required further analysis). Class 1 indicates that the claim was approved immediately.

My job is not to predict whether a new order should be approved immediately, but to predict the probability of immediate approval of each claim. This allows the insurer to prioritize orders over 80% likely to be approved immediately, for example.

This is a binary classification problem, but instead of predicting classes, I am predicting probabilities.

There is an additional obstacle. The variables are not identified as they are anonymous data. 

Competition page: https://www.kaggle.com/c/competicao-dsa-machine-learning-dec-2019/

Dataset are available in competition's pages.

Files description:
* **kernel.csv** - the Jupyter Notebook file with all project work flow (data cleaning, preparation, analysis, machine learning, etc.).
* **dataset_treino.csv** - contains the training dataset with 114,321 rows (claims) and 133 columns (features).
* **dataset_teste.csv** - contains the test dataset with 114,393 rows and 132 columns.
* **sample_submission.csv** - a sample of the submission file.

The evaluation metric for this competition is Log Loss (the smaller the better).

In this competition my best score was 0.4929 and I got the position 38 on the [leaderboard](https://www.kaggle.com/c/competicao-dsa-machine-learning-dec-2019/leaderboard).