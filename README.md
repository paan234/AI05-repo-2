# Diamond Price Prediction

## 1. Summary of the project
The goal of this project is to predict the price of diamond, based on the following features:
- 4C (cut, clarity, color, carat)
- Dimensions (x, y, z)
- Depth
- Table

The model is trained with dataset from [Diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds).
This classic dataset contains the prices and ohter atributes of almost 54,000 diamonds. It's a great dataset for beginners learning to work with data analysis and visualization.

## 2. IDE and Framework
This project is created using Spyder as the main IDE. The main frameworks used in this project are:
- Pandas
- Scikit-learn
- TensorFlow Keras

## 3. Methodology

### _3.1 Data Pipeline_
The data is first loaded and preprocessed, such that unwanted features are removed. Categorical features are encoded ordinally. 
Then the data is split into train-validation-test sets, with a ratio of 60:20:20.

### _3.2 Model Pipeline_
A feedforward neural network is constructed that is catered for classification problem. The structure of the mofel is fairy simple. 
Figure below shows the structure of the model.

![alt text](https://github.com/paan234/AI05-repo-2/blob/master/Image/Model.png)

The model is trained with a batch size of 64 and for 100 epochs. Early stopping is applied in this training. 
The training stops at epoch 21, with a training MAE of 636 and validation MAE of 372. The two figures below show the graph of the training process,
indicating the convergence of model training.

![alt text](https://github.com/paan234/AI05-repo-2/blob/master/Image/loss_graph.png)

![alt text](https://github.com/paan234/AI05-repo-2/blob/master/Image/mae_graph.png)

## 4. Results
Upon evaluating the model with test data, the model obtain the following test results, as shown in figure below.

![alt text](https://github.com/paan234/AI05-repo-2/blob/master/Image/Test_result.jpg)

The model is also used to made prediction with test data. A graph of prediction vs label is plotted, as shown in the image below.

![alt text](https://github.com/paan234/AI05-repo-2/blob/master/Image/Result.png)

Based on the graph, a clear trendline of y=x can be seen, indicating the predictions are fairly similar as labels.
However, several outlier can be seen in the graph.
