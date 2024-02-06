## **PrCRS:** **A prediction model of severe CRS in CAR-T therapy based on transfer learning**

​	PrCRS is a deep learning prediction model based on U-net and Transformer. Given the limited data available for CAR-T patients, we employ transfer learning using data from COVID-19 patients. We propose six models to forecast the probability of severe CRS for patients with one, two, and three days in advance. Additionally, we present a strategy to convert the model's output into actual probabilities of severe CRS and provide corresponding predictions.

## Network

![network](https://github.com/wzy38828201/PrCRS/blob/main/network.png)

## Train and Inference

#### 1.download

Git clone https://github.com/wzy38828201/PrCRS.git

#### 2.requirements

```
Python 3.7.6
sklearn 0.22.1
numpy 1.18.5
scipy 1.4.1
pandas 0.24.2
dgl 0.9.1
torch 1.11.0+cu113
torch-geometric 2.0.4
dgl-cuda11.0 0.9.0 py37_0
torch-scatter 2.0.9
torch-sparse 0.6.13
```

#### 3.Usage

It is divided into two folders "main" and "Web source code", where:
I. The "main" folder contains the transfer learning model used for prediction, including 1, 2, and 3 days ahead of the forecast.

1. Before running, run "preprocess_data1.py" to preprocess the COVID-19 data, and then run "train_data1.py" to train the COVID-19 data, and save the best trained model to the corresponding position. Then, you can run "preprocess_data2.py", the program is the pre-processing of CAR-T data. Then, run "train_data2.py", which is the main program for the model, where you can choose whether to call the saved model run by "train_data1.py" for migration. After running and saving the calculated model to the appropriate location, run the "test.py" program, which is used to test the data.

2. The Test9 folder contains evaluations of additional data sets performed by nine people. The model was retrained using a sample size of 193 individuals and subsequently assessed on the remaining nine participants.

3. In the "Actual probability" folder is the method of converting the predicted probability into the actual probability

4. In the "Model comparison" folder are performance comparisons between different models

5. The drawing program is in "draw". Example data for CAR-T, COVID-19, and secondary processing are in the Data folder. The "output1,2" folder is used to store the output of the model

## Online web service

The forecast page on our website is bilingual and can be accessed via the link http://prediction.unicar-therapy.com/index-en.html.