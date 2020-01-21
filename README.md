# Classification-Of-Consumer-Finance-Complaints

## Purpose of Project

Classification of consumer's complaints from .csv file. It is simple text classification study. A sample output is shown below.

![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/sample_output.PNG)

## Dataset
- Each week the CFPB sends thousands of consumers’ complaints about financial products and services to companies for response. Those complaints are published here after the company responds or after 15 days, whichever comes first. By adding their voice, consumers help improve the financial marketplace.

- I used .csv file  you can find there [Kaggle](https://www.kaggle.com/cfpb/us-consumer-finance-complaints)

- Information of .csv file is shown below in Spyder IDE.

![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/dataframe_inf.png)

- I used just two columns that are 'product' and 'consumer_complaint_narrative'. The aim is to predict the product according to consumer narrative.

- 'product' was including 13 different value. I joined same department and then now it has 10 different value. These values' information is shown below.

![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/product_inf.png)


###### Splitting Dataset

 I splited dataset for 90% training and 10% test data. 
 
 ###### Preprocessing
 
I found the words that don't change the classification. I added  these words to the STOPWORDS list so I reduced the number of tokens.

 - ['bank', 'america', 'x/xx/xxxx', '00'] 

Numbers, punctuations, special characters removed from complaints.

- [^0-9a-z #+_]
- [/(){}\[\]\|@,;] 

We reduced the number of tokens by turning all of the complaints into lowercase letters.

## Hyper Parameters

- Training samples: train=142.078 , test=15.787
- Learning Rate: 0,01
- Epoch: 5
- Batch Size: 256
- Activation Functions: softmax
- Dropout: 20%
- Number of Hidden Layer and Units: GRU(100)+GRU(100)
- Loss functions: categorical_crossentropy
- Optimizer: Adam


## Model

![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/model1.png)
![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/model2.png)

## Tried model with different parameters
![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/diff_value.PNG)

## Setup

- Computer information in test and train steps

CPU: model i7-4510U, Speed 2.00 GHz, Cache 4 MB 

RAM: 8 GB,  1600 MHz, DDR3L 

GPU: NVIDIA,  GeForce 840M,  4 GB

- Packages in working  environment

![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/packages.PNG)

## Performance Measures
![GitHub Logo](https://github.com/nursultanbolel/Classification-Of-Consumer-Finance-Complaints/blob/master/images/performance.PNG)

## References
https://www.kaggle.com/cfpb/us-consumer-finance-complaints
