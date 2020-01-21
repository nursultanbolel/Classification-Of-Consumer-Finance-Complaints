# -*- coding: utf-8 -*-
"""
@author: nur sultan bolel
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from sklearn.model_selection import train_test_split
import re#regular expression
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
from tensorflow.keras import backend
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import roc_curve, auc

#Model eğitilirken bazı bilgilerin test aşamasında kullanılması gerekiyor. 
X_test_before_tokenizer = pd.read_csv("X_test_before_tokenizer.csv")
X_test_before_tokenizer.columns = ['Index','consumer_complaint_narrative']    

X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
class_name = np.load('class_name.npy')

model = load_model('my_model.h5')
#Model çıktı olarak class numarası vermektedir. Hangi class numarasının hangi sınıfa ait olduğu
#bilgisi yazdırılır.
print ('\n###################\n#CLASS INFORMATION#\n###################')
print('class number --> class name')
for i in range(10):
    print('\n',i,'-->',class_name[i])
    
y_test=np.zeros((Y_test.shape[0],1))
for i in range(Y_test.shape[0]):
    for j in range(10):
        if(Y_test[i,j]==1):
            y_test[i]=j
#Test datası içerisinden bir şikayet modele verilir ve belenen değer ile birlikte modelin bulduğu
#class bilgisi yazdırılır            
print ('\n###########################\n#GIVING TEST DATA TO MODEL#\n###########################')
for i in range(4,10):
    print('\nCONSUMER COMPLAINT WITHOUT PREPROCESSING ===> "',X_test_before_tokenizer.iloc[i]['consumer_complaint_narrative'],'"')
    print('EXPECTED CLASS ===> ',Y_test[i],'NAME: [',class_name[int(y_test[i])],']')
    yhat = model.predict_classes(X_test[i:i+1], verbose=0)
    print('PREDICTED CLASS ===> ',yhat,'NAME:',class_name[yhat])
#•Performansı ölçmek için kullanılan değerler hesaplanır.
print ('\n######################\n#PERFORMANCE MEASURES#\n######################\n')
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#Test verilerinin tamamı modele verilir ve sonuçları yhat değişkeninde tutulur.
yhat = model.predict_classes(X_test, verbose=0)
#İki farklı RMSE değerinin hesaplanması
# rmse = sqrt(mean_squared_error(y_test, yhat))
# print("RMSE : ",rmse)
#predict the test result
# y_pred=model.predict(X_test)
# rss=((Y_test-y_pred)**2).sum()
# mse=np.mean((Y_test-y_pred)**2)
# print("Final rmse value is =",np.sqrt(np.mean((Y_test-y_pred)**2)))

y_score=np.zeros((Y_test.shape[0],Y_test.shape[1]))
for i in range(Y_test.shape[0]):
    y_score[i,yhat[i]]=1;

fpr = dict()
tpr = dict()
roc_auc = dict()

#Her bir class için TPR,FPR,AUC değerleri ayrı ayrı hesaplanır.
for i in range(0,10):
    print('\n---> CLASS ',class_name[i],'\n')
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    print('TPR :', tpr[i])
    print('FPR :',fpr[i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print('AUC : %0.2f' %np.float(roc_auc[i]))
    print('OneVsAll Approach ROC curve')
#ROC curve çizimi
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' %np.float(roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()