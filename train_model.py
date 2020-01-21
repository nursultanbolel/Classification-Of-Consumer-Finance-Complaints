# -*- coding: utf-8 -*-
"""
@author: nur sultan bolel
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout,GRU
from sklearn.model_selection import train_test_split
import re#regular expression
import numpy as np 
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from tensorflow.keras import backend
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
import keras

# -----------------------------------------------------------------------
# GPU Kontrolü: Kodun çalıştığı bilgisayarın GPU donanımının bilgilerini
# göstermek için aşağıdaki kod satırları eklenir.
# -----------------------------------------------------------------------
import tensorflow as tf
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

def _get_available_devices():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print( _get_available_devices())
# -----------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# Dataset'i projeye dahil etme: Projede kullandığımız dataset .csv formatındadır.
# Dataset içerisindeki iki sütun kullanılır(product,consumer_complaint_narrative)
# Projedeki amaç müşteri şikayetlerine(consumer_complaint_narrative) göre şikayetin
# hangi departmana(product) ait olduğunu tespit etmektir.
# Kullanılacak sütunlardan null değer içeren sütunlar dataframe'e dahil edilmez.
# ----------------------------------------------------------------------------------
df = pd.read_csv("consumer_complaints.csv")
print("\nDATAFRAME INFORMATION:\n")
print(df.info())

fields= ['product','consumer_complaint_narrative'] 
df=pd.read_csv('consumer_complaints.csv', usecols=fields)

df = df[pd.notnull(df['consumer_complaint_narrative'])]
# ----------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------
# Sınıf sayısını azaltma: Product sütunu içerisinde bazı sınıflar birleştirildi.
# Örn: 'Prepaid card' ile 
# Creadit card aynı departmanda olmalıdır. Bu iki sınıfı birleştirdik. Böylece sınıf
# sayısını azalttık ve verisi az olan sınıfın verileri arttı.
# -----------------------------------------------------------------------------------
for i in range(df.shape[0]):
    if (df.iloc[i]=="Prepaid card").any():
        df.iloc[i]="Credit card"
    if (df.iloc[i]=="Virtual currency").any():
        df.iloc[i]="Other financial service"  

print("\nPRODUCT VALUE COUNT:\n")
print(df["product"].value_counts())
print("\nDATAFRAME HEAD:\n")
print(df.head())
# -----------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# PREPROCESSING-1
# ---------------------------------------------------------------------------------
# STOPWORDS değişkeni, sınıflandırmaya etkisi olmayan kelimeri bulundurur.
# Bu çalışmada amaç müşterilerin şikayetlerini sınıflandırmak olduğundan 'bank',
# 'america' gibi kelimeler de STOPWORDS değişkenine eklendi. Noktalama işaretleri,
# tarihleri belirten ifadeler, sayılar,boşluk karakteri, X karakteri gibi ifadeler
# model eğitiminden önce consumer_complaint narrative'den çıkarıldı.
# Kullandığımız dataset ingilizce müşteri şikayetleri içerdiğinden preprocessing
# ingilizceye göre yapılmıştır.
# ---------------------------------------------------------------------------------
STOPWORDS=stopwords.words('english')
stopwords_extra=['bank', 'america', 'x/xx/xxxx', '00']
STOPWORDS.extend(stopwords_extra)

remove_caracteres = re.compile('[^0-9a-z #+_]')
replace_espaco = re.compile('[/(){}\[\]\|@,;]')
df = df.reset_index(drop=True)

# GRU NN'i sayısal değerleri input olarak aldığından sınıflar isimleri 0-9 aralığında
# numaralandı.
Y = pd.get_dummies(df["product"]).values
print("shape Y", Y.shape)

# Test aşamasında tahmin edilen sınıfı model sayı olarak çıktı vereceğinden sınıf
# isimlerini korumak için sınıf isimlerini kaydettik.
class_name = ["" for x in range(10)]
for i in range(100):
    for j in range(10):
        if(Y[i,j]==1):
            class_name[j]=df.iloc[i]['product']
np.save('class_name',class_name)            
# Modelimizin ezberlemesini önlemek için verilerimizi karıştırdık.                   
X, Y = shuffle(df['consumer_complaint_narrative'] , Y)
# Train ve test aşamalarında aynı verileri kullanmamak için verilerimizi 0.1
# oranında böldük.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state = 42)
# Test aşamasında müşterilerin şikayetlerini sayılar ile göstermemek için kaydettik.
X_test.to_csv('X_test_before_tokenizer.csv', header=True)
#Sınıflandırmatı etkilemeyecek kelimeleri,noktalama işaretlerini,karakterleri çıkardık.
def pre_processamento(text):
    text = text.lower()
    text = remove_caracteres.sub('', text)
    text = replace_espaco.sub(' ', text)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

X_train = X_train.apply(pre_processamento)
X_test = X_test.apply(pre_processamento)
# ---------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# PREPROCESSING-2
# --------------------------------------------------------------------------------------
# Projemizde GRU NN'i kullandığımızdan input değerleri sayısal olmalıdır. Dataset'i
# sayısal ifadelere çevirmek için Tokenizer() fonksiyonundan yararlandık. Bu fonksiyon
# ile en çok geçen 5000 kelimeye sayısal değerler atanır. Token sayısının küçük
# olması için tüm kelimleri küçük harflere çevirdik.
# Sayısal değerlere çevrilmiş şikayetler 250 adet sayısal değerler içerecektir.
# --------------------------------------------------------------------------------------
n_max_palavras = 5000
tamanho_maximo_sent = 250

tokenizer = Tokenizer(num_words=n_max_palavras, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['consumer_complaint_narrative'].values)
word_index = tokenizer.word_index
print(' %s tokens unicos.' % len(word_index))


X_train = tokenizer.texts_to_sequences(X_train.values)
X_train = pad_sequences(X_train, maxlen=tamanho_maximo_sent)
print("shape X_train", X_train.shape)

X_test = tokenizer.texts_to_sequences(X_test.values)
X_test = pad_sequences(X_test, maxlen=tamanho_maximo_sent)
print("shape X_test", X_test.shape)

# Test aşamasında modele sayısal değerleri input olarak vermek için kaydettik.
np.save('X_test', X_test)
np.save('Y_test', Y_test)
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# TRAINING
# --------------------------------------------------------------------------------------
# Modelimiz içerisinde 100 adet node bulunan iki tane GRU(Gated Recurrent Units) NN vardır.
# Epoch: 5
# Batch size: 128
# Dropout: 0.2
# Activation func: softmax
# Optimizer: adam
# --------------------------------------------------------------------------------------
embedding_dimensions = 100
epochs = 3
batch_size = 512

model = Sequential()
model.add(Embedding(n_max_palavras, embedding_dimensions, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(GRU(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(100))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
history=model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,shuffle=True)

# Test aşamasında kullanmak için modeli kaydettik.
model.save('my_model.h5')
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# TRAINING aşamasından sonra accuracy ve loss değerlerini görmek için hesaplamalar ve
# grafik çizimi ekledik.
# --------------------------------------------------------------------------------------
fig1 = plt.figure()
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves :RNN - GRU',fontsize=16)
plt.show()

scores = model.evaluate(X_test, Y_test, verbose=0)
print("\nAccuracy: %.2f%%" % (scores[1]*100))
# --------------------------------------------------------------------------------------