#Menggunakan library standar untuk classification
import pandas as pd
import numpy as np

#ImportAI_ naive bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, OneHotEncoder

#Visualisasi hasil
from sklearn.metrics import accuracy_score

def naive(df):
    #Menentukan target yang akan diambil
    xTarget = df.drop(['Kartu Kredit'],axis = 1)
    print(xTarget)
    #Target classification dari yang diambil
    yTarget = df['Kartu Kredit']
    print("\n")
    print(yTarget)
    #Merubah nilai yang diambil menjadi nilai biner
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(yTarget)
    print("\n")
    print(Y)
    #Merubah nilai atribut menjadi index nilai
    tfidf_transformer = OneHotEncoder()
    X = tfidf_transformer.fit_transform(xTarget)
    print("\n")
    print(X)
    print(X.shape)
    #Membuat data dari dataset
    X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size= 0.3, random_state= 1)
    print("\n")
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    #Melakukan pembuatan nilai dengan naive bayes
    NB = MultinomialNB().fit(X_train, np.ravel(Y_train, order = 'C'))
    print("\n")
    print(NB)
    #Prediksi terhadap nilai yang telah dibuat
    Prediksi = NB.predict(X_test)
    Akurasi = accuracy_score(Y_test,Prediksi)
    print("\n")
    print(Prediksi)
    print(Akurasi)
    
def data_test(jk,pd,bp,ku):
    #Menghitung nilai total dari data
    Total_DT = df.shape[0]
    print(Total_DT)
    Total_True = df[df['Kartu Kredit'].isin(['True'])].shape[0]
    Total_False = df[df['Kartu Kredit'].isin(['False'])].shape[0]
    print(Total_True)
    print(Total_False)
    
def langkah_1(jk,pd,bp,ku):
    #Data diambil
    JK_T = df[df['Jenis Kelamin'].isin([JK])]
    PD_T = df[df['Pendidikan'].isin([PD])]
    BP_T = df[df['Bidang Pekerjaan'].isin([BP])]
    KU_T = df[df['Kelompok Usia'].isin([KU])]
    print("Sesuai jenis kelamin")
    print(JK_T)
    print("\nSesuai pendidikan")
    print(PD_T)
    print("\nSesuai bidang pekerjaan")
    print(BP_T)
    print("\nSesuai kelompok usia")
    print(KU_T)
    print("Menghitung nilai True dan False")
    #Mengambil nilai True dan False dari Jenis Kelamin data baru
    T_JK = JK_T[JK_T['Kartu Kredit'].isin(['True'])].shape[0] #Nilai True
    F_JK = JK_T[JK_T['Kartu Kredit'].isin(['False'])].shape[0] #Nilai False
    print("Jenis Kelamin :")
    print(T_JK)
    print(F_JK)
    #Mengambil nilai True dan False dari Pendidikan data baru
    T_PD = PD_T[PD_T['Kartu Kredit'].isin(['True'])].shape[0] #Nilai True
    F_PD = PD_T[PD_T['Kartu Kredit'].isin(['False'])].shape[0] #Nilai False
    print("Pendidikan :")
    print(T_PD)
    print(F_PD)
    #Mengambil nilai True dan False dari Bidang Pekerjaan data baru
    T_BP = BP_T[BP_T['Kartu Kredit'].isin(['True'])].shape[0] #Nilai True
    F_BP = BP_T[BP_T['Kartu Kredit'].isin(['False'])].shape[0] #Nilai False
    print("Bidang Pekerjaan :")
    print(T_BP)
    print(F_BP)
    #Mengambil nilai True dan False dari Kelompok Usia data baru
    T_KU = KU_T[KU_T['Kartu Kredit'].isin(['True'])].shape[0] #Nilai True
    F_KU = KU_T[KU_T['Kartu Kredit'].isin(['False'])].shape[0] #Nilai False
    print("Kelompok Usia :")
    print(T_KU)
    print(F_KU)
    #Memasukkan datasesuai dengan yang baru dimasukkan
    print("Jumlah elemen tabel :")
    Total_DT = df.shape[0]
    print(Total_DT)
    print("Jumlah elemen True :")
    Total_True = df[df['Kartu Kredit'].isin(['True'])].shape[0]
    print(Total_True)
    print("Jumlah elemen False :")
    Total_False = df[df['Kartu Kredit'].isin(['False'])].shape[0]
    print(Total_False)
    Hasil_True = (Total_True/Total_DT)*(T_JK/Total_DT)*(T_PD/Total_DT)*(T_BP/Total_DT)*(T_KU/Total_DT)
    Hasil_False = (Total_False/Total_DT)*(F_JK/Total_DT)*(F_PD/Total_DT)*(F_BP/Total_DT)*(F_KU/Total_DT)
    print("Hasil penjumlahan nilai True:")
    print(Hasil_True)
    print("Hasil penjumlahan nilai False :")
    print(Hasil_False)
    if Hasil_True > Hasil_False :
        Hasil = 'True'
        print("Nilai untuk kartu kredit bagi data baru yaitu")
        DT.loc[1] = [JK,PD,BP,KU,Hasil]
        print(DT)
        df.loc[8] = [JK,PD,BP,KU,Hasil]
        print(df)
    else:
        Hasil = 'False'
        print("Nilai untuk kartu kredit bagi data baru yaitu")
        DT.loc[1] = [JK,PD,BP,KU,Hasil]
        print(DT)
        df.loc[8] = [JK,PD,BP,KU,Hasil]
        print(df)

#Dataset Penerima Kartu Kredit
df = pd.DataFrame(columns=['Jenis Kelamin','Pendidikan','Bidang Pekerjaan','Kelompok Usia','Kartu Kredit'])
df.loc[1]=['Wanita','S2','Pendidikan','Tua','True']
df.loc[2]=['Pria','S1','Marketing','Muda','True']
df.loc[3]=['Wanita','SMA','Wirausaha','Tua','True']
df.loc[4]=['Pria','S1','Professional','Tua','True']
df.loc[5]=['Pria','S2','Professional','Muda','False']
df.loc[6]=['Pria','SMA','Wirausaha','Muda','False']
df.loc[7]=['Wanita','SMA','Marketing','Muda','False']
print(df)

##Menghitung prediksi dan akurasi
naive(df)

#Inputan untuk data testing
JK = input("Jenis Kelamin :")
PD = input("Pendidikan :")
BP = input("Bidang Pekerjaan :")
KU = input("KelompokUsia :")
DT = pd.DataFrame(columns=['Jenis Kelamin','Pendidikan','Bidang Pekerjaan','Kelompok Usia','Kartu Kredit'])
DT.loc[1] = [JK,PD,BP,KU,'']
print(DT)
test = data_test(JK,PD,BP,KU)
print(test)
langkah_1(JK,PD,BP,KU)
naive(df)
