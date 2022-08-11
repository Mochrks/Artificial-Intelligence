
#library
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

#Hasil
from sklearn.metrics import accuracy_score

def naive(df):
    #Menentukan target 
    xTarget = df.drop(['ISPA'],axis = 1)
    print(xTarget)
    #Target classification 
    yTarget = df['ISPA']
    print("\n")
    print(yTarget)
    #Merubah nilai 
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(yTarget)
    print("\n")
    print(Y)
    #Merubah nilai atribut R
    tfidf_transformer = OneHotEncoder()
    X = tfidf_transformer.fit_transform(xTarget)
    print("\n")
    print(X)
    print(X.shape)
    #Membuat data dari dataset yang ada
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
    
def data_test(um,bb,dm,lp,bt,sn):
    #Menghitung nilai total dari data
    Total_DT = df.shape[0]
    print(Total_DT)
    Total_True = df[df['ISPA'].isin(['True'])].shape[0]
    Total_False = df[df['ISPA'].isin(['False'])].shape[0]
    print(Total_True)
    print(Total_False)
    
def langkah_1(um,bb,dm,lp,bt,sn):
    #Data yang diambil
    UM_T = df[df['Umur'].isin([UM])]
    BB_T = df[df['Berat badan'].isin([BB])]
    DM_T = df[df['Demam'].isin([DM])]
    LP_T = df[df['Lama pilek'].isin([LP])]
    BT_T = df[df['Batuk'].isin([BT])]
    SN_T = df[df['Sesak napas'].isin([BT])]
    
    print("==========================================================")
    print("Sesuai   Kelompok Umur")
    print("==========================================================")
    print(UM_T)
    print("\nSesuai Berat Badan")
    print("==========================================================")
    print(BB_T)
    print("\nSesuai Demam")
    print("==========================================================")
    print(DM_T)
    print("\nSesuai Lama pilek")
    print("==========================================================")
    print(LP_T)
    print("\nSesuai batuk")
    print("==========================================================")
    print(BT_T)
    print("\nSesuai Sesak napas")
    print("==========================================================")
    print(SN_T)
    print("\n")
    print("==========================================================")
    print("Perhitungan nilai True dan False")
    print("==========================================================")

    #Mengambil nilai True dan False dari kelompok umur 
    T_UM = UM_T[UM_T['ISPA'].isin(['True'])].shape[0] #Nilai True
    F_UM = UM_T[UM_T['ISPA'].isin(['False'])].shape[0] #Nilai False
    print("Umur :")
    print(T_UM)
    print(F_UM)
    #Mengambil nilai True dan False dari Berat badan
    T_BB = BB_T[BB_T['ISPA'].isin(['True'])].shape[0] #Nilai True
    F_BB = BB_T[BB_T['ISPA'].isin(['False'])].shape[0] #Nilai False
    print("Berat badan :")
    print(T_BB)
    print(F_BB)
    #Mengambil nilai True dan False dari Demam
    T_DM = DM_T[DM_T['ISPA'].isin(['True'])].shape[0] #Nilai True
    F_DM = DM_T[DM_T['ISPA'].isin(['False'])].shape[0] #Nilai False
    print("Demam  :")
    print(T_DM)
    print(F_DM)
    #Mengambil nilai True dan False dari Lama pilek
    T_LP = LP_T[LP_T['ISPA'].isin(['True'])].shape[0] #Nilai True
    F_LP = LP_T[LP_T['ISPA'].isin(['False'])].shape[0] #Nilai False
    print("Lama pilek :")
    print(T_LP)
    print(F_LP)
    #Mengambil nilai True dan False dari  batuk
    T_BT = BT_T[BT_T['ISPA'].isin(['True'])].shape[0] #Nilai True
    F_BT = BT_T[BT_T['ISPA'].isin(['False'])].shape[0] #Nilai False
    print("Dari batuk :")
    print(T_BT)
    print(F_BT)
    #Mengambil nilai True dan False dari  Sesak napas
    T_SN = SN_T[SN_T['ISPA'].isin(['True'])].shape[0] #Nilai True
    F_SN = SN_T[SN_T['ISPA'].isin(['False'])].shape[0] #Nilai False
    print("Sesak napas :")
    print(T_SN)
    print(F_SN)
    #Memasukkan data sesuai dengan yang baru dimasukkan
    print("Jumlah elemen tabel dataset :")
    Total_DT = df.shape[0]
    print(Total_DT)
    print("Jumlah elemen True dataset :")
    Total_True = df[df['ISPA'].isin(['True'])].shape[0]
    print(Total_True)
    print("Jumlah elemen False dataset :")
    Total_False = df[df['ISPA'].isin(['False'])].shape[0]
    print(Total_False)
    #perhitungan
    Hasil_True = (Total_True/Total_DT)*(T_UM/Total_True)*(T_BB/Total_True)*(T_DM/Total_True)*(T_LP/Total_True)*(T_BT/Total_True)*(T_SN/Total_True)
    Hasil_False = (Total_False/Total_DT)*(F_UM/Total_False)*(F_BB/Total_False)*(F_DM/Total_False)*(F_LP/Total_False)*(F_BT/Total_False)*(F_SN/Total_False)
    print("====================================================================")
    print("Hasil penjumlahan nilai True pada dataset:")
    print(Hasil_True)
    print("Hasil penjumlahan nilai False pada dataset :")
    print(Hasil_False)
    print("=====================================================================")
    if Hasil_True > Hasil_False :
        Hasil = 'True'
        print("==================================================================")
        print("Nilai ISPA data inputan baru adalah")
        DT.loc[1] = [UM,BB,DM,LP,BT,SN,Hasil]
        print(DT)
        df.loc[51] = [UM,BB,DM,LP,BT,SN,Hasil]
        print(" Hasil yang didapat bernilai true ,Maka hasilnya kemungkinan terjangkit penyakit ISPA")
        print("\n")
        print(df)
    else:
        Hasil = 'False'
        print("Nilai ISPA data inputan baru adalah")
        DT.loc[1] = [UM,BB,DM,LP,BT,SN,Hasil]
        print(DT)
        df.loc[51] = [UM,BB,DM,LP,BT,SN,Hasil]
        print ("Hasil yang di dapat bernilai false , Maka hasilnya tidak terjangkit penyakit ISPA")
        print("\n")
        print(df)
        print("=================================================================")
        
        
#Dataset Penyakit ispa
print("============================ Dataset ISPA ==============================")
print("========================================================================")
df = pd.DataFrame(columns=['Umur','Berat badan','Demam','Lama pilek','Batuk','Sesak napas','ISPA'])
df.loc[1]=['Remaja','C','Ya','6','Ya','Ya','True']
df.loc[2]=['Remaja','C','Tidak','3','Tidak','Ya','False']
df.loc[3]=['Kanak-Kanak','B','Tidak','3','Ya','Tidak','False']
df.loc[4]=['Remaja','D','Ya','4','Tidak','Tidak','False']
df.loc[5]=['Remaja','D','Ya','4','Ya','Tidak','False']
df.loc[6]=['Kanak-kanak','B','Tidak','4','Tidak','Ya','False']
df.loc[7]=['Dewasa','D','Tidak','7','Ya','Ya','True']
df.loc[8]=['Lansia','C','Ya','6','Ya','Ya','True']
df.loc[9]=['Remaja','D','Ya','7','Ya','Tidak','True']
df.loc[10]=['Dewasa','D','Ya','1','Ya','Tidak','False']
df.loc[11]=['Kanak-kanak','B','Ya','7','Tidak','Tidak','True']
df.loc[12]=['Dewasa','D','Tidak','4','Tidak','Ya','True']
df.loc[13]=['Dewasa','D','Tidak','4','Tidak','Tidak','False']
df.loc[14]=['Dewasa','D','Tidak','5','Ya','Ya','True']
df.loc[15]=['Kanak-kanak','B','Ya','2','Tidak','Tidak','False']
df.loc[16]=['Lansia','C','Tidak','1','Ya','Tidak','False']
df.loc[17]=['Kanak-kanak','C','Tidak','7','Tidak','Tidak','True']
df.loc[18]=['Lansia','D','Ya','4','Tidak','Ya','True']
df.loc[19]=['Kanak-kanak','B','Ya','5','Tidak','Tidak','False']
df.loc[20]=['Dewasa','C','Tidak','1','Ya','Ya','True']
df.loc[21]=['Dewasa','C','Ya','4','Ya','Tidak','False']
df.loc[22]=['Dewasa','D','Tidak','1','Ya','Tidak','False']
df.loc[23]=['Kanak-kanak','B','Ya','1','Tidak','Tidak','False']
df.loc[24]=['Kanak-kanak','B','Ya','5','Ya','Ya','True']
df.loc[25]=['Balita','A','Ya','3','Ya','Ya','False']
df.loc[26]=['Dewasa','C','Tidak','4','Tidak','Tidak','True']
df.loc[27]=['Balita','A','Tidak','2','Ya','Tidak','True']
df.loc[28]=['Dewasa','D','Tidak','5','Ya','Tidak','False']
df.loc[29]=['Kanak-kanak','D','Tidak','5','Tidak','Ya','False']
df.loc[30]=['Dewasa','D','Ya','1','Tidak','Ya','False']
df.loc[31]=['Dewasa','D','Ya','7','Tidak','Ya','False']
df.loc[32]=['Dewasa','C','Tidak','5','Tidak','Ya','True']
df.loc[33]=['Kanak-kanak','B','Ya','6','Tidak','Tidak','False']
df.loc[34]=['Lansia','C','Tidak','3','Tidak','Ya','True']
df.loc[35]=['Kanak-kanak','C','Ya','4','Ya','Ya','True']
df.loc[36]=['Dewasa','C','Tidak','4','Tidak','Ya','False']
df.loc[37]=['Kanak-kanak','B','Ya','4','Ya','Tidak','True']
df.loc[38]=['Dewasa','D','Tidak','3','Tidak','Tidak','True']
df.loc[39]=['Dewasa','B','Tidak','7','Tidak','Tidak','False']
df.loc[40]=['Kanak-kanak','B','Tidak','1','Ya','Tidak','True']
df.loc[41]=['Remaja','D','Tidak','1','Ya','Tidak','Tidak']
df.loc[42]=['Lansia','C','Tidak','6','Tidak','Tidak','True']
df.loc[43]=['Remaja','C','Ya','1','Ya','Ya','True']
df.loc[44]=['Remaja','C','Ya','7','Tidak','Tidak','True']
df.loc[45]=['Kanak-kanak','B','Tidak','2','Ya','Tidak','True']
df.loc[46]=['Remaja','C','Ya','1','Tidak','Ya','False']
df.loc[47]=['Remaja','D','Tidak','7','Tidak','Tidak','True']
df.loc[48]=['Lansia','C','Ya','2','Tidak','Ya','False']
df.loc[49]=['Lansia','C','Tidak','5','Ya','Tidak','True']
df.loc[50]=['Kanak-kanak','B','Tidak','4','Ya','Ya','True']

print(df)
print("=========================================================================")

#Inputan data/menerima masukan data(nilai)yang akan diprediksi
UM = input("Masukan Kelompok Umur : ")
BB = input("Masukan Berat Badan : ")
DM = input("Apakah Demam : ")
LP = input("Lama pilek (1 - 7 hari) : ")
BT = input("Apakah mengalami batuk  : ")
SN = input("Apakah mengalami sesak napas : ")
print("===============================================================")
DT = pd.DataFrame(columns=['Umur','Berat Badan','Demam','Lama pilek','Batuk','Sesak napas','ISPA'])
DT.loc[1] = [UM,BB,DM,LP,BT,SN,'']
print(DT)
test = data_test(UM,BB,DM,LP,BT,SN)
print(test)
langkah_1(UM,BB,DM,LP,BT,SN)
