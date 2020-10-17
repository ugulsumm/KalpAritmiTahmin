import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.datasets import make_blobs

# Veri setinin yüklenmesi
dataset = pd.read_csv('cleveland.csv')
dataset.info()
# Pandalar dataframe.info()işlevi, veri çerçevesinin özlü bir özetini almak için kullanılır. Verilerin keşif analizi yaparken gerçekten kullanışlı geliyor.
print(dataset.shape)  # veri stinin boyutu
print(dataset.describe())  # veri setinin istatistiksel özeti

rcParams['figure.figsize'] = 20, 14
plt.matshow(dataset.corr())
plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
plt.colorbar()
plt.show()

print(dataset.hist())

rcParams['figure.figsize'] = 8, 6
plt.bar(dataset['target'].unique(), dataset['target'].value_counts(), color=['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.show()

dataset = pd.get_dummies(dataset, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
print(dataset)  # veri setinin içeriği


# Machine learning için
y = dataset['target']
X = dataset.drop(['target'], axis=1)

X, y = make_blobs(n_samples=100, centers=2, n_features=13, random_state=1)


# veri setinin eğitim ve test verileri olarak ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# sıra, farklı modelleri uygulayıp, crossvalidation sonuçlarını karşlaştırarak en uygun modeli seçmekte

# modellerin listesinin oluşturulması
models = [
    ('LR', LogisticRegression()),

]
# Modeller için 'cross validation' sonuçlarının  yazdırılması
# K-kat çapraz doğrulama(CV) verileri kıvrımlara bölerek ve her katın bir noktada bir test seti olarak kullanılmasını sağlar.
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

print()
print()

# Uygun algoritmanın seçilmesi ve tahmin yapılması
# confusion matrix(karşılık matrisi):Karışıklık matrisi, bir sınıflandırıcı tarafından doğru tahmin edilen ve yanlış tahmin edilen değerleri görüntüler.
# Karışıklık matrisinden TP ve TN'NİN toplamı, sınıflandırıcı tarafından doğru sınıflandırılmış girişlerin sayısıdır

print('LogisticRegression:')
lr = LogisticRegression()
lr.fit(X_train, y_train)  # Modeli Eğitme
predictions_LR = lr.predict(X_test)   # Test Seti ile Hedef sınıfları tahmin etme
accuary_LR=accuracy_score(y_test, predictions_LR)
print('accuracy degeri :',accuary_LR) # accuary:doğruluk
print(confusion_matrix(y_test, predictions_LR))  # hata matrisi
print(classification_report(y_test, predictions_LR))   # Ana sınıflandırma metriklerini gösteren bir metin raporunun oluşturulması
print()
print()


#Modeli kaydetmek
filename = 'finalized_model.sav'
pickle.dump(lr, open(filename, 'wb'))

#Zaman geçtikten sonra...

#Modeli diskten geri yükle
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
print()
print()








