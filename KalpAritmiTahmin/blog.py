from flask import Flask,render_template,request
import kalp
from numpy import *
import pickle   #kaydedilmiş modeli kullanmak için

app = Flask(__name__)

@app.route('/')
def entry_page() -> 'html':
    return render_template('degerler.html', page_title='Kalp Aritmi Tespit Sayfası')

@app.route('/degeral', methods=['POST'])
def sum() -> 'html':


    a = int(request.form['age'])
    b = int(request.form['trestbps'])
    c = int(request.form['chol'])
    d = int(request.form['thalach'])
    e = float(request.form['oldpeak'])
    f = int(request.form['ca'])

    g = int(request.form['sex'])

    if(g==0):
        cins='Kadın'
    else:
        cins='Erkek'

    h = int(request.form['cp'])
    if(h==1):
        agriTipi='typical angina'
    elif(h==2):
        agriTipi='atypical angina'
    elif (h == 3):
        agriTipi = 'non - anginal pain'
    elif (h == 4):
        agriTipi = 'asymptotic'

    k = int(request.form['fbs'])
    if (k == 0):
        sekhas = 'şeker hastası değil'
    elif (k == 1):
        sekhas = 'şeker hastası'

    m = int(request.form['restecg'])
    if(m==0):
        elektrocard='normal'
    elif(m==1):
        elektrocard='having ST-T wave abnormality'
    elif (m == 2):
        elektrocard= 'left ventricular hyperthrophy'

    n = int(request.form['exang'])
    if (n == 0):
        gagrisi = 'göğüs ağrısı yok'
    elif (n == 1):
        gagrisi = 'göğüs ağrısı var'

    o = int(request.form['slope'])
    if (o == 1):
        stSegmenti = 'upsloping'
    elif (o == 2):
        stSegmenti = 'flat'
    elif (o == 3):
        stSegmenti = 'downsloping'

    p = int(request.form['thal'])
    if (p == 1):
        hasar = 'normal'
    elif (p == 2):
        hasar = 'fixed defect'
    elif (p == 3):
        hasar = 'reversable defect'



    r = request.form['isim']

    degerler = [a, b, c, d, e, f, g, h, k, m, n, o, p]
    print(degerler)

    #alınan değerleri dosyaya yazıyoruz
    sayim = 1
    dosya = open('degerler.csv', 'w')

    for i in degerler:
        deger = i
        degerstr = str(deger)
        print(degerstr)
        dosya.write(degerstr)
        if sayim < 13:
            dosya.write(",")
            sayim = sayim + 1
    dosya.close()


    liste2 = [degerler]  # iki boyutlu liste oluşturduk
    print(liste2)

    # prediction = model.predict(liste2)
    # print('LR TAHMİNİ')
    # print(prediction)


    ynew = kalp.lr.predict(liste2)
    print("X=%s, Predicted=%s" % (liste2[0], ynew[0]))
    if ynew[0] == 1:
        sonucLR = 'aritmi tespit edildi'
    elif ynew[0] == 0:
        sonucLR = 'aritmi tespit edilmedi'
    print(sonucLR)
    dogrulukLR = kalp.accuary_LRs

    ynewNB = kalp.nb.predict(liste2)
    print("X=%s, Predicted=%s" % (liste2[0], ynew[0]))
    if ynewNB[0] == 1:
        sonucNB = 'aritmi tespit edildi'
    elif ynewNB[0] == 0:
        sonucNB = 'aritmi tespit edilmedi'
    print(sonucNB)
    dogrulukNB = kalp.accuary_NBs

    ynewKNN = kalp.knn.predict(liste2)
    print("X=%s, Predicted=%s" % (liste2[0], ynew[0]))
    if ynewKNN[0] == 1:
        sonucKNN = 'aritmi tespit edildi'
    elif ynewKNN[0] == 0:
        sonucKNN = 'aritmi tespit edilmedi'
    print(sonucKNN)
    dogrulukKNN = kalp.accuary_KNNs

    ynewDTC = kalp.dtc.predict(liste2)
    print("X=%s, Predicted=%s" % (liste2[0], ynew[0]))
    if ynewDTC[0] == 1:
        sonucDTC = 'aritmi tespit edildi'
    elif ynewDTC[0] == 0:
        sonucDTC = 'aritmi tespit edilmedi'
    print(sonucDTC)
    dogrulukDTC = kalp.accuary_DTCs




    return render_template('result.html', page_title='Calculation result',

                           yas=a, dinkanbas=b, kolesterol=c, maxkalpr=d, st=e, anadamarsay=f,
                           cinsiyet=cins, agritipi=agriTipi, sekerhastaligi=sekhas, ecg=elektrocard,
                           gogusagrisi=gagrisi, stsegmenti=stSegmenti, hasarorani=hasar,
                           ad=r, durumLR=sonucLR, dogruluk1=dogrulukLR, durumNB=sonucNB, dogruluk2=dogrulukNB,
                           durumKNN=sonucKNN, dogruluk3=dogrulukKNN, durumDTC=sonucDTC, dogruluk4=dogrulukDTC,

                           )


@app.route("/about")
def about():
    return render_template("about.html")
if __name__ == "__main__":
    app.run(debug=True)
