import pandas as pd
import datetime as dt
import scipy.stats as st
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

#######################################################################
# Görev 1:  
#######################################################################

df = pd.read_csv("Datasets/amazon_review.csv")
df.head()

# Adım 1:   Ürünün ortalama puanını hesaplanması.

df["overall"].mean()
# 4.587589013224822

# Adım 2:  Tarihe göre ağırlıklı puan ortalamasının hesaplanması

df["reviewTime"] = pd.to_datetime(df["reviewTime"])

current_date = df["reviewTime"].max()

# Adım 2:  Tarihe göre ağırlıklı puan ortalamasını hesaplanması.
#   -reviewTime değişkenini tarih değişkeni olarak tanıtmanız
#   - reviewTime'ın max değerini current_date olarak kabul etmeniz
#   -her bir puan-yorum tarihi ile current_date'in farkını gün cinsinden
#     ifade ederek yeni değişken oluşturmanız ve gün cinsinden ifade edilen
#     değişkeni quantile fonksiyonu ile 4'e bölüp (3 çeyrek verilirse 4 parça çıkar)
#     çeyrekliklerden gelen değerlere göre ağırlıklandırma yapmanız gerekir.
#     Örneğin q1 = 12 ise ağırlıklandırırken 12 günden az süre önce yapılan
#     yorumların ortalamasını alıp bunlara yüksek ağırlık vermek gibi.

df["days"] = (current_date-df["reviewTime"]).dt.days

df["days"].quantile([0.25, 0.5, 0.75])

# 0.25  -->  280.0
# 0.50  -->  430.0
# 0.75  -->  600.0

df[df["days"] < 280]["overall"].mean()
df[(df["days"] > 280) & (df["days"] < 430)]["overall"].mean()
df[(df["days"] > 430) & (df["days"] < 600)]["overall"].mean()
df[df["days"] > 600]["overall"].mean()


# son zamanlarda yapılan değerlendirmelere bakarak ortalama hesapladık.

average_rating = ((df[df["days"] < 280]["overall"].mean())*40/100 ) + \
                 ((df[(df["days"] > 280) & (df["days"] < 430)]["overall"].mean())*20/100 )+ \
                 ((df[(df["days"] > 430) & (df["days"] < 600)]["overall"].mean())*20/100 )+\
                 ((df[df["days"] > 600]["overall"].mean())*20/100 )
# 4.608908707530874

# Adım 3:  time_based_weighted_average fonksiyonunu ile, day_diff'i
# gün sayısına göre veya quartile değerlerine göre parçalayıp ağırlıklandırdım.

df["day_diff"]

def time_based_average(dataframe, w1=40 , w2 = 20  , w3 = 20, w4 = 20):
    q = df["day_diff"].quantile([0.25, 0.5, 0.75]).values
    average_rating = ((dataframe[dataframe["day_diff"] < q[0]]["overall"].mean()) * w1 / 100) + \
                     ((dataframe[(dataframe["day_diff"] > q[0]) & (dataframe["day_diff"] < q[1])]["overall"].mean()) * w2 / 100) + \
                     ((dataframe[(dataframe["day_diff"] > q[1]) & (df["day_diff"] < q[2])]["overall"].mean()) * w3 / 100) + \
                     ((dataframe[dataframe["day_diff"] > q[2]]["overall"].mean()) * w4 / 100)
    return average_rating
time_based_average(df, 50 , 20 ,20, 10)
# 4.63375956877718

# Adım 4:  Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıldı.

df[df["days"] < 280]["overall"].mean() # 4.694762684124386
df[(df["days"] > 280) & (df["days"] < 430)]["overall"].mean() #4.637335526315789
df[(df["days"] > 430) & (df["days"] < 600)]["overall"].mean() # 4.571428571428571
df[df["days"] > 600]["overall"].mean() # 4.4462540716612375

# Son zamanlardaki değerlendirme sonuçları önceki değerlendirmelere  göre daha yüksek bir
# ortalamaya sahip bunun sebebi üründe bir iyileşme olması veya başka bir olumlu nedenden
# kaynaklanıyor olabilir.
# Gün sayısını azaltarak  değerlendirme ortalamalarını hesapladığımızda açık bir şekilde
# bu durum gözükmektedir.
# örneğin son 1 günde yapılan değerlendirmelerin ortalaması 4.833 eşitken
# 50 günden daha uzun zaman önce değerlendirenlerin ortalaması 4.582933444606718 eşittir.

df[df["days"] < 1]["overall"].mean()
df[(df["days"] > 50)]["overall"].mean()

#########################################################################
# Görev 2:  Ürün için ürün detay sayfasında görüntülenecek 20 review’i belirleyiniz.
#########################################################################
# Adım 1:  helpful_no değişkenini üretiniz.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# Adım 2:  score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplandı.

def score_pos_neg_diff(pos, neg):
    return pos - neg

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()

def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

def wilson_lower_bound(pos, neg, confidence=0.95):

    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.
    ikili veriye çevirerek diğer işlemlerimizide bu fonksiyon ile yapabiliriz
    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head(10)

# Adım 3:  sonuçları Yorumlayınız
df.sort_values("wilson_lower_bound", ascending=False).head(20)
df.sort_values("wilson_lower_bound", ascending=False)["reviewText"].head(20)


# score_pos_neg_diff fonksiyonu kullanılarak hesaplanan skor yüzde olarak bir hesaplama yapmadığı için
# frekans olarak önemli ama yüzde bilgisi eksik sıralama sonuçları vermiştir.
# score_average_rating fonksiyonu pozitif rate oranları ile bir skor hesaplanır ve sıralama yapılır.
# Bu yöntem oran problemimizi çözmüş olsada frekansı bilgisini kaçırmaktadır.

# Her ikisinide değerlendirmek için wilson_lower_bound fonksiyonu ile skor hesaplaması yaptık.
# Bu yöntem, bernolli parametresi p için bir güven aralığı hesaplar ve bu güven aralığının alt sınırını wlb skor olarak kabul eder.
# Güven aralığı girmemizin nedeni müşteriler ile ilgili bütün etkileşimleri içeren verinin elimizde bulunmamasıdır.
# Örneklem alarak var olan veri  içerisinden up olanları bütün kitleye yansıttık.
# Sonuç olarak bulunan aralıktan alt skor ile sıralama yaparak, oran ve frekansı bilgisininde göz önünde bulundurulduğu %95 doğrulukta bir
# sıralama sonucu elde edilmiştir.




