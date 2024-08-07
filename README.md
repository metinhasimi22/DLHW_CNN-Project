# 📊 DLHW_CNN-Project    

#  ✨  Introduction - (Tanım)
bu proje kapsamında 4 farklı veri seti kullanarak  CNN ,VGG16 gibi Derin ogrenme teknikleri ile projleri çözmek sonra iyi bir başarı oranı ile modellerimi kaydedip , kayıt edilen modelleri  streamlit yardımı ile Hugifaceye yüklemek ([Kaynak](https://huggingface.co/Metinhsimi/activity/spaces)  ve test etmek .
amacımız Derin ogrenme ile classification yaparak görsel verileri işleyerek ayırt etme ve bunları ihtiyaca yönelik uygulamalarda kulanabilme .

# 🔍 Analysis - (Analiz)
Her bir veri seti üzerinde, verilerin yapısı, sınıf dağılımı ve veri setinin genel özellikleri incelenmiştir.

## 🌸 Flower Classification - (Çiçek)
- Veri Seti: Çiçek türlerini sınıflandırmak amacıyla kullanılan bu veri seti, birden fazla çiçek türü içermektedir.
- Görselleştirme: Veri seti üzerinde sınıf dağılımları ve örnek görüntüler incelenmiştir.
- [Kaynak!](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- ![Ekran görüntüsü 2024-08-07 203628](https://github.com/user-attachments/assets/f25d3776-11ca-440b-b384-5ab85ca78566)

## 🍇 Date Fruit Classification (Hurma)
- Veri Seti: Hurma türlerinin görsellerini içeren bu veri seti, 9 sınıfa ayrılmıştır.
- Görselleştirme: Veri setinde her sınıfa ait örnek görüntüler ve sınıf dağılımları incelenmiştir.
- [Kaynak!](https://www.kaggle.com/datasets/wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment?select=Sokari)
## 🍃 Leaf Disease Detection - (Üzüm Yaprağı Hastalıkları)
- Veri Seti: Yaprak hastalıklarını sınıflandırmak için kullanılan bu veri seti 4 sınıfa ayrılmıştır.
- Görselleştirme: Hastalıklı ve sağlıklı yaprak görüntüleri, sınıf dağılımı analiz edilmiştir.
- [Kaynak!](https://www.kaggle.com/datasets/rm1000/augmented-grape-disease-detection-dataset)
## 🌾 Rice Classification - (Pirinç)
- Veri Seti: Pirinç türlerini sınıflandıran bu veri seti, çeşitli pirinç türlerinin görsellerini içermektedir.
- Görselleştirme: Veri setindeki örnek görüntüler ve sınıfların dengesi görselleştirilmiştir.
- [Kaynak!](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)



# 🛠️ Methods - (Yöntemler)
### Her bir veri seti için ayrı bir Convolutional Neural Network (CNN) modeli geliştirilmiştir. Modellerin mimarisi genel hatlarıyla aşağıdaki gibidir:

- Girdi Katmanı: 32x32 piksel boyutunda görüntülerin girişi sağlanmıştır.
- Katmanlar: Her modelde en az 5 adet Conv2D katmanı, 3 adet MaxPooling katmanı kullanılmıştır.
- Conv2D Katmanları: 32, 64, 128, 128, ve 256 filtre sayıları ile kullanılmıştır.
- Aktivasyon Fonksiyonu: ReLU kullanılmıştır.
- MaxPooling Katmanları: 2x2 boyutlarında uygulanmıştır tabi bazılarında farklılık gösterebilir .
- Dropout Katmanı: Overfitting’i önlemek için kullanılmıştır , önemli .
- Çıktı Katmanı: Softmax aktivasyon fonksiyonu ile sınıflandırma yapılmıştır.
### Hiperparametreler:
##### "Modellden modele değişklilk gösterebilir"
- Batch Size: 32  
- Epoch Sayısı: 50
- Kayıp Fonksiyonu: Sparse Categorical Crossentropy
- Optimizasyon Algoritması: Adam


# 📈 Results - (Sonuçlar)
- Her bir veri seti için oluşturulan modellerin eğitim ve test sonuçları aşağıda     sunulmaktadır:  her model için ortalama en yüksek değerler alınmaktadır       
  bazılarını   99 arasında  90  tutarak sonuçlanmıştur
# Projenin Sonuçları HuggingFace:[Kanağı](https://huggingface.co/Metinhsimi/activity/spaces)

# 💡 Reflection - (Değerlendirme)
- Bu proje sırasında, farklı veri setleri üzerinde CNN modellerinin nasıl uygulanabileceği ve performanslarının nasıl optimize edilebileceği öğrenilmiştir. Sınıf dengesizliği olan veri setlerinde modelin performansını artırmak için veri artırma teknikleri kullanılabilir. Gelecekte, daha büyük veri setleri ve daha karmaşık model mimarileri kullanarak sonuçları geliştirmeyi planlıyorum. Ayrıca, transfer öğrenme gibi teknikleri de kullanarak modellerin performansını daha ileriye taşımak mümkündür.
