# AI-Powered Medical Image Analysis Suite

Özet: Bu depo, çeşitli tıbbi görüntüleme verileri üzerinde yüksek doğrulukla analiz yapabilen dört farklı derin öğrenme modelini içermektedir. Modeller; hematoloji, nöroloji, oftalmoloji ve radyoloji alanlarında teşhis süreçlerini desteklemek amacıyla optimize edilmiştir.

## 🚀 Model Portföyü ve Performans Analizi

- Bu proje kapsamında geliştirilen modellerin teknik metrikleri ve kullanım alanları aşağıda detaylandırılmıştır:

  ### 1. HemaDeep: Kan Kanseri Tespit Modeli

        Özel olarak geliştirilmiş bir mimari olan HemaDeep, mikroskobik kan görüntülerinden kanser hücrelerini ayırt etmek için tasarlanmıştır.

        Test Doğruluğu (Test Acc): %99.07

  ### 2. Beyin Tümörü Segmentasyonu (U-Net)

        MRI taramaları üzerinde tümörlü bölgelerin tespiti ve segmentasyonu için U-Net mimarisi kullanılmıştır. Düşük kayıp oranları, modelin piksel bazlı hassasiyetini göstermektedir.

        Eğitim Doğruluğu: %99.60 | Eğitim Kaybı (Loss): 0.0264

        Doğrulama Doğruluğu (Val Acc): %99.31 | Doğrulama Kaybı (Val Loss): 0.0345

  ### 3. Diyabetik Retinopati Tespiti (Hybrid MobileNetV2-VGG16)

        Göz dibi (fundus) fotoğrafları üzerinden diyabetik retinopati teşhisi için MobileNetV2 ve VGG16 mimarilerinin güçlü yönlerini birleştiren hibrit bir yaklaşım uygulanmıştır.

        Genel Doğruluk: %96

  ### 4. Zatürre (Pneumonia) Tespiti (U-Net)

        Göğüs röntgenleri (X-Ray) üzerinden akciğer enfeksiyonlarını tespit etmek için optimize edilmiş U-Net modelidir.

        Eğitim Doğruluğu: %98.09 | Eğitim Kaybı (Loss): 0.0761

        Doğrulama Doğruluğu (Val Acc): %97.26 | Doğrulama Kaybı (Val Loss): 0.0866

## 🛠️ Kurulum ve Kullanım

### Gereksinimler

    Modelleri çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

    Python 3.8+

    TensorFlow / Keras

    OpenCV

    NumPy & Pandas

## 📊 Mimari Yapı Hakkında Notlar

Projede kullanılan U-Net mimarisi, özellikle tıbbi görüntü segmentasyonunda "encoder-decoder" yapısı sayesinde nesne sınırlarını belirlemede yüksek başarı sağlamaktadır. MobileNetV2 kullanımı ise modelin hızını optimize ederek düşük donanımlı cihazlarda bile çalışabilmesine olanak tanır.

### Yasal Uyarı: Bu projede sunulan modeller tıbbi karar destek mekanizmalarıdır. Kesin teşhis için uzman doktor onayı gereklidir.

Bu proje MIT LICENSE ile lisanslanmıştır.
