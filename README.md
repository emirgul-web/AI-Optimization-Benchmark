🚀 AI Optimizasyon Algoritmaları Kıyaslaması: Saf C Uygulaması
Bu proje, temel yapay zeka optimizasyon algoritmalarının (Gradient Descent, Stochastic Gradient Descent ve Adam) performanslarını, saf C dili kullanılarak sıfırdan geliştirilmiş bir motor üzerinden karşılaştırır.

PyTorch veya TensorFlow gibi hazır kütüphanelerin aksine, bu proje yapay zeka eğitiminin arkasındaki matematiği (Backpropagation, Türev, Matris İşlemleri) anlamak amacıyla, eğitim motorunu 2049 boyutlu yüksek bir vektör uzayında manuel olarak inşa etmiştir.

🧠 Öne Çıkan Teknik Özellikler
Hibrit Mimari (C & Python):

Veri Üretimi (Python): ytu-ce-cosmos/Turkish-Gemma-9b-T1 Büyük Dil Modeli (LLM) kullanılarak sentetik soru-cevap çiftleri üretilmiş ve embedding vektörlerine dönüştürülmüştür.

Eğitim Motoru (C): Ağırlık güncellemeleri ve hata hesaplamaları, dış kütüphane bağımlılığı olmadan saf C ile, düşük seviyeli bellek yönetimi kullanılarak kodlanmıştır.

Görselleştirme (Python): 2049 boyutlu ağırlık uzayındaki değişim, t-SNE algoritması ile 2 boyuta indirgenerek algoritmaların "öğrenme yörüngeleri" görselleştirilmiştir.

Özel Veri Hattı (Pipeline): Pekiştirmeli öğrenme (RLHF) senaryolarına hazırlık amacıyla, her soru için "İyi" ve "Kötü" cevaplar üreten bir DPO (Direct Preference Optimization) veri hazırlama betiği içerir.

Tekrarlanabilirlik: Deneylerin adil olması için her algoritma, sabitlenmiş başlangıç ağırlıkları (initial_weights) ile test edilmiştir.
