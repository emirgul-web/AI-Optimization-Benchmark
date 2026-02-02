```markdown
# 🤖 AI Optimization Benchmark: Pure C Implementation

![Language](https://img.shields.io/badge/Language-C%20%7C%20Python-blue)
![Focus](https://img.shields.io/badge/Focus-Math%20%26%20Optimization-green)
![Viz](https://img.shields.io/badge/Visualization-t--SNE-purple)

Bu proje, yapay zeka eğitiminde kullanılan temel optimizasyon algoritmalarının (**Adam, SGD, GD**) performanslarını, **saf C dili** ile sıfırdan yazılmış bir motor üzerinde karşılaştıran hibrit bir çalışmadır.

Hazır kütüphaneler (PyTorch, Keras) yerine, matematiği (Türev, Gradyan, Matris) manuel olarak kodlanarak **2049 boyutlu** vektör uzayında eğitim gerçekleştirilmiştir.

## 🚀 Proje Özellikleri

Bu proje, bir "Araştırma & Geliştirme" (R&G) çalışması olarak tasarlanmıştır:

* **Saf C Motoru:** Geri yayılım (Backpropagation) ve ağırlık güncelleme işlemleri, harici kütüphane olmadan C ile yazılmıştır.
* **LLM Destekli Veri:** Eğitim verileri, `Turkish-Gemma-9b-T1` modeli kullanılarak sentetik olarak üretilmiş ve embedding'e dönüştürülmüştür.
* **Yörünge Görselleştirme:** 2049 boyutlu ağırlık değişimleri, **t-SNE** ile 2 boyuta indirgenerek algoritmaların öğrenme yolları çizilmiştir.
* **Benchmark Sonuçları:** Adam algoritmasının, SGD ve GD'ye göre %40 daha hızlı yakınsadığı (convergence) kanıtlanmıştır.

## 🧠 Algoritma Mantığı

Eğitim motoru şu döngüyü (Epoch) izler:

1.  **Forward Pass:** Girdi vektörü ($X$) ile Ağırlık matrisi ($W$) çarpılır.
2.  **Loss Calculation:** Tahmin ile Gerçek değer arasındaki fark (Hata) hesaplanır.
3.  **Gradient Computation:** Hatanın ağırlıklara göre türevi ($\partial E / \partial W$) C ile hesaplanır.
4.  **Update (Adam/SGD):** Ağırlıklar, seçilen algoritmanın matematiksel formülüne (Momentum, Varyans vb.) göre güncellenir.

## 📂 Proje Yapısı

```bash
AI-Optimization-Benchmark/
├── C_code/             # main.c (Eğitim motoru)
├── python_scripts/     # Veri hazırlama ve Görselleştirme (t-SNE)
├── training_data/      # Vektörleştirilmiş soru-cevap setleri
├── docs/               # Sonuç grafikleri ve Raporlar
└── README.md
