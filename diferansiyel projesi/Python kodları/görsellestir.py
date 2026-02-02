import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CSV Dosyasını Oku
try:
    df = pd.read_csv("sonuclar.csv")
    print("Veri seti başarıyla yüklendi.")
    print(df.head()) # İlk 5 satırı göster
except FileNotFoundError:
    print("Hata: 'sonuclar.csv' dosyası bulunamadı. Lütfen önce C kodunu çalıştırın.")
    exit()

# Görselleştirme Ayarları
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Renk Paleti
palette = {"GD": "tab:blue", "SGD": "tab:orange", "Adam": "tab:green"}

# --- GRAFİK 1: Epoch vs Train Loss ---
# Algoritmaların öğrenme kararlılığını gösterir
sns.lineplot(data=df, x="Epoch", y="TrainLoss", hue="Algorithm", palette=palette, ax=axes[0, 0])
axes[0, 0].set_title("Eğitim Kaybı (Epoch Bazlı)", fontsize=14)
axes[0, 0].set_ylabel("Loss (Hata)")
axes[0, 0].set_xlabel("Epoch")

# --- GRAFİK 2: Epoch vs Test Accuracy ---
# Modelin genelleme başarısını (overfitting olup olmadığını) gösterir
sns.lineplot(data=df, x="Epoch", y="TestAcc", hue="Algorithm", palette=palette, ax=axes[0, 1])
axes[0, 1].set_title("Test Başarısı (Epoch Bazlı)", fontsize=14)
axes[0, 1].set_ylabel("Accuracy (Başarı)")
axes[0, 1].set_xlabel("Epoch")

# --- GRAFİK 3: Time vs Train Loss ---
# Hangi algoritmanın daha hızlı öğrendiğini (süre açısından) gösterir
sns.lineplot(data=df, x="Time", y="TrainLoss", hue="Algorithm", palette=palette, ax=axes[1, 0])
axes[1, 0].set_title("Eğitim Kaybı (Süre Bazlı)", fontsize=14)
axes[1, 0].set_ylabel("Loss (Hata)")
axes[1, 0].set_xlabel("Süre (Saniye)")

# --- GRAFİK 4: Time vs Test Accuracy ---
# Hedef başarıya kimin daha çabuk ulaştığını gösterir
sns.lineplot(data=df, x="Time", y="TestAcc", hue="Algorithm", palette=palette, ax=axes[1, 1])
axes[1, 1].set_title("Test Başarısı (Süre Bazlı)", fontsize=14)
axes[1, 1].set_ylabel("Accuracy (Başarı)")
axes[1, 1].set_xlabel("Süre (Saniye)")

plt.tight_layout()
plt.savefig("karsilastirma_sonuclari.png", dpi=300)
print("\nGrafikler 'karsilastirma_sonuclari.png' dosyasına kaydedildi.")
plt.show()