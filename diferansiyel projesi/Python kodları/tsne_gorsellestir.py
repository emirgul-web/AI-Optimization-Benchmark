import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- AYARLAR ---
CSV_FILE = "agirlik_gecmisi.csv"
PERPLEXITY = 30  
LEARNING_RATE = 200 

print(f"1. Veri okunuyor: {CSV_FILE} (Bu islem biraz surebilir...)")
try:
    # Pandas ile büyük CSV'yi oku
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"Hata: '{CSV_FILE}' bulunamadi. Once C kodunu calistirin.")
    sys.exit(1)

print(f"Veri yuklendi. Boyut: {df.shape}")

# 2. Veriyi Hazırla
# Ağırlık sütunlarını (W0, W1, ...) ayıralım.
weight_columns = [col for col in df.columns if col.startswith('W')]
X = df[weight_columns].values

# Etiket bilgileri
meta_data = df[['Algorithm', 'Run', 'Epoch']].copy()

print(f"2. t-SNE Basliyor... (Boyut: {X.shape} -> 2D)")
print("Lutfen bekleyin, bu islem veri boyutuna gore birkac dakika surebilir.")

# --- t-SNE UYGULAMA ---
# Hata veren parametreler temizlendi
tsne = TSNE(n_components=2, perplexity=PERPLEXITY, learning_rate=LEARNING_RATE, random_state=42)

# !!! İŞTE EKSİK OLAN SATIR BURASIYDI !!!
X_2d = tsne.fit_transform(X)

print("t-SNE tamamlandı.")

# 3. Sonuçları Birleştir
meta_data['x_coord'] = X_2d[:, 0]
meta_data['y_coord'] = X_2d[:, 1]

# --- GÖRSELLEŞTİRME ---
print("3. Grafikler ciziliyor...")
sns.set_theme(style="whitegrid")

# Her algoritma için ayrı bir grafik oluştur
algorithms = meta_data['Algorithm'].unique()
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)

# Renk paleti
run_palette = sns.color_palette("husl", df['Run'].nunique())

for i, algo in enumerate(algorithms):
    ax = axes[i]
    subset = meta_data[meta_data['Algorithm'] == algo]
    
    # Yörüngeleri çiz
    sns.lineplot(
        data=subset, 
        x="x_coord", y="y_coord", 
        hue="Run", legend="full", palette=run_palette, 
        sort=False, lw=2, alpha=0.8, ax=ax
    )
    
    # Başlangıç ve Bitiş Noktalarını İşaretle
    for run_id in subset['Run'].unique():
        run_data = subset[subset['Run'] == run_id]
        # Başlangıç - Daire (o)
        start_point = run_data.iloc[0]
        ax.scatter(start_point['x_coord'], start_point['y_coord'], marker='o', color='green', s=100, label='Start' if run_id==0 else "")
        # Bitiş - Çarpı (X)
        end_point = run_data.iloc[-1]
        ax.scatter(end_point['x_coord'], end_point['y_coord'], marker='X', color='red', s=150, label='End' if run_id==0 else "")

    ax.set_title(f"Optimizasyon Yorungesi: {algo}", fontsize=14)
    ax.set_xlabel("t-SNE Boyut 1")
    if i == 0:
        ax.set_ylabel("t-SNE Boyut 2")
    else:
        ax.set_ylabel("")
        try:
            ax.get_legend().remove()
        except:
            pass

plt.suptitle(f"2049 Boyutlu Agirlik Uzayinin t-SNE Ile 2D Gosterimi\n(5 Farkli Baslangic Noktasi)", fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig("tsne_yorungeler.png", dpi=300, bbox_inches='tight')
print("Grafik 'tsne_yorungeler.png' olarak kaydedildi.")
plt.show()