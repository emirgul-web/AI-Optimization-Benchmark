import torch
import numpy as np
import random
import os
from transformers import AutoTokenizer, AutoModel

# --- AYARLAR ---
MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"
INPUT_FILE = "soru_cevap.txt"

# Hedeflenen Çift Sayıları 
TRAIN_COUNT = 50  # 50 İyi + 50 Kötü = 100 Dosya üretir
TEST_COUNT = 50   # 50 İyi + 50 Kötü = 100 Dosya üretir

TRAIN_FOLDER = "egitim_verileri"
TEST_FOLDER = "test_verileri"

# --- MODEL YÜKLEME ---
print(f"Model yükleniyor: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Cihaz: {device}")

def get_embedding(text, is_query=False):
    """Metni vektöre çevirir."""
    prefix = "query: " if is_query else "passage: "
    text_input = prefix + text.strip()
    
    inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()[0]

def save_single_file(filepath, vec_q, vec_a, label):
    """Vektörleri dosyaya kaydeder."""
    with open(filepath, "w", encoding="utf-8") as f:
        # Soru
        for val in vec_q:
            f.write(f"{val:.6f}, ")
        # Cevap
        for val in vec_a:
            f.write(f"{val:.6f}, ")
        # Etiket
        f.write(f"{label}")

def process_batch(pairs, output_folder, file_prefix):
    """
    Belirli bir listeyi işleyip pozitif ve negatif dosyaları oluşturur.
    """
    # Klasörü temizle veya oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"\n--- '{output_folder}' klasörüne {len(pairs)*2} dosya (50 iyi + 50 kötü) yazılıyor... ---")

    for i, (soru, dogru_cevap) in enumerate(pairs):
        # 1. Embeddingleri Hesapla
        vec_soru = get_embedding(soru, is_query=True)
        vec_dogru = get_embedding(dogru_cevap, is_query=False)
        
        # 2. Negatif Örnek Bul (Rastgele yanlış cevap)
        # Sadece bu setin içinden değil, tüm veri havuzundan rastgele seçebiliriz
        # ama basitlik için listeden seçiyoruz.
        random_idx = i
        while random_idx == i and len(pairs) > 1:
            random_idx = random.randint(0, len(pairs) - 1)
        
        yanlis_cevap = pairs[random_idx][1]
        vec_yanlis = get_embedding(yanlis_cevap, is_query=False)

        # 3. Dosyaları Kaydet
        # Pozitif Dosya (Örn: train_0_pozitif.txt)
        path_pos = os.path.join(output_folder, f"{file_prefix}_{i}_pozitif.txt")
        save_single_file(path_pos, vec_soru, vec_dogru, 1.0)

        # Negatif Dosya (Örn: train_0_negatif.txt)
        path_neg = os.path.join(output_folder, f"{file_prefix}_{i}_negatif.txt")
        save_single_file(path_neg, vec_soru, vec_yanlis, -1.0)

        if (i+1) % 10 == 0:
            print(f"  {file_prefix.upper()} İlerleme: {i+1}/{len(pairs)} çift işlendi.")

def main():
    # 1. Dosyayı Oku
    if not os.path.exists(INPUT_FILE):
        print(f"HATA: '{INPUT_FILE}' bulunamadı.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if "|" in line]
    
    pairs = []
    for line in lines:
        parts = line.split("|")
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
            
    print(f"Toplam okunan satır sayısı: {len(pairs)}")
    
    # Yeterli veri kontrolü
    needed_total = TRAIN_COUNT + TEST_COUNT
    if len(pairs) < needed_total:
        print(f"UYARI: Toplam {needed_total} satır gerekli ama dosyada sadece {len(pairs)} satır var!")
        print("Mevcut veriler tekrar kullanılarak eksikler tamamlanacak...")
        # Veri yetmezse listeyi çoğaltarak tamamla
        while len(pairs) < needed_total:
            pairs += pairs
    
    # 2. Veriyi Karıştır
    random.shuffle(pairs)
    
    # 3. Kesin Sayılarla Böl (50 tane Eğitim, 50 tane Test)
    train_pairs = pairs[:TRAIN_COUNT]
    test_pairs = pairs[TRAIN_COUNT : TRAIN_COUNT + TEST_COUNT]
    
    print(f"\nEğitim için ayrılan çift sayısı: {len(train_pairs)} (Üretilecek dosya: {len(train_pairs)*2})")
    print(f"Test için ayrılan çift sayısı: {len(test_pairs)} (Üretilecek dosya: {len(test_pairs)*2})")

    # 4. Dosyaları Oluştur
    # Eğitim Klasörü: train_0...train_49
    process_batch(train_pairs, TRAIN_FOLDER, file_prefix="train")

    # Test Klasörü: test_0...test_49 (Burada index yine 0'dan başlar ki C kodu test_0'dan okusun)
    # C kodun load_data_batch fonksiyonu her klasörde 0'dan başlayarak dosya arıyor.
    process_batch(test_pairs, TEST_FOLDER, file_prefix="test")

    print("\n------------------------------------------------")
    print("İŞLEM TAMAMLANDI!")
    print(f"'{TRAIN_FOLDER}' içinde toplam 100 dosya var.")
    print(f"'{TEST_FOLDER}' içinde toplam 100 dosya var.")

if __name__ == "__main__":
    main()