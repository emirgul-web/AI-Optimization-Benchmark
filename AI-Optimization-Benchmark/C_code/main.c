#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Temel ayarlar
#define VECTOR_DIM 1024        
#define INPUT_DIM (2 * VECTOR_DIM + 1) 
#define MAX_SAMPLES 2000      // Bellek tasmasin diye buyuk bir sayi sectim
#define EPOCHS 100            
#define LEARNING_RATE 0.001    
#define NUM_RUNS 5            

// Klasor isimleri 
#define TRAIN_FOLDER "egitim_verileri"
#define TEST_FOLDER "test_verileri"
#define WEIGHT_PREFIX "w_init" 

// Adam 
#define BETA1 0.9
#define BETA2 0.999
#define EPSILON 1e-8

// struct 
typedef struct {
    double features[INPUT_DIM]; 
    double label;                
} Sample;



double activation_tanh(double x) {
    return tanh(x); // Etiketler -1 ve 1 oldugu icin tanh kullandim
}

double activation_derivative(double output) {
    return 1.0 - (output * output);
}

// Tahmin fonksiyonu (Agirlik * Girdi)
double predict(double *weights, double *features) {
    int j; 
    double z = 0.0;
    for (j = 0; j < INPUT_DIM; j++) {
        z += weights[j] * features[j];
    }
    return activation_tanh(z);
}

void load_weights_file(const char *filename, double *weights_buffer) {
    int i;
    FILE *file = fopen(filename, "r");
    
    // Dosya yoksa hata vermesin, rastgele baslatsin
    if (file == NULL) {
        printf("UYARI: '%s' bulunamadi, rastgele agirliklar atanacak.\n", filename);
        for(i=0; i<INPUT_DIM; i++) weights_buffer[i] = ((double)rand() / RAND_MAX) * 0.01;
        return;
    }

    for (i = 0; i < INPUT_DIM; i++) {
        if (fscanf(file, "%lf,", &weights_buffer[i]) != 1) {
             if(fscanf(file, "%lf", &weights_buffer[i]) != 1) {
                 weights_buffer[i] = 0.0; 
             }
        }
    }
    fclose(file);
}

int read_single_file(const char *filepath, Sample *dataset, int index) {
    int j;
    FILE *file = fopen(filepath, "r");
    
    if (file == NULL) return 0; 

    // Vektorleri dosyadan okuyup struct'a atiyoruz
    for(j=0; j<VECTOR_DIM; j++) fscanf(file, "%lf,", &dataset[index].features[j]);
    for(j=0; j<VECTOR_DIM; j++) fscanf(file, "%lf,", &dataset[index].features[VECTOR_DIM + j]);
    dataset[index].features[2 * VECTOR_DIM] = 1.0; // Bias icin sona 1 ekledim
    fscanf(file, "%lf", &dataset[index].label);

    fclose(file);
    return 1;
}

// Klasordeki tum dosyalari tarayip yukleyen fonksiyon.
int load_data_batch(const char *folder, const char *prefix, Sample *dataset, int start_index) {
    int i;
    int loaded_count = 0;
    char filepath[256];

    printf("Veriler '%s' klasorunden ('%s' on eki ile) yukleniyor...\n", folder, prefix);
    
    // Klasordeki dosyalari tek tek deniyoruz.
    for (i = 0; i < MAX_SAMPLES; i++) {
        // Pozitif Ornek
        sprintf(filepath, "%s/%s_%d_pozitif.txt", folder, prefix, i);
        if (read_single_file(filepath, dataset, start_index + loaded_count)) {
            loaded_count++;
        }

        // Negatif Ornek
        sprintf(filepath, "%s/%s_%d_negatif.txt", folder, prefix, i);
        if (read_single_file(filepath, dataset, start_index + loaded_count)) {
            loaded_count++;
        }
    }
    
    printf("  -> Bu klasorden %d adet ornek yuklendi.\n", loaded_count);
    return loaded_count; 
}

// --- EGITIM DONGUSU ---
void run_training_session(int algo_type, int run_id, Sample *dataset, int train_count, int test_count, double *init_weights, FILE *csv_file, FILE *weight_history_file) {
    int i, j, epoch, t_step = 0;
    int correct_train, correct_test;
    double total_loss, train_acc, test_acc;
    double start_time, end_time, time_elapsed = 0.0; 
    
    double corr1, corr2; // Adam duzeltme katsayilari

    // Bellek yonetimi 
    double *weights = (double *)malloc(INPUT_DIM * sizeof(double));
    double *gradients = (double *)malloc(INPUT_DIM * sizeof(double)); 
    double *m = (double *)calloc(INPUT_DIM, sizeof(double)); 
    double *v = (double *)calloc(INPUT_DIM, sizeof(double)); 

    // Baslangic agirliklarini kopyala
    for (j = 0; j < INPUT_DIM; j++) weights[j] = init_weights[j];

    char *algo_name = (algo_type == 1) ? "GD" : (algo_type == 2) ? "SGD" : "Adam";
    printf(">> %s | Run: %d Basliyor (Train: %d, Test: %d)...\n", algo_name, run_id + 1, train_count, test_count);

    for (epoch = 0; epoch < EPOCHS; epoch++) {
        start_time = (double)clock() / CLOCKS_PER_SEC;
        total_loss = 0.0;
        correct_train = 0;

        // --- ALGORITMA SECIMI ---
        if (algo_type == 1) { // GD (Batch Gradient Descent)
            for (j = 0; j < INPUT_DIM; j++) gradients[j] = 0.0;
            
            // Tum veriyi gez, hatayi topla
            for (i = 0; i < train_count; i++) { 
                double output = predict(weights, dataset[i].features);
                double error = output - dataset[i].label;
                total_loss += 0.5 * error * error;
                if ((output > 0 && dataset[i].label > 0) || (output < 0 && dataset[i].label < 0)) correct_train++;
                
                double common = error * activation_derivative(output);
                for (j = 0; j < INPUT_DIM; j++) gradients[j] += common * dataset[i].features[j];
            }
            // Agirliklari en sonda guncelle
            for (j = 0; j < INPUT_DIM; j++) weights[j] -= LEARNING_RATE * (gradients[j] / train_count);
        }
        else { // SGD ve Adam (Her adimda guncelleme)
            for (i = 0; i < train_count; i++) { 
                double output = predict(weights, dataset[i].features);
                double error = output - dataset[i].label;
                total_loss += 0.5 * error * error;
                if ((output > 0 && dataset[i].label > 0) || (output < 0 && dataset[i].label < 0)) correct_train++;
                
                double common = error * activation_derivative(output);

                if (algo_type == 2) { // SGD mantigi
                    for (j = 0; j < INPUT_DIM; j++) weights[j] -= LEARNING_RATE * common * dataset[i].features[j];
                } 
                else if (algo_type == 3) { // Adam mantigi
                    t_step++;
                    corr1 = 1.0 - pow(BETA1, t_step);
                    corr2 = 1.0 - pow(BETA2, t_step);

                    for (j = 0; j < INPUT_DIM; j++) {
                        double g = common * dataset[i].features[j];
                        m[j] = BETA1 * m[j] + (1.0 - BETA1) * g;
                        v[j] = BETA2 * v[j] + (1.0 - BETA2) * g * g;
                        
                        double m_hat = m[j] / corr1;
                        double v_hat = v[j] / corr2;
                        
                        weights[j] -= LEARNING_RATE * m_hat / (sqrt(v_hat) + EPSILON);
                    }
                }
            }
        }

        end_time = (double)clock() / CLOCKS_PER_SEC;
        time_elapsed += (end_time - start_time);

        // Grafik cizdirmek icin kayit aliyoruz (Her 10 epochta bir)
        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == EPOCHS - 1) {
            fprintf(weight_history_file, "%s,%d,%d", algo_name, run_id, epoch + 1);
            for(j = 0; j < INPUT_DIM; j++) {
                fprintf(weight_history_file, ",%.6f", weights[j]);
            }
            fprintf(weight_history_file, "\n");
        }

        // TEST KISIMI
        double test_total_loss = 0.0;
        correct_test = 0;
        
        for(i = train_count; i < train_count + test_count; i++) {
             double out = predict(weights, dataset[i].features);
             double err = out - dataset[i].label;
             test_total_loss += 0.5 * err * err;
             if ((out > 0 && dataset[i].label > 0) || (out < 0 && dataset[i].label < 0)) correct_test++;
        }

        train_acc = (double)correct_train / train_count;
        test_acc = (test_count > 0) ? (double)correct_test / test_count : 0.0;
        double avg_train_loss = total_loss / train_count;
        double avg_test_loss = (test_count > 0) ? test_total_loss / test_count : 0.0;

        // Sonuclari CSV'ye yaz
        fprintf(csv_file, "%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
                algo_name, run_id, epoch + 1, time_elapsed, avg_train_loss, train_acc, avg_test_loss, test_acc);
    }
    
    free(weights); free(gradients); free(m); free(v);
}

int main() {
    int train_count, test_count, algo, run, j;
    char w_filename[50];
    
    srand(time(NULL)); 

   
    double *temp_initial_weights = (double *)malloc(INPUT_DIM * sizeof(double));
    Sample *dataset = (Sample *)malloc(MAX_SAMPLES * 4 * sizeof(Sample)); 

    if (!temp_initial_weights || !dataset) {
        printf("Bellek Hatasi!\n"); return 1;
    }

    // 1. Egitim verilerini yukle
    train_count = load_data_batch(TRAIN_FOLDER, "train", dataset, 0);

    // 2. Test verilerini yukle (Egitimden sonrasina ekle)
    test_count = load_data_batch(TEST_FOLDER, "test", dataset, train_count);

    if (train_count == 0 && test_count == 0) {
        printf("HATA: Veriler bulunamadi! Klasor isimlerini kontrol et.\n");
        printf("Beklenen: %s/train_0_pozitif.txt vb.\n", TRAIN_FOLDER);
        getchar(); return 1;
    }

    printf("TOPLAM: Egitim Seti: %d adet, Test Seti: %d adet\n", train_count, test_count);

    FILE *csv_file = fopen("sonuclar.csv", "w");
    FILE *weight_history_file = fopen("agirlik_gecmisi.csv", "w");

    if (csv_file == NULL || weight_history_file == NULL) { printf("Dosya olusturma hatasi!\n"); return 1; }
    
    fprintf(csv_file, "Algorithm,Run,Epoch,Time,TrainLoss,TrainAcc,TestLoss,TestAcc\n");
    fprintf(weight_history_file, "Algorithm,Run,Epoch");
    for(j=0; j<INPUT_DIM; j++) fprintf(weight_history_file, ",W%d", j);
    fprintf(weight_history_file, "\n");

    // Deneyleri sirayla calistir 
    for (algo = 1; algo <= 3; algo++) {
        for (run = 0; run < NUM_RUNS; run++) {
            sprintf(w_filename, "%s%d.txt", WEIGHT_PREFIX, run + 1);
            load_weights_file(w_filename, temp_initial_weights);
            
            run_training_session(algo, run, dataset, train_count, test_count, temp_initial_weights, csv_file, weight_history_file);
        }
    }

    printf("\nIslem bitti. Sonuclar CSV dosyasina yazildi.\n");

    fclose(csv_file);
    fclose(weight_history_file);
    free(dataset);
    free(temp_initial_weights);
    return 0;
}
