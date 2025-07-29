import h5py
import sys

def inspect_h5_file(filepath):
    """
    Mencetak struktur file H5, dengan fokus pada bobot model Keras/TF.
    Fungsi ini akan mencoba beberapa metode untuk menemukan nama lapisan.
    """
    print(f"\n--- Memeriksa File H5: {filepath} ---")
    try:
        with h5py.File(filepath, 'r') as f:
            # Metode 1: Format bobot Keras standar (`save_weights`)
            if 'layer_names' in f.attrs:
                print("Metode 1: Terdeteksi format bobot Keras standar.")
                layer_names = [name.decode('utf-8') for name in f.attrs['layer_names']]
                print("Nama-nama lapisan yang ditemukan di dalam file:")
                for name in layer_names:
                    print(f"- {name}")
            
            # Metode 2: Format Checkpoint TensorFlow (dengan grup 'layers')
            elif 'layers' in f:
                print("Metode 2: Terdeteksi format Checkpoint TensorFlow (grup 'layers').")
                print("Nama-nama lapisan yang ditemukan di dalam grup 'layers':")
                layer_group = f['layers']
                # Mencetak semua nama lapisan di dalam grup 'layers'
                for layer_name in layer_group.keys():
                    print(f"- {layer_name}")
            
            # Metode 3: Fallback, jika format tidak dikenali
            else:
                print("Metode 3 (Fallback): Format tidak dikenal. Mencetak grup tingkat atas:")
                for key in f.keys():
                    print(f"- {key}")

    except Exception as e:
        print(f"Terjadi error saat membaca file: {e}")
        print("Pastikan file tidak korup dan path-nya benar.")
    
    print("\n--- Akhir Pemeriksaan ---")
    print("Silakan salin dan tempel daftar nama lapisan di atas sebagai balasan.")


if __name__ == "__main__":
    # Pastikan nama file ini sesuai dengan file bobot Anda
    model_path = "model_frcnn_fpn_standard_kaggle.weights.h5"
    
    # Cek apakah file ada sebelum mencoba membukanya
    try:
        # Menggunakan 'rb' (read binary) lebih aman untuk file non-teks
        with open(model_path, 'rb'):
            pass
    except FileNotFoundError:
        print(f"ERROR: File '{model_path}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan skrip ini.")
        sys.exit(1)

    inspect_h5_file(model_path)
