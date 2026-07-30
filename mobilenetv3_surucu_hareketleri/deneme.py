import torch

def inspect_model_classes(pkl_path):
    print(f"Modül analiz ediliyor: {pkl_path}")
    
    try:
        # Ağırlıkları ve meta verileri CPU üzerinde belleğe alıyoruz
        data = torch.load(pkl_path, map_location=torch.device('cpu'))
        
        # Eğer veri bir sözlük (dictionary) ise içindeki anahtarları tararız
        if isinstance(data, dict):
            print("\nModel içindeki anahtarlar (Keys):", list(data.keys()))
            
            # Sınıf listesini barındırabilecek olası anahtar isimleri
            possible_keys = ['classes', 'class_names', 'names', 'class_to_idx', 'categories']
            
            found = False
            for key in possible_keys:
                if key in data:
                    print(f"\n[ BAŞARILI ] Sınıflar '{key}' anahtarında bulundu:")
                    print(data[key])
                    found = True
                    break
                    
            if not found:
                print("\n[ BİLGİ ] Model dosyası sınıfları içermiyor. Sadece ağırlıklar (state_dict) kaydedilmiş.")
                
        else:
            print("\nModel yapısı bir sözlük değil. Türü:", type(data))
            
    except Exception as e:
        print(f"\n[ HATA ] Dosya okunurken bir sorun oluştu: {str(e)}")

# Dosya yolunu kendi sisteminize göre güncelleyin
dosya_yolu = "mobilenetv3_surucu_hareketleri/data.pkl"
inspect_model_classes(dosya_yolu)