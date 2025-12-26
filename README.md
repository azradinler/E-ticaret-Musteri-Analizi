# 📊 Müşteri Segmentasyonu, Derin Öğrenme ve CLV Analizi

## 📌 Proje Özeti
Bu proje, bir e-ticaret veri seti kullanılarak **müşteri segmentasyonu**, **derin öğrenme ile segment tahmini** ve **Müşteri Yaşam Boyu Değeri (Customer Lifetime Value – CLV)** analizi gerçekleştirmeyi amaçlamaktadır.

Çalışma kapsamında **K-Means kümeleme**, **yapay sinir ağı (MLP)** ve **olasılıksal CLV modelleri (BG/NBD ve Gamma-Gamma)** birlikte kullanılarak uçtan uca bir veri bilimi çözümü sunulmuştur.

---

## 🧠 Kullanılan Yöntemler
- Müşteri Bazlı Özellik Mühendisliği  
- K-Means ile Müşteri Segmentasyonu  
- Derin Öğrenme (MLP) ile Segment Sınıflandırma  
- Kümeleme Modeli Değerlendirme Metrikleri  
  - Davies–Bouldin Index  
  - Calinski–Harabasz Index  
- Müşteri Yaşam Boyu Değeri (CLV) Analizi  
  - BG/NBD Modeli  
  - Gamma-Gamma Modeli  

---

## 📂 Veri Seti
- **Dosya Adı:** `ecommerce_customer_behavior_dataset_v2.csv`
- **Veri Türü:** E-ticaret işlem (transaction) bazlı veri
- **Temel Değişkenler:**
  - Customer_ID
  - Order_ID
  - Date
  - Total_Amount
  - Quantity
  - Discount_Amount
  - Session_Duration_Minutes
  - Pages_Viewed
  - Delivery_Time_Days
  - Is_Returning_Customer
