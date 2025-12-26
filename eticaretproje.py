# =====================================================
# CUSTOMER SEGMENTATION + DEEP LEARNING + CLV ANALYSIS
# =====================================================

# ------------------------
# 0) IMPORTS
# ------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    davies_bouldin_score,
    calinski_harabasz_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

# ------------------------
# 1) VERİYİ YÜKLE
# ------------------------
df = pd.read_csv("ecommerce_customer_behavior_dataset_v2.csv")
df["Date"] = pd.to_datetime(df["Date"])

print("Transaction-level Shape:", df.shape)

# =====================================================
# 2) CUSTOMER-LEVEL FEATURE ENGINEERING
# =====================================================
customer_df = df.groupby("Customer_ID").agg({
    "Total_Amount": "sum",
    "Quantity": "sum",
    "Discount_Amount": "sum",
    "Session_Duration_Minutes": "mean",
    "Pages_Viewed": "mean",
    "Delivery_Time_Days": "mean",
    "Is_Returning_Customer": "max",
    "Order_ID": "count"
}).reset_index()

customer_df.rename(columns={"Order_ID": "Total_Orders"}, inplace=True)

print("Customer-level Shape:", customer_df.shape)

# =====================================================
# 3) FEATURE SELECTION & SCALING
# =====================================================
features = [
    "Total_Amount",
    "Quantity",
    "Session_Duration_Minutes",
    "Pages_Viewed",
    "Discount_Amount",
    "Delivery_Time_Days"
]

X = customer_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# 4) K-MEANS CLUSTERING
# =====================================================
kmeans = KMeans(n_clusters=4, random_state=42)
customer_df["Customer_Segment"] = kmeans.fit_predict(X_scaled)

# =====================================================
# 5) SEGMENT ANALYSIS
# =====================================================
print("\n================ SEGMENT ANALİZİ ================\n")

segment_profile = customer_df.groupby("Customer_Segment")[features].mean().round(2)
segment_counts = customer_df["Customer_Segment"].value_counts().sort_index()
segment_returning_rate = (
    customer_df.groupby("Customer_Segment")["Is_Returning_Customer"]
    .mean()
    .round(3)
)

summary = segment_profile.copy()
summary["Customer_Count"] = segment_counts
summary["Returning_Rate"] = segment_returning_rate

print(summary)

# =====================================================
# 6) VISUALIZATION
# =====================================================
sns.scatterplot(
    data=customer_df,
    x="Pages_Viewed",
    y="Total_Amount",
    hue="Customer_Segment",
    palette="Set2"
)
plt.title("Customer Segmentation (Customer Level)")
plt.show()

# =====================================================
# 7) DEEP LEARNING – SEGMENT CLASSIFICATION
# =====================================================
print("\n============= DERİN ÖĞRENME AŞAMASI =============\n")

X_dl = X_scaled
y_dl = customer_df["Customer_Segment"]
y_dl_cat = to_categorical(y_dl)

X_train, X_test, y_train, y_test = train_test_split(
    X_dl, y_dl_cat, test_size=0.2, random_state=42
)

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nDerin Öğrenme Model Accuracy: {accuracy:.3f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Tahmin Edilen Segment")
plt.ylabel("Gerçek Segment")
plt.title("Confusion Matrix - MLP Segment Tahmini")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# =====================================================
# 8) CLUSTERING MODEL EVALUATION
# =====================================================
print("\n=========== KÜMELEME MODEL DEĞERLENDİRME ===========\n")

db_index = davies_bouldin_score(X_scaled, customer_df["Customer_Segment"])
ch_index = calinski_harabasz_score(X_scaled, customer_df["Customer_Segment"])

print(f"Davies–Bouldin Index: {db_index:.4f}")
print(f"Calinski–Harabasz Index: {ch_index:.2f}")

# =====================================================
# 9) CUSTOMER LIFETIME VALUE (CLV) ANALYSIS
# =====================================================
clv_df = summary_data_from_transaction_data(
    df,
    customer_id_col="Customer_ID",
    datetime_col="Date",
    monetary_value_col="Total_Amount"
)

clv_df = clv_df[clv_df["frequency"] > 0]

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(clv_df["frequency"], clv_df["recency"], clv_df["T"])

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(clv_df["frequency"], clv_df["monetary_value"])

clv_df["Tahmini_Gelecek_Deger"] = ggf.customer_lifetime_value(
    bgf,
    clv_df["frequency"],
    clv_df["recency"],
    clv_df["T"],
    clv_df["monetary_value"],
    time=6,
    freq="D",
    discount_rate=0.01
)

clv_df["CLV_Segment"] = pd.qcut(
    clv_df["Tahmini_Gelecek_Deger"],
    4,
    labels=["D", "C", "B", "A"]
)

print("\n========== CLV TOP CUSTOMERS ==========\n")
print(
    clv_df.sort_values("Tahmini_Gelecek_Deger", ascending=False)
    .head(10)
)

# CLV Visualization
plt.figure(figsize=(10,6))
sns.boxplot(
    x="CLV_Segment",
    y="Tahmini_Gelecek_Deger",
    data=clv_df,
    palette="magma"
)
plt.title("CLV Segmentlerine Göre Tahmini Gelecek Değer")
plt.show()
