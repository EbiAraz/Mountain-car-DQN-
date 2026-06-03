import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


np.random.seed(42)
tf.random.set_seed(42)


script_dir = Path(__file__).resolve().parent
data_candidates = [
    script_dir / "data" / "customer_data.csv",
    script_dir / "customer_data.csv",
]

data_path = next((path for path in data_candidates if path.exists()), None)
if data_path is None:
    print(
        "Dataset not found at './data/customer_data.csv' or './customer_data.csv'. "
        "Using a synthetic demo dataset instead."
    )
    sample_size = 2000
    tenure = np.random.randint(1, 73, size=sample_size)
    monthly_charges = np.random.uniform(20, 120, size=sample_size)
    total_charges = (tenure * monthly_charges) + np.random.normal(0, 50, size=sample_size)
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=sample_size, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=sample_size, p=[0.35, 0.5, 0.15])
    paperless_billing = np.random.choice(["Yes", "No"], size=sample_size, p=[0.6, 0.4])

    churn_score = (
        0.9 * (contract_type == "Month-to-month").astype(float)
        + 0.6 * (internet_service == "Fiber optic").astype(float)
        + 0.4 * (paperless_billing == "Yes").astype(float)
        - 0.02 * tenure
        + np.random.normal(0, 0.7, size=sample_size)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score))
    churned = (np.random.rand(sample_size) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": np.maximum(total_charges, 0),
            "Contract": contract_type,
            "InternetService": internet_service,
            "PaperlessBilling": paperless_billing,
            "churned": churned,
        }
    )
else:
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path}")

if "churned" not in df.columns and "Churn" in df.columns:
    df["churned"] = df["Churn"].map({"Yes": 1, "No": 0})

if "churned" not in df.columns:
    raise ValueError("No target column found. Expected 'churned' or 'Churn'.")

feature_df = df.drop(columns=["churned"], errors="ignore")
feature_df = feature_df.drop(columns=["customerID", "Churn"], errors="ignore")
feature_df = pd.get_dummies(feature_df, drop_first=True)

x = feature_df.values
y = df["churned"].astype(int).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

print(f"\ntraining set size: {len(x_train)} samples")
print(f"testing set size: {len(x_test)} samples")

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("\nModel summary:")
print("MODEL ARCHITECTURE:")
print("=" * 70)

Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

model = tf.keras.Sequential([
    Dense(units=32, activation="relu", input_shape=(x_train_scaled.shape[1],), name="hidden_layer_1"),
    Dropout(rate=0.3, name="dropout_1"),
    Dense(units=16, activation="relu", name="hidden_layer_2"),
    Dropout(rate=0.2, name="dropout_2"),
    Dense(units=8, activation="relu", name="hidden_layer_3"),
    Dense(units=1, activation="sigmoid", name="output_layer")
])

print("\nModel_summary:")
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("Model compiled succesfully!")

print("\n" + "=" * 70)
print("Model training:")
print("=" * 70)

history = model.fit(
    x_train_scaled,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
        patience=10,
        restore_best_weights=True),

        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=0.0001,),

        ],

)

print("\Training compiled!")

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(x_test_scaled, y_test, verbose=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(history.history["accuracy"], label="Training Accuracy")
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history["loss"], label="Training Loss")
ax2.plot(history.history["val_loss"], label="Validation Loss")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=300, bbox_inches="tight")
print("\nTraining history plot saved as 'training_history.png'.")
plt.show()