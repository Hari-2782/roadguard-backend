import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# === Dataset Folders ===
base_dir = r"C:\Users\mural\OneDrive\Desktop\RESEARCH\lane_data_gen"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# === Sanity Check ===
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise Exception("❌ Folder missing: Make sure lane_data_gen/train and lane_data_gen/val exist with lane/ and no_lane/")

print("✅ Dataset structure verified.")
print(f"Training folder: {train_dir}")
print(f"Validation folder: {val_dir}")

# === Data Generators ===
train_gen = ImageDataGenerator(
    rescale=1/255.,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_gen = ImageDataGenerator(rescale=1/255.)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary"
)

# === Build MobileNetV2 Model ===
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # freeze base layers for now

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("\n✅ MobileNetV2 model compiled successfully!")
print(f"Detected classes: {train_data.class_indices}")
print("\nModel Summary:")
model.summary()

# === Train the model ===
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)

# === Save model safely ===
save_path = r"C:\Users\mural\OneDrive\Desktop\RESEARCH\mobilenet_lane_v3.keras"
model.save(save_path)
print(f"\n🎉 Model training complete and saved to {save_path}")

# === Plot Accuracy & Loss ===
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
