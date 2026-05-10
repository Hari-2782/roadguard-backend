import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === Dataset Folders ===
base_dir = "C:/Users/mural/OneDrive/Desktop/RESEARCH/lane_data_gen"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# === Sanity check: ensure folders exist ===
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise Exception("❌ Make sure lane_data_gen/train and lane_data_gen/val exist with subfolders lane/ and no_lane/")

# === Data Generators (auto-detect 2 classes) ===
train_gen = ImageDataGenerator(rescale=1/255., horizontal_flip=True, zoom_range=0.2)
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

# === Build MobileNetV2 model ===
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("✅ MobileNetV2 model compiled successfully!")
print(f"Detected classes: {train_data.class_indices}")

# === Train model ===
history = model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

# === Save model (new version) ===
model.save("C:/Users/mural/OneDrive/Desktop/RESEARCH/mobilenet_lane_v2.keras")
print("🎉 Model training complete and saved to mobilenet_lane_v2.keras")
