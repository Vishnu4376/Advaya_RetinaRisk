1️ Install Required Libraries
pip install tensorflow numpy opencv-python pillow flask


2️ Load & Train Model in Google Colab
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

INPUT_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5

train_datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    validation_split=0.2, preprocessing_function=tf.keras.applications.resnet50.preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Heart attack prediction/dataset/',
    target_size=(INPUT_SIZE, INPUT_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/Heart attack prediction/dataset/',
    target_size=(INPUT_SIZE, INPUT_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(INPUT_SIZE, INPUT_SIZE, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=5e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/Heart attack prediction/The_model_resnet.keras', save_best_only=True)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop, checkpoint]
)

base_model.trainable = True
for layer in base_model.layers[:150]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stop, checkpoint]
)

model.save('/content/drive/MyDrive/Heart attack prediction/The_model_resnet_finetuned.keras')


3️ Download Model Locally
from google.colab import files
files.download('/content/drive/MyDrive/Heart attack prediction/The_model_resnet_finetuned.keras')


4️ Load Model Locally and Run Predictions
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

model = tf.keras.models.load_model("The_model_resnet_finetuned.keras")

def preprocess_single_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

image_path = r"E:\Project\Heart attack prediction\uploads\00cc2b75cddd.png"
image = preprocess_single_image(image_path)

prediction = model.predict(image)
predicted_class = np.argmax(prediction, axis=1)[0]

risk_levels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
print(f"Predicted risk level: {risk_levels[predicted_class]}")


5️ Implement Grad-CAM for Visualization
import tensorflow.keras.backend as K
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

model = tf.keras.models.load_model("The_model_resnet_finetuned.keras")

def preprocess_single_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0), image

def grad_cam(model, image, layer_name='conv5_block3_out'):
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap, predicted_class

def overlay_heatmap(heatmap, original_image):
    heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    superimposed_image = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    
    plt.imshow(cv2.cvtColor(superimposed_image, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM Activation Map")
    plt.axis("off")
    plt.show()

image_path = r"E:\Project\Heart attack prediction\uploads\00cc2b75cddd.png"
processed_image, original_image = preprocess_single_image(image_path)
heatmap, predicted_class = grad_cam(model, processed_image)
overlay_heatmap(heatmap, original_image)

risk_levels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
print(f"Predicted risk level: {risk_levels[predicted_class]}")


6️⃣ Deploy as a Web App
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model("The_model_resnet_finetuned.keras")

def preprocess_single_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file)
        image = preprocess_single_image(image)

        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        risk_levels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
        return f"Predicted risk level: {risk_levels[predicted_class]}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)


This README should guide you step by step from training to prediction to deployment!

