import tensorflow as tf

# Image dimensions
IMG_SIZE = (224, 224)

# Dataset folder paths
DATASET_DIR = "datasets/plant_disease"

# Load training images
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR + "/train",
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode="categorical"
)

# Load validation images
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR + "/validation",
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode="categorical"
)

# Load testing images
test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR + "/test",
    image_size=IMG_SIZE,
    batch_size=32,
    label_mode="categorical"
)

# Normalize pixel values between 0 and 1
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

test_dataset = test_dataset.map(
    lambda images, labels: (normalization_layer(images), labels)
)

# Image augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

print("Image preprocessing completed!")
print("Image size:", IMG_SIZE)
print("Normalization: 0 to 1")
print("Augmentation: Rotation, Zoom, Flip and Shift")