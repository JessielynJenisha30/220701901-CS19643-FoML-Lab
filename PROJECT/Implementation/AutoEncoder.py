import os
import numpy as np
import string
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def render_glyphs_from_ttf(ttf_path, img_size=(64, 64), chars = string.ascii_letters + string.digits + string.punctuation
):
    font = ImageFont.truetype(ttf_path, size=48)
    glyphs = []
    for char in chars:
        img = Image.new("L", img_size, color=0)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((img_size[0]-w)//2, (img_size[1]-h)//2), char, fill=255, font=font)
        glyphs.append(np.array(img) / 255.0)
    return np.array(glyphs).reshape(len(chars), img_size[0], img_size[1], 1)
def build_conv_autoencoder(img_shape=(64, 64, 1), encoding_dim=128):
    from tensorflow.keras import layers, models

    # Encoder
    encoder_input = layers.Input(shape=img_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = x

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(encoder_input, decoded)
    encoder = models.Model(encoder_input, encoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder
  
font_folder = "Fonts"
all_images = []

for font_file in os.listdir(font_folder):
    if font_file.endswith(".ttf"):
        path = os.path.join(font_folder, font_file)
        glyphs = render_glyphs_from_ttf(path)
        all_images.append(glyphs)

all_images = np.vstack(all_images)
print("Total glyphs:", all_images.shape)

autoencoder, encoder = build_conv_autoencoder()
autoencoder.summary()

# Train
autoencoder.fit(all_images, all_images, epochs=50, batch_size=32)

# Pick a few random samples
num_samples = 10
indices = np.random.choice(len(all_images), num_samples)

test_images = all_images[indices]
reconstructed_images = autoencoder.predict(test_images)

plt.figure(figsize=(20, 4))
for i in range(num_samples):
    # Original images
    ax = plt.subplot(2, num_samples, i + 1)
    plt.imshow(test_images[i].reshape(64, 64), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed images
    ax = plt.subplot(2, num_samples, i + 1 + num_samples)
    plt.imshow(reconstructed_images[i].reshape(64, 64), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()


