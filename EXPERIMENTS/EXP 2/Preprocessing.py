import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from PIL import Image
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

 --- 1. Tabular Data Preprocessing ---
def preprocess_tabular_data(df):
# Handling missing values
    imputer = SimpleImputer(strategy='mean') # Fill missing values with mean
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])))

# Encoding categorical variables
    encoder = LabelEncoder()
    df_encoded = df.apply(lambda col: encoder.fit_transform(col) if col.dtypes == 'object' else col)

# Normalizing numerical features
    scaler = StandardScaler()
    df_scaled = df_encoded.copy()
    df_scaled[df_scaled.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df_scaled.select_dtypes(include=[np.number]))

    return df_scaled

def preprocess_image(image_path, target_size=(224, 224)):
# Open image using PIL
    img = Image.open(image_path)

# Resize image
    img_resized = img.resize(target_size)

# Convert to numpy array and normalize pixel values (0-255 to 0-1)
    img_array = np.array(img_resized) / 255.0

    return img_array

def preprocess_text(text):
# Lowercasing the text
    text_lower = text.lower()

# Tokenization
    tokens = word_tokenize(text_lower)

# Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens_filtered = [word for word in tokens if word not in stop_words]

# Stemming
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(word) for word in tokens_filtered]

# Return preprocessed text (list of stemmed tokens)
    return tokens_stemmed

df_example = pd.DataFrame({
'age': [25, 30, np.nan, 40, 22],
'gender': ['male', 'female', 'female', 'male', 'female'],
'income': [50000, 60000, 65000, np.nan, 45000]
})

print("Original Tabular Data:")
print(df_example)

df_processed = preprocess_tabular_data(df_example)
print("\nProcessed Tabular Data:")
print(df_processed)
text_example = "This is a random sentence, with some unnecessary information"
text_processed = preprocess_text(text_example)
print("\nProcessed Text Data:", text_processed)

image_path = 'Daisy.jpg' # Provide the path to an image
image_processed = preprocess_image(image_path)
print("\nProcessed Image Data Shape:", image_processed.shape)
