from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Cargar modelos
model_bin = tf.keras.models.load_model('modelo_binario.h5')
model_cls = tf.keras.models.load_model('modelo_razas.h5')

label_names = ["Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau", "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese", "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland", "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed", "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier", "Wheaten Terrier", "Yorkshire Terrier"]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array * 255.0)
    return img_array

@app.route('/')
def home():
    return 'Dog Classifier API - Usa /classify'

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)

        bin_pred = model_bin.predict(processed_image, verbose=0)
        dog_probability = float(bin_pred[0][0])

        result = {'dog_probability': dog_probability, 'is_dog': dog_probability > 0.5}

        if dog_probability > 0.5:
            cls_pred = model_cls.predict(processed_image, verbose=0)
            probabilities = cls_pred[0]
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_breeds = []
            
            for idx in top_indices:
                top_breeds.append({
                    'breed': label_names[idx],
                    'probability': float(probabilities[idx])
                })
            
            result['top_breeds'] = top_breeds

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
