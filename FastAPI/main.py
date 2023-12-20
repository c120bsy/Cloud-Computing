from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import uuid
import os


IMAGEDIR = "images/"

model = tf.keras.models.load_model('model/model.h5')

class_labels = [
    'Biru', 
    'Coklat', 
    'Hijau', 
    'Hitam', 
    'Jingga', 
    'Kuning', 
    'Merah', 
    'Putih'
    ]

app = FastAPI()

@app.get('/')
def main():
    return {'message': 'Welcome to Capstone CH2-PS036'}

@app.post('/upload')
async def upload(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    # Save file
    with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
        f.write(contents)

    return {"filename" : file.filename}

@app.get('/predict')
def prediction():
    files = os.listdir(IMAGEDIR)
    path = f"{IMAGEDIR}{files[0]}"

    
    img = image.load_img(path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array)
    class_index = np.argmax(pred)
    class_prediction = class_labels[class_index]

    confidence_score = pred[0][class_index] * 100

    if os.path.isfile(path):
        os.remove(path)

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": confidence_score,
        "filename" : path
    }

   



