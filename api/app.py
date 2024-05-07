from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()
MODEL = tf.keras.models.load_model("C:\\Users\\sujoy\\OneDrive\\Desktop\\potato-disease-prediction\\models\\1", compile=False)
#MODEL = tf.keras.models.load_model("../models/1")
CLASS_NAMES = ['Early_blight', 'Late_blight', 'Healthy']

def get_image_data(data) -> np.ndarray:
    image_data = np.array(Image.open(BytesIO(data)))
    return image_data
    
@app.get("/start")
async def ping():
    return "Welcome to potato disease prediction app"

@app.post("/prediction")
async def prediction(
    file: UploadFile = File(...)
):
    image_data = get_image_data(await file.read())
    image_array = np.expand_dims(image_data, 0)
    
    prediction = MODEL.predict(image_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) * 100
    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
    
