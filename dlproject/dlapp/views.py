from django.shortcuts import render
from PIL import Image
from keras.preprocessing import image as keras_image
import io
import numpy as np
from keras.models import load_model

def homepage(request):
    return render(request,"index.html")


def predict(request):
    if request.method == 'POST':
        file = request.FILES['image']
        image = Image.open(file)
        img_io = io.BytesIO()
        image.save(img_io, format='JPEG')
        img_io.seek(0)

        img = keras_image.load_img(img_io, target_size=(28, 28))
        img = keras_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        # Normalize the image
        img = img / 255.0
        model = load_model('D:\SEMISTER 6\DL\DL_Main_Project\my_model.h5')
        prediction = model.predict(img)
        if prediction >= 0.2:
            result = 'Glaucoma'
        else:
            result = 'No Glaucoma'

        return render(request, 'result.html', {'result': result})

def result(request):
    return render(request,"result.html")