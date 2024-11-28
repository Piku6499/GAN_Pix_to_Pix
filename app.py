from flask import Flask, render_template, request
import pickle
from keras.preprocessing.image import img_to_array, array_to_img, load_img
import base64
import os
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Load the model once when the app starts
model = pickle.load(open("gen_model.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    in_image_base64 = None
    out_image_base64 = None

    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the uploaded image to the 'uploads' directory
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Load and process the image directly from the saved file
            img = load_img(file_path, target_size=(256, 512))
            img_array = img_to_array(img)

            # Extract the satellite image
            sat_img = img_array[:, :256]
            img = array_to_img(sat_img)
            sat_img = (sat_img - 127.5) / 127.5
            sat_img = np.expand_dims(sat_img, axis=0) 

            # Generate the map image using the model
            gen_img = model.predict(sat_img)
            gen_img = (gen_img + 1) / 2.0  # Rescale to [0, 1]
            gen_img = np.squeeze(gen_img, axis=0)  

            # Convert the generated image to a PIL image
            gen_img_pil = array_to_img(gen_img)
            gen_image_filename = "generated_image.jpg"
            gen_image_path = os.path.join('uploads', gen_image_filename)
            gen_img_pil.save(gen_image_path)

            # Convert the input image to base64 for displaying in the browser
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            in_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Convert the generated image to base64 for displaying in the browser
            buffered = BytesIO()
            gen_img_pil.save(buffered, format="JPEG")
            out_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('index.html', in_image=in_image_base64, out_image=out_image_base64)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

