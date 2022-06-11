from flask import Flask, render_template, Response, request
import cv2
import os
from threading import Thread
import detection_model as dm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import tkinter.messagebox



camera = cv2.VideoCapture(0)
capture = 0
detection_model = dm.detect()


app = Flask(__name__)

picsFolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picsFolder

result_val = "Results Will Be Shown Here"


@app.route('/')
def home():
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'icon.png')
    return render_template('index.html', icon_image = pic1, results = result_val)


def gen_frames():
    global capture

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if capture:
                capture = 0
                cv2.imwrite('shots/shot.png', frame)
                img = image.load_img(os.path.join('shots', 'shot.png'), target_size=(224, 224))
                img_array = image.img_to_array(img)
                print(img_array)
                img_batch = np.expand_dims(img_array, axis=0)
                img_preprocessed = preprocess_input(img_batch)

                result = dm.predict(img_preprocessed, detection_model)
                print(result)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera, result_val
    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'icon.png')
    if request.method == 'POST':
        if request.form.get('click') == 'Capture & Detect':
            global capture
            capture=1


    elif request.method == 'GET':
        return render_template('index.html',  icon_image = pic1, results = result_val)

    return render_template('index.html',  icon_image = pic1, results = result_val)


if __name__ == '__main__':
    app.run(debug = True)
