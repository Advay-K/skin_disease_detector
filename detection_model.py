import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras import layers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

def detect():
    global model
    path = 'output'

    train_dir = os.path.join(path, 'train')
    test_dir = os.path.join(path, 'test')

    train_eczema = os.path.join(train_dir, '1. Eczema 1677')
    train_melanoma = os.path.join(train_dir, '2. Melanoma 15.75k')
    train_atopic = os.path.join(train_dir, '3. Atopic Dermatitis - 1.25k')
    train_basal = os.path.join(train_dir, '4. Basal Cell Carcinoma (BCC) 3323')
    train_melanocytic = os.path.join(train_dir, '5. Melanocytic Nevi (NV) - 7970')
    train_benignKerat = os.path.join(train_dir, '6. Benign Keratosis-like Lesions (BKL) 2624')
    train_psoriasis = os.path.join(train_dir, '7. Psoriasis pictures Lichen Planus and related diseases - 2k')
    train_seborrheic = os.path.join(train_dir, '8. Seborrheic Keratoses and other Benign Tumors - 1.8k')
    train_tinea = os.path.join(train_dir, '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k')
    train_warts = os.path.join(train_dir, '10. Warts Molluscum and other Viral Infections - 2103')
    train_acne = os.path.join(train_dir, 'Acne')
    train_normal = os.path.join(train_dir, 'Normal')

    test_eczema = os.path.join(test_dir, '1. Eczema 1677')
    test_melanoma = os.path.join(test_dir, '2. Melanoma 15.75k')
    test_atopic = os.path.join(test_dir, '3. Atopic Dermatitis - 1.25k')
    test_basal = os.path.join(test_dir, '4. Basal Cell Carcinoma (BCC) 3323')
    test_melanocytic = os.path.join(test_dir, '5. Melanocytic Nevi (NV) - 7970')
    test_benignKerat = os.path.join(test_dir, '6. Benign Keratosis-like Lesions (BKL) 2624')
    test_psoriasis = os.path.join(test_dir, '7. Psoriasis pictures Lichen Planus and related diseases - 2k')
    test_seborrheic = os.path.join(test_dir, '8. Seborrheic Keratoses and other Benign Tumors - 1.8k')
    test_tinea = os.path.join(test_dir, '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k')
    test_warts = os.path.join(test_dir, '10. Warts Molluscum and other Viral Infections - 2103')
    test_acne = os.path.join(test_dir, 'Acne')
    test_normal = os.path.join(test_dir, 'Normal')



    train_img_gen = ImageDataGenerator(rescale = 1./255)
    test_img_gen = ImageDataGenerator(rescale = 1./255)

    generate_train = train_img_gen.flow_from_directory(batch_size=128, directory=train_dir, shuffle=True,
                                                       target_size=(224, 224),
                                                       class_mode='binary')

    generate_test = test_img_gen.flow_from_directory(batch_size=128, directory=test_dir, target_size=(224, 224),
                                                     class_mode= 'binary')


    model = Sequential([
            layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (224, 224, 3)),
            layers.MaxPooling2D(pool_size = (2, 2)),
            layers.Flatten(),
            layers.Dense(10, activation= 'relu'),
            layers.Dense(1, activation = 'sigmoid')
        ])

    model.summary()

    model.compile(optimizer='adam', loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])

    # total_train_size = len(os.listdir(train_eczema)) + len(os.listdir(train_melanoma)) + len(os.listdir(train_atopic)) + len(os.listdir(train_basal))
    # + len(os.listdir(train_melanocytic)) + len(os.listdir(train_benignKerat)) + len(os.listdir(train_psoriasis)) + len(os.listdir(train_seborrheic))
    # + len(os.listdir(train_tinea)) + len(os.listdir(train_warts)) + len(os.listdir(train_acne)) + len(os.listdir(train_normal))
    #
    #
    # total_test_size = len(os.listdir(test_eczema)) + len(os.listdir(test_melanoma)) + len(os.listdir(test_atopic)) + len(os.listdir(test_basal))
    # + len(os.listdir(test_melanocytic)) + len(os.listdir(test_benignKerat)) + len(os.listdir(test_psoriasis)) + len(os.listdir(test_seborrheic))
    # + len(os.listdir(test_tinea)) + len(os.listdir(test_warts)) + len(os.listdir(test_acne)) + len(os.listdir(test_normal))
    #
    # model.fit(generate_train, batch_size=128, epochs=1, validation_data=generate_test)

    return model


def predict(image, model):
    result = model.predict(image)
    #

    if (result[0][0] == 0):
        return "Eczema"
    elif (result[0][0] == 1):
        return "Warts Molluscum and other Viral Infections"
    elif (result[0][0] == 2):
        return "Melanoma"
    elif (result[0][0] == 3):
        return "Atopic Dermatitis"
    elif (result[0][0] == 4):
        return "Basal Cell Carcinoma"
    elif (result[0][0] == 5):
        return "Melanocytic Nevi"
    elif (result[0][0] == 6):
        return "Benign Keratosis-like Lesions"
    elif (result[0][0] == 7):
        return "Psoriasis"
    elif (result[0][0] == 8):
        return "Seborrheic Keratoses and other Benign Tumors"
    elif (result[0][0] == 9):
        return "Tinea Ringworm Candidiasis and other Fungal Infections"
    elif (result[0][0] == 10):
        return "Acne"
    elif (result[0][0] == 11):
        return "Normal Skin"

if __name__ == '__main__':
    detect()