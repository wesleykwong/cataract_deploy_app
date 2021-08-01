# -*- coding: utf-8 -*-
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array

import imageio
from mlxtend.image import extract_face_landmarks

st.sidebar.title("Settings")

language = st.sidebar.selectbox('', ['English', 'Spanish'])

if language == 'English':
    st.title("""Cataract Detection App""")
    intro_text = """
    Authors: Arda Mark, Casey Yoon, Daniel Lampert, Richard Du, Wesley Kwong (5th Year UC Berkeley MIDS)

    Cataract is the world's leading cause of blindness with over 51% of the world's population affected. In Mexico alone, it is estimated to be the cause of 63% of all blindness and affects over 10 million people. It is characterized by a cloudy area in the lens of the eye that worsens your vision over time. We created this app that uses AI to allow people to self diagnose themselves for cataract. However, we recommend seeing a medical professional if possible.

    Please insert a picture of your entire face with your eyes as wide open as possible with no head accessories. There should be minimal to no glare in the eyes as this will affect the diagnosis.
    """
else:
    st.title("""Detección de Cataratas""")
    intro_text = """
    Autores: Arda Mark, Casey Yoon, Daniel Lampert, Richard Du, Wesley Kwong (5th Year UC Berkeley MIDS)

    La catarata es la principal causa de ceguera en el mundo, con más del 51% de la población mundial afectada. Solo en México, se estima que es la causa del 63% de toda la ceguera y afecta a más de 10 millones de personas. Se caracteriza por un área nublada en el cristalino del ojo que empeora la visión con el tiempo. Creamos esta aplicación que utiliza la IA para permitir que las personas se autodiagnostiquen a sí mismas para la catarata. Sin embargo, recomendamos ver a un profesional médico si es posible.

    Por favor, inserte una imagen de toda su cara con los ojos tan abiertos como sea posible con accesorios mínimos para la cabeza. Debe haber un deslumbramiento mínimo o nulo en los ojos, ya que esto afectará el diagnóstico.
    """

st.write(intro_text)

uploaded_file = st.file_uploader("")

if language == 'English':
    if not uploaded_file:
        st.warning('Please input a picture above.')
        st.stop()
    st.success('Thank you for inputting a picture! Please wait a few moments for your result.')
else:
    if not uploaded_file:
        st.warning('Por favor, introduzca una imagen de arriba.')
        st.stop()
    st.success('Gracias por introducir una foto! Por favor, espere unos momentos para su resultado.')

file_array = imageio.imread(uploaded_file)
landmarks = extract_face_landmarks(file_array)

def cropped_image2(image, landmarks, which_eye="left"):

  # Get the four polygon coordinates around the eye
  # to make a square around the iris

  # Also calcuates the "center" of the eye using all 6 landmarks
  if which_eye == "left":
    left_eye_4l = np.array([37, 38, 40, 41])
    coordinates = [landmarks[i] for i in left_eye_4l]

    left_eye = np.array([36, 37, 38, 39, 40, 41])
    eye_center = np.mean(landmarks[left_eye], axis=0)
  elif which_eye == "right":
    right_eye_4l = np.array([43, 44, 46, 47])
    coordinates = [landmarks[i] for i in right_eye_4l]

    right_eye = np.array([42, 43, 44, 45, 46, 47])
    eye_center = np.mean(landmarks[right_eye], axis=0)
  else:
    print("Choose left or right eye")
    return

  top_left_x = np.min([coordinates[0][0],
                       coordinates[1][0],
                       coordinates[2][0],
                       coordinates[3][0]])
  top_left_y = np.min([coordinates[0][1],
                       coordinates[1][1],
                       coordinates[2][1],
                       coordinates[3][1]])
  bot_right_x = np.max([coordinates[0][0],
                        coordinates[1][0],
                        coordinates[2][0],
                        coordinates[3][0]])
  bot_right_y = np.max([coordinates[0][1],
                        coordinates[1][1],
                        coordinates[2][1],
                        coordinates[3][1]])

  half_width = (bot_right_x - top_left_x)/1.5 #2
  half_height = (bot_right_y - top_left_y)/1.5 #2

  top_left_x = int(eye_center[0]-half_width)
  top_left_y = int(eye_center[1]-half_height)
  bot_right_x = int(eye_center[0]+half_width)
  bot_right_y = int(eye_center[1]+half_height)

  return image[top_left_y:bot_right_y, top_left_x:bot_right_x]

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def build_model():
    inputs = layers.Input(shape=(380, 380, 3))
    model = ResNet50(include_top=False, input_tensor=img_augmentation(inputs), weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D()(model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    #model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = build_model()
model.load_weights("resnet50_bal_train.h5") #.expect_partial()

def make_prediction(img):
    "Makes prediction on individual image"
    #img = Image.fromarray(img)
    #img = img.resize((380, 380))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(x = img, verbose = 0)
    class_idx = np.argmax(prediction, axis = 1)[0]
    class_name = ['Normal', 'Cataract'][class_idx]
    prob = prediction[0][class_idx]
    return class_name, prob

left_eye = cropped_image2(file_array, landmarks, which_eye="left")
right_eye = cropped_image2(file_array, landmarks, which_eye="right")
left_eye_rescaled = Image.fromarray(left_eye).resize((380, 380))
right_eye_rescaled = Image.fromarray(right_eye).resize((380, 380))
left_pred_class, left_prob = make_prediction(left_eye_rescaled)
right_pred_class, right_prob = make_prediction(right_eye_rescaled)

col1, col2 = st.beta_columns(2)
if language == 'English':
    with col1:
        st.subheader("Left Eye")
        left_prob_rounded = str(np.round(left_prob, 3))
        left_result_caption = "Prediction: **{0}** with a probability of {1}".format(left_pred_class, left_prob_rounded)
        st.image(left_eye_rescaled)
        st.markdown(left_result_caption)

    with col2:
        st.subheader("Right Eye")
        right_prob_rounded = str(np.round(right_prob, 3))
        right_result_caption = "Prediction: **{0}** with a probability of {1}".format(right_pred_class, right_prob_rounded)
        st.image(right_eye_rescaled)
        st.markdown(right_result_caption)

    st.header("Saliency Mapping")

    saliency_text = """
    What is the AI model looking at? The saliency map reveals which pixels the model considers important during cataract diagnosis.
    """

else:
    with col1:
        st.subheader("Ojo izquierdo")
        left_prob_rounded = str(np.round(left_prob, 3))

        if left_pred_class == "Cataract":
            left_pred_class = "Catarata"

        left_result_caption = "Predicción: **{0}** con una probabilidad de {1}".format(left_pred_class, left_prob_rounded)
        st.image(left_eye_rescaled)
        st.markdown(left_result_caption)

    with col2:
        st.subheader("Ojo Derecho")
        right_prob_rounded = str(np.round(right_prob, 3))

        if right_pred_class == "Cataract":
            right_pred_class = "Catarata"

        right_result_caption = "Predicción: **{0}** con una probabilidad de {1}".format(right_pred_class, right_prob_rounded)
        st.image(right_eye_rescaled)
        st.markdown(right_result_caption)

    st.header("Mapeo de saliencia")

    saliency_text = """
    ¿Qué está mirando el modelo de IA? El mapa de prominencia revela qué píxeles considera importantes el modelo durante el diagnóstico de cataratas.
    """

st.write(saliency_text)

def saliency_map(img):
  img = img_to_array(img)
  img = img.reshape((1, *img.shape))
  img = img[0].reshape(1, 380, 380, 3)
  img = tf.Variable(img, dtype=float)

  with tf.GradientTape() as tape:
    pred = model(img, training=False)
    #pred = model.predict(x = img, verbose = 0)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]

  grads = tape.gradient(loss, img)
  dgrad_abs = tf.math.abs(grads)
  dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

  ## normalize to range between 0 and 1
  arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
  grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
  fig, ax = plt.subplots() 
  ax = plt.imshow(grad_eval, cmap='hot', alpha=1)
  plt.axis('off')
  plt.margins(0, 0)
  plt.figure(constrained_layout=True)
  return fig

col3, col4 = st.beta_columns(2)
with col3:
    left_eye_saliency = saliency_map(left_eye_rescaled)
    #st.image(left_eye_saliency)
    st.pyplot(left_eye_saliency)
with col4:
    right_eye_saliency = saliency_map(right_eye_rescaled)
    st.pyplot(right_eye_saliency)

if language == 'English':
    st.header("What should you do if you have cataract?")

    prevention_text = """
    We highly recommend seeking out a medical professional for further followup and treatment.

    However, if you are unable to access medical treatment we recommend the following to slow the progression of cataract:
    - Quit smoking
    - Eat lots of fruits and vegetables
    - Wear sunglasses
    - Reduce alcohol use
    """
    st.write(prevention_text)

    st.header("Free Cataract Surgery in Mexico")

    organization_text = """
    - Vamos Viendo
        - (55) 6503 7852
    - Fundación Ale y Fundación Cinepolis
        - (55) 5626 3708
    - Lions Club International's SightFirst Program
        - https://www.lionsclubs.org/en/start-our-approach/club-locator
    """
    st.write(organization_text)

else:
    st.header("¿Qué debe hacer si tiene cataratas?")

    prevention_text = """
    Recomendamos encarecidamente buscar un profesional médico para un mayor seguimiento y tratamiento.

    Sin embargo, si no puede acceder al tratamiento médico, le recomendamos lo siguiente para retardar la progresión de la catarata:
    - Dejar de fumar
    - Comer muchas frutas y verduras
    - Usar gafas de sol
    - Reducir el consumo de alcohol
    """
    st.write(prevention_text)

    st.header("Cirugía gratuita de cataratas en México")

    organization_text = """
    - Vamos Viendo
        - (55) 6503 7852
    - Fundación Ale y Fundación Cinepolis
        - (55) 5626 3708
    - Lions Club International's SightFirst Program
        - https://www.lionsclubs.org/en/start-our-approach/club-locator
    """
    st.write(organization_text)
