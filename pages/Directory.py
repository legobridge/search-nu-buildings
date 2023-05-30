import os
import cv2 as cv
import streamlit as st

for image_name in os.listdir('images'):
    if '.jpeg' not in image_name and '.jpg' not in image_name:
        continue
    index_of_dot = image_name.find('.')
    building_name = image_name[0:(index_of_dot - 2)]

    image_path = 'images/' + image_name
    img = cv.imread(image_path)

    st.image(img, caption=building_name)