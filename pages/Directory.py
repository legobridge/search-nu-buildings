import os
import streamlit as st
from PIL import Image

imgs = []
building_names = []
img_width = 360
for image_name in sorted(os.listdir('images')):
    if '.jpeg' not in image_name and '.jpg' not in image_name:
        continue
    index_of_dot = image_name.find('.')
    building_name = image_name[0:index_of_dot]
    building_names.append(building_name)

    image_path = 'images/' + image_name
    img = Image.open(image_path)
    dsize = (img_width, int(img.size[1] / (img.size[0] / img_width)))
    img = img.resize(dsize)
    imgs.append(img)

st.image(imgs, width=img_width, caption=building_names)