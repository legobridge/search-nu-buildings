import math
import os

import cv2 as cv
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# --------------- SIFT ---------------


def get_sift_keypoints(img, resize_width=1366):
    dsize = (resize_width, int(img.shape[0] / (img.shape[1] / resize_width)))
    img = cv.resize(img, dsize)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(nfeatures=1500)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def create_sift_database():
    labels = []
    features = []
    for image_name in os.listdir('images'):
        if '.jpeg' not in image_name and '.jpg' not in image_name:
            continue
        index_of_dot = image_name.find('.')
        building_name = image_name[0:index_of_dot]

        image_path = 'images/' + image_name
        img = cv.imread(image_path)

        kp, des = get_sift_keypoints(img)
        labels += [building_name for i in range(len(kp))]

        features.append(np.vstack(des))
    return labels, features


# --------------- BRIEF ---------------


def get_brief_keypoints(img, resize_width=1366):
    dsize = (resize_width, int(img.shape[0] / (img.shape[1] / resize_width)))
    img = cv.resize(img, dsize)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    star = cv.xfeatures2d.StarDetector_create()
    kp = star.detect(gray, None)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp, des = brief.compute(gray, kp)
    return kp, des


def create_brief_database():
    labels = []
    features = []
    for image_name in os.listdir('images'):
        if '.jpeg' not in image_name and '.jpg' not in image_name:
            continue
        index_of_dot = image_name.find('.')
        building_name = image_name[0:index_of_dot]

        image_path = 'images/' + image_name
        img = cv.imread(image_path)

        kp, des = get_brief_keypoints(img)
        labels += [building_name for i in range(len(kp))]

        features.append(np.vstack(des))
    return labels, features


# --------------- ORB ---------------


def get_orb_keypoints(img, resize_width=1366):
    dsize = (resize_width, int(img.shape[0] / (img.shape[1] / resize_width)))
    img = cv.resize(img, dsize)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create(nfeatures=1000)
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp)
    return kp, des


def create_orb_database():
    labels = []
    features = []
    for image_name in os.listdir('images'):
        if '.jpeg' not in image_name and '.jpg' not in image_name:
            continue
        index_of_dot = image_name.find('.')
        building_name = image_name[0:index_of_dot]

        image_path = 'images/' + image_name
        img = cv.imread(image_path)

        kp, des = get_orb_keypoints(img)
        labels += [building_name for i in range(len(kp))]

        features.append(np.vstack(des))
    return labels, features


def create_requested_feature_database(create_database):
    labels, features = create_database()
    features_ar = np.vstack(features)
    faiss_size = list(features_ar.shape)[1]
    index = faiss.IndexFlatL2(faiss_size)
    index.add(features_ar)
    return labels, index


def get_percentage_scores(top_tuple_list, softmax_temp=25):
    # Softmax calculation
    top_list = []
    sum_of_scores = 0
    for score, name in top_tuple_list:
        top_list.append(name)
        sum_of_scores += math.exp(score / softmax_temp)
    percentage_scores = [math.exp(top_tuple_list[i][0] / softmax_temp) * 100 / sum_of_scores for i in
                         range(len(top_tuple_list))]
    return top_list, percentage_scores


def find_closest_image_match(img, k, method, labels, index):
    if method == 'SIFT':
        kp, des = get_sift_keypoints(img)
    elif method == 'BRIEF':
        kp, des = get_brief_keypoints(img)
    elif method == 'ORB':
        kp, des = get_orb_keypoints(img)
    else:
        raise NotImplemented()

    preds = {}
    for d in des:
        D, I = index.search(d.reshape((1, index.d)), k)
        for i, idx in enumerate(I[0]):
            pred = labels[idx]
            if pred not in preds:
                preds[pred] = 0
            preds[pred] += 1 / (i + 1)
    top_tuple_list = sorted([(v, k) for k, v in preds.items()], reverse=True)
    scores = [top_tuple[0] for top_tuple in top_tuple_list]
    top_tuple_list = [(v - max(scores), k) for v, k in top_tuple_list]
    top_list, percentage_scores = get_percentage_scores(top_tuple_list)
    return top_list, percentage_scores


def display_results(buildings, img, percentage_confidences, top_list):
    # Primary Results
    st.subheader("Results")
    if percentage_confidences[0] >= 50:
        # We probably found it
        predicted_building = top_list[0]

        # Show images
        col1, col2 = st.columns([2, 2])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        with col1:
            st.image(img, width=360, caption='Uploaded Image', use_column_width='never')
        with col2:
            st.image(f'images/{predicted_building}.jpg', width=360, caption='Database Image', use_column_width='never')

        # Output details about the building
        details_df = pd.read_csv('database.csv', index_col='Name')
        st.markdown(f'This looks like **{predicted_building}** (we hope)')
        st.markdown('**Address:** ' + details_df.loc[predicted_building].Address)
        st.markdown('**Google Maps Link:** ' + details_df.loc[predicted_building].Link)
        st.markdown('**Description:** ' + details_df.loc[predicted_building].Description)
    else:
        # We probably didn't find it
        st.markdown("**Hmm, we're not quite sure what this is...**")

    # Probability Distribution
    top_k = 5
    st.subheader(f"Top {top_k} Image Matches")
    y_pos = np.arange(top_k)
    fig, ax = plt.subplots()
    ax.barh(y_pos, percentage_confidences[:top_k], align='center')
    ax.set_yticks(y_pos, labels=buildings[:top_k])
    ax.invert_yaxis()
    ax.set_xlabel('Confidence (%)')
    st.pyplot(fig)


def main():
    # Set page title and layout
    st.set_page_config(page_title="Image Matching App", layout="wide")
    st.title('Northwestern University Buildings')
    st.subheader('Search by Image')

    method = st.radio('Select Feature-Extraction Algorithm to use:',
                      options=['SIFT', 'BRIEF', 'ORB'])

    # Create feature database
    if method == 'SIFT':
        labels, index = create_requested_feature_database(create_sift_database)
    elif method == 'BRIEF':
        labels, index = create_requested_feature_database(create_brief_database)
    elif method == 'ORB':
        labels, index = create_requested_feature_database(create_orb_database)
    else:
        raise NotImplemented()

    # Upload image file
    uploaded_file = st.file_uploader("Upload Image of an NU Building")

    # Create uploaded files directory if it doesn't exist
    uploaded_files_dir = 'uploaded_files'
    if not os.path.exists(uploaded_files_dir):
        os.mkdir(uploaded_files_dir)

    # Process uploaded image if available
    if uploaded_file is not None:
        # Save uploaded file
        file_path = f'{uploaded_files_dir}/{uploaded_file.name}'
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Read and process uploaded image
        img = cv.imread(file_path)
        buildings = []
        percentage_confidences = []
        top_list, percentage_scores = find_closest_image_match(img, 10, method, labels, index)

        # Collect results
        for i in range(len(top_list)):
            name = top_list[i]
            percentage_score = percentage_scores[i]
            buildings.append(name)
            percentage_confidences.append(float(f'{percentage_score:.2f}'))

        # Display results
        display_results(buildings, img, percentage_confidences, top_list)


if __name__ == "__main__":
    main()
