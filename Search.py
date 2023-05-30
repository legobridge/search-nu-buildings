import math
import os

import cv2 as cv
import faiss
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def get_sift_keypoints(img, resize_width=1366):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dsize = (resize_width, int(gray.shape[0] / (gray.shape[1] / resize_width)))
    gray = cv.resize(gray, dsize)
    sift = cv.SIFT_create(nfeatures=2000)
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def create_sift_database():
    labels = []
    features = []
    for image_name in os.listdir('images'):
        if '.jpeg' not in image_name and '.jpg' not in image_name:
            continue
        index_of_dot = image_name.find('.')
        building_name = image_name[0:(index_of_dot - 2)]

        image_path = 'images/' + image_name
        img = cv.imread(image_path)

        kp, des = get_sift_keypoints(img)
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


def get_percentage_scores(top_tuple_list, softmax_temp=50):
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
    if method == 'sift':
        kp, des = get_sift_keypoints(img)

    # elif method == 'orb':
    #     kp, des = get_orb_keypoints(img)
    #
    # elif method == 'brief':
    #     kp, des = get_brief_keypoints(img)

    else:
        return None

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


# def main():
#     labels, index = create_requested_feature_database(create_sift_database)
#     uploaded_file = st.file_uploader("Upload an image file")
#     uploaded_files_dir = 'uploaded_files'
#     if not os.path.exists(uploaded_files_dir):
#         os.mkdir(uploaded_files_dir)
#     if uploaded_file is not None:
#         file_path = f'{uploaded_files_dir}/{uploaded_file.name}'
#         with open(file_path, 'wb') as f:
#             f.write(uploaded_file.read())
#         img = cv.imread(file_path)
#         buildings = []
#         percentage_confidences = []
#         k = 5
#         method = "sift"
#         top_list, percentage_scores = find_closest_image_match(img, k, method, labels, index)
#         for i in range(len(top_list)):
#             name = top_list[i]
#             percentage_score = percentage_scores[i]
#             buildings.append(name)
#             percentage_confidences.append(f'{percentage_score:.2f}')

#         df = pd.DataFrame({'building': buildings, 'percentage_confidence': percentage_confidences})
#         df.iloc[:5]


def main():
    # Set page title and layout
    st.set_page_config(page_title="Image Matching App", layout="wide")
    st.title('NORTHWESTERN IMAGE MATCHING APP')

    # Create database
    labels, index = create_requested_feature_database(create_sift_database)

    # Upload image file
    uploaded_file = st.file_uploader("Upload image file")

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
        k = 5
        method = "sift"
        top_list, percentage_scores = find_closest_image_match(img, k, method, labels, index)

        # Collect results
        for i in range(len(top_list)):
            name = top_list[i]
            percentage_score = percentage_scores[i]
            buildings.append(name)
            percentage_confidences.append(float(f'{percentage_score:.2f}'))

        # Display results
        top_k = 5
        st.subheader(f"Top {top_k} Image Matches")
        y_pos = np.arange(top_k)
        fig, ax = plt.subplots()
        ax.barh(y_pos, percentage_confidences[:top_k], align='center')
        ax.set_yticks(y_pos, labels=buildings[:top_k])
        ax.invert_yaxis()
        ax.set_xlabel('Confidence (%)')
        st.pyplot(fig)
        st.subheader("Uploaded Image")
        st.image(img, use_column_width=True)


if __name__ == "__main__":
    main()
