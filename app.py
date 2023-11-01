import streamlit as st
import cv2
import yaml
from PIL import Image
from deepface import DeepFace
import numpy as np


# Path: code\app.py
st.set_page_config(layout="wide")


red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)
black_color = (0, 0, 0)
white_color = (255, 255, 255)

# font
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 10


def draw_result(text, color, img_face, img_card):
    font_scale = 5
    box_width = face_rb_point[0] - face_lt_point[0]
    (text_width, text_height) = cv2.getTextSize(
        text, font, fontScale=font_scale, thickness=1
    )[0]
    while text_width > box_width:
        font_scale = font_scale - 0.1
        (text_width, text_height) = cv2.getTextSize(
            text, font, fontScale=font_scale, thickness=1
        )[0]
    thickness = 10
    box_coords = (
        (face_lt_point[0], face_lt_point[1]),
        (
            face_rb_point[0],
            face_lt_point[1] - text_height,
        ),
    )

    cv2.rectangle(img_face, box_coords[0], box_coords[1], color, cv2.FILLED)
    cv2.rectangle(img_face, box_coords[0], box_coords[1], color, thickness=thickness)
    cv2.rectangle(
        img=img_card,
        pt2=card_rb_point,
        pt1=card_lt_point,
        color=color,
        thickness=thickness,
    )
    cv2.rectangle(
        img=img_face,
        pt1=face_lt_point,
        pt2=face_rb_point,
        color=color,
        thickness=thickness,
    )
    cv2.putText(
        img=img_face,
        text=text,
        org=box_coords[0],
        fontFace=font,
        fontScale=font_scale,
        color=white_color,
        thickness=5,
    )
    cv2.putText(
        img=img_face,
        text=text,
        org=box_coords[0],
        fontFace=font,
        fontScale=font_scale,
        color=black_color,
        thickness=5,
    )


# add  logo
with st.sidebar:
  st.image("./logo.png", width=300)

### add some slidebar menu
menu = ["Picture","Webcam"]
choice = st.sidebar.selectbox("Input type",menu)
st.sidebar.date_input("Current Date")


#Config
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']

# st.sidebar.title("Settings")

# menu = ["Picture","Webcam"]
# choice = st.sidebar.selectbox("Input type",menu)

# if choice == "Picture":
st.title("AI Face Matching App")
st.write(PICTURE_PROMPT)
column1, column2 = st.columns(2)
with column1:
    image1 = st.file_uploader("Reference Image", type=['jpg','png', 'jpeg'])
with column2:
    image2 = st.file_uploader("Target Image", type=['jpg','png','jpeg'])

if (image1 is not None) & (image2  is not None):
    col1, col2 = st.columns(2)

    ### cover PIL image into numpy array
    image1_org=np.array(Image.open(image1))
    image2_org=np.array(Image.open(image2))
    
    ### conver the Pillow: RGB to cv2: BGR formate
    image1 = cv2.cvtColor(image1_org, cv2.COLOR_RGBA2BGR)
    image2 = cv2.cvtColor(image2_org, cv2.COLOR_RGBA2BGR)

    ## dispaly input images on app
    with col1:
        st.image(image1_org, width=600)
    with col2:
        st.image(image2_org,width=600)

#### Run the deepface algo
with column2:
    if st.button("Plesae Click for Face Varification"):
        result =  DeepFace.verify(img1_path=image1,img2_path=image2,detector_backend="opencv",)
        
        # get face area
        card_lt_point = (
        result["facial_areas"]["img1"]["x"],
        result["facial_areas"]["img1"]["y"],
        )
        card_rb_point = (
        result["facial_areas"]["img1"]["x"] + result["facial_areas"]["img1"]["w"],
        result["facial_areas"]["img1"]["y"] + result["facial_areas"]["img1"]["h"],
        )

        face_lt_point = (
        result["facial_areas"]["img2"]["x"],
        result["facial_areas"]["img2"]["y"],
        )
        face_rb_point = (
        result["facial_areas"]["img2"]["x"] + result["facial_areas"]["img2"]["w"],
        result["facial_areas"]["img2"]["y"] + result["facial_areas"]["img2"]["h"],
        )

        if result["verified"]:
            st.write("FACE IS MATCHED")
            #  draw_result("MATCHED", green_color, image1, image2)
        else:
            st.write("FACE IS NOT MATCHED")
            # draw_result("UNMATCHED", red_color, image1, image2)
        
        # cv2.namedWindow("image1", flags=cv2.WINDOW_NORMAL)
        # cv2.namedWindow("image2", flags=cv2.WINDOW_NORMAL)

        # cv2.imshow("img_card", image1)
        # cv2.imshow("img_face", image2)

        # cv2.imwrite("./person2_result5.jpg", image1)
        # cv2.imwrite("./person2_card.jpg", image2)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



    

