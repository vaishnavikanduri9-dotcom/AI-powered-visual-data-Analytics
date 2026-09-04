import streamlit as st 
from ultralytics import YOLO 
from PIL import Image 
import numpy as np
#title
st.set_page_config(
   page_title="PPE Detection",
   page_icon="🦺",
   layout="wide"
)
st.title("🦺 PPE Detection using YOLOv8")
model=YOLO("best.pt")
uploaded_file=st.file_uploader(
    "upload a construction worker image",
    type=["jpg","jpeg","png"]
)
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.subheader("original image")
    st.image(image, use_container_width=True)
    img_array=np.array(image)
    results=model(img_array)
    result_image=results[0].plot()
    st.subheader("PPE Detection Result")
    st.image(result_image,use_container_width=True)
    st.subheader("Detected PPE")
    boxes=results[0].boxes
    if boxes is not None and len(boxes)>0:
        for box in boxes:
            class_id=int(box.cls[0])
            confidence=float(box.conf[0])
            class_name=model.names[class_id]
            st.write(
                f"**{class_name}**→"
                f"confidence:{confidence:.2%}"
            )
    else:
        st.warning("No PPE Detected")