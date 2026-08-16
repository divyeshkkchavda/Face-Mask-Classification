import streamlit as st
from PIL import Image
from streamlit_app.classify_image import classify

def main():
    st.title('Face Mask Classification')

    st.subheader('Upload your image here')

    file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

    if file is not None:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        class_name = classify(image)

        st.write("## {}".format(class_name))

    else:
        st.info('Please upload an image to test')


if __name__ == "__main__":
    main()
