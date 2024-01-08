import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import streamlit as st

def dilation(img, d):
    dilated = np.zeros(img.shape)
    for i in range(int(d / 2), img.shape[0] - int(d / 2)):
        for j in range(int(d / 2), img.shape[1] - int(d / 2)):
            dilated[i, j] = np.amax(img[i - int(d / 2):i + int(d / 2) + 1, j - int(d / 2):j + int(d / 2) + 1])
    return np.abs(dilated)


def erosion(img, d):
    eroded = np.zeros(img.shape)
    for i in range(int(d / 2), img.shape[0] - int(d / 2)):
        for j in range(int(d / 2), img.shape[1] - int(d / 2)):
            eroded[i, j] = np.amin(img[i - int(d / 2):i + int(d / 2) + 1, j - int(d / 2):j + int(d / 2) + 1])
    return np.abs(eroded)


def increaseContrast(image, mask):
    x = np.asarray(image)
    colorImageArray = np.zeros(x.shape)
    for i in range(0, 3):
        img = x[:, :, i]
        # Closing operation
        closed = erosion(dilation(img, mask), mask)
        # Opening operation
        opened = dilation(erosion(img, mask), mask)
        # Top-hat transform : This operation produced an images that contains the
        #  bright elemnts of the image that are smaller than the structuring element.
        Top_hat = np.subtract(img, opened)
        # bright_elements=np.multiply(Top_hat,a)
        # Black-hat transform : This operation produced an images that contains the
        # darker elements of the image that are smaller than the structuring element.
        Black_hat = np.subtract(closed, img)
        # dark_elements=np.multiply(Black_hat,b)
        kernal = np.subtract(Top_hat, Black_hat)
        colorImageArray[:, :, i] = np.clip(np.add(img, kernal), 0, 255)
    colorImageArray = colorImageArray.astype(np.uint8)

    z = Image.fromarray(colorImageArray)

    fig = plt.figure(frameon=False)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(z)
    return z

    # comment on the kernal size(mask)
    # As the kernal size increases the elements that only the large elements
    # of the image will be incorprated in the result and the small details will be lost.

    # In order to reduce contrast
    # In order to reduce contrast we can add the output of the black-hat operation and
    # substract the white-hat opration from the image. Thus excluding the bright elements
    # in the image and including only the darker elements which will result in a lower contrast.


def main():
    # open colour image
    st.title('Top hat Transformation')
    st.sidebar.title('Transformation')
    col1, col2 = st.columns(2)
    image_file = st.sidebar.file_uploader("Upload the image", type=['jpg', 'png', 'jpeg'])
    val = st.sidebar.slider('Select Mask Size', 0, 10, 3)
    if image_file is not None:
        col1.image(image_file, caption='Uploaded Image', use_column_width=True))
        if st.button('Process'):
            chest_x_rays_1 = increaseContrast(image_file, val)
            col2.image(chest_x_rays_1, caption='Processed Image', use_column_width=True)
            
    
    
    #image_file = Image.open('/content/drive/MyDrive/chest_x_rays.jpg')
    # Proposed method
    
    #plt.savefig('chest_x_rays_1.jpg')
    # After increasing mask size
    #chest_x_rays_2 = increaseContrast(image_file, 9)
    #plt.savefig('chest_x_rays_2.jpg')


if __name__ == '__main__':
    main()

