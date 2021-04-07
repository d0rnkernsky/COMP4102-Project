import stitcher as st
import object_removal as rm
import cv2 as cv

if __name__ == '__main__':
    images = []

    # Read sample images from local directory
    images.append(cv.imread(r'Images/Example_3/1.jpg'))
    images.append(cv.imread(r'Images/Example_3/2.jpg'))
    images.append(cv.imread(r'Images/Example_3/3.jpg'))
    images.append(cv.imread(r'Images/Example_3/4.jpg'))

    # Build panorama using sample images
    st.build_panorama(images)
    rm.remove('stitch-result.jpg')