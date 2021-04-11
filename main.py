import stitcher as st
import object_removal as rm
import cv2 as cv

if __name__ == '__main__':
    images = []

    # Read sample images from local directory
    images.append(cv.imread(r'Images/ex5/1.jpg'))
    images.append(cv.imread(r'Images/ex5/2.jpg'))
    images.append(cv.imread(r'Images/ex5/3.jpg'))
    images.append(cv.imread(r'Images/ex5/4.jpg'))
    images.append(cv.imread(r'Images/ex5/5.jpg'))

    # Build panorama using sample images
    stitch_result = st.build_panorama(images)

    # Convert RGB image to BGR (for opencv compatibility)
    result = cv.cvtColor(stitch_result, cv.COLOR_RGB2BGR)

    # Save panorama to file
    img_name = 'stitch-result.jpg'
    cv.imwrite(img_name, result)
    rm.remove(img_name)