# Imports
import numpy as np # Matrix operations
import imutils     # Image processing functions
import cv2         # OpenCV

# Function for building panorama from array of images
def build_panorama(images):

    # Create dictionary for images info
    imageDict = {}

    # Iterate through images, adding info to nested dictionary
    for i in range(1, len(images) + 1):

        # Name image based on index
        index = "image_" + i

        # Create nested dictionary for current image
        imageDict[index] = {}

        # Create variables for current image to simplify code
        thisImage = imageDict[index]

        # Detect keypoints
        thisImage[keypoints] = detectAndDescribe(images[i-1])[0]

        # Extract features
        thisImage[features] = detectAndDescribe(images[i-1])[1]

    # Iterate through images to build panorama
    for i in range(1, len(images) + 1):

        # Create variable for current image to simplify code
        thisImage = imageDict[index]

        # Build panorama on right neighbour (if exists)
        if (i != len(images)):
            
            # Create variable for right neighbour to simplify code
            rightImage = imageDict["image_" + i+1]

            # Match features to right neighbour
            M = matchKeypoints(thisImage[keypoints],
                            rightImage[keypoints],
                            thisImage[features],
                            rightImage[features])

def main():
    # Read sample images from local directory
    images[0] = cv2.imread('image1.jpg')
    images[1] = cv2.imread('image2.jpg')
    images[2] = cv2.imread('image3.jpg')
    images[3] = cv2.imread('image4.jpg')
    images[4] = cv2.imread('image5.jpg')
    images[5] = cv2.imread('image6.jpg')

    # Build panorama using sample images
    build_panorama(images)

if __name__ == "__main__":
    main()
