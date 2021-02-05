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

        # Name image dictionary based on index
        index = "image_" + str(i)

        # Create nested dictionary for current image
        imageDict[index] = {}

        # Create variables for current image to simplify code
        thisImage = imageDict[index]
        thisImage["image"] = images[i-1]

        # Convert image to grayscale to detect keypoints
        thisImage["grey"] = cv2.cvtColor(thisImage["image"], cv2.COLOR_BGR2GRAY)

        # Find keypoints using SIFT
        feature_finder = cv2.SIFT_create()
        thisImage["keypoints"] = feature_finder.detect(thisImage["grey"], None)

        # Find features using SIFT
        thisImage["keypoints"] = feature_finder.compute(thisImage["grey"],
                                                        thisImage["keypoints"])[0]
        thisImage["features"] = feature_finder.compute(thisImage["grey"],
                                                        thisImage["keypoints"])[1]

    # Iterate through images to build panorama
    for i in range(1, len(images) + 1):

        # Get current image dictionary based on index
        index = "image_" + str(i)

        # Create variable for current image dictionary to simplify code
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

    # Create array to hold sample images
    images = []

    # Read sample images from local directory
    images.append(cv2.imread(r'Images\Example_1\1.jpg'))
    images.append(cv2.imread(r'Images\Example_1\2.jpg'))
    images.append(cv2.imread(r'Images\Example_1\3.jpg'))
    images.append(cv2.imread(r'Images\Example_1\4.jpg'))
    images.append(cv2.imread(r'Images\Example_1\5.jpg'))
    images.append(cv2.imread(r'Images\Example_1\6.jpg'))

    # Build panorama using sample images
    build_panorama(images)

if __name__ == "__main__":
    main()
