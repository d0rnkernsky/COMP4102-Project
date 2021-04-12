# Imports
import numpy as np                       # Matrix operations
import imutils                           # Image processing functions
import cv2                               # OpenCV
from matplotlib import pyplot as plt     # Graph plotting

# Function for building panorama from array of images
def build_panorama(images):
    '''
    Build panorama from array of images.

    Parameters:
    -----------
    images:         array of OpenCV images
        Array of images to be stitched together for panorama.

    Returns: None
    '''

    # Create dictionary for images info
    imageDict = {}

    # Iterate through images, adding info to nested dictionary
    for i in range(1, len(images) + 1):

        # Initialize image dictionary
        imageDict = initialize_image(images, imageDict, None, i = i)

    # Iterate through images to build panorama
    for i in range(1, len(images) + 1):

        # If not first image, set left image to stitched result
        if (i == 1):
            leftImage = imageDict["image_1"]
        else:
            imageDict = initialize_image(images, imageDict, stitchedImage)
            leftImage = imageDict["stitched"]

        # Build panorama on right neighbour (if exists)
        if (i != len(images)):
            
            # Create variable for right neighbour to simplify code
            rightImage = imageDict["image_" + str(i+1)]

            # Create descriptor matcher
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            
            # Find matches for each descriptor
            matches = matcher.match(leftImage["features"], rightImage["features"])

            # Sort matches by distance
            good_matches = sorted(matches, key = lambda x: x.distance)

            # Compute homography matrix with >= 4 good matches
            if len(good_matches) >= 4:

                # Get points corresponding to good matches
                leftPoints = []
                rightPoints = []
                for m in range(0, len(good_matches)):
                    # Check that keypoint exists
                    if (len(leftImage["keypoints"]) >= good_matches[m].trainIdx):
                        rightPoints.append(rightImage["keypoints"][good_matches[m].trainIdx].pt)
                        leftPoints.append(leftImage["keypoints"][good_matches[m].queryIdx].pt)

                # Convert points to float
                rightPoints = np.float32(rightPoints).reshape(-1, 1, 2)
                leftPoints = np.float32(leftPoints).reshape(-1, 1, 2)

                # UNCOMMENT to draw keypoints on images + display
                
                '''
                # Draw circles over original images
                img1_with_points = rightImage["grey"].copy()
                img2_with_points = leftImage["grey"].copy()

                plt.imshow(img1_with_points, cmap="gray")
                plt.show()
                plt.imshow(img2_with_points, cmap="gray")
                plt.show()

                for p in rightPoints:
                    img1_with_points = cv2.circle(img1_with_points, (p[0][0],p[0][1]), 10, (255, 0, 0), thickness=3)
                for p in leftPoints:
                    img2_with_points = cv2.circle(img2_with_points, (p[0][0],p[0][1]), 10, (255, 0, 0), thickness=3)

                # Display images with overlayed circles
                plt.imshow(img1_with_points, cmap="gray")
                plt.show()
                plt.imshow(img2_with_points, cmap="gray")
                plt.show()
                '''

                # Get homography matrix between left image and right neighbour
                # (using RANSAC)
                (h_matrix, mask) = cv2.findHomography(rightPoints, 
                                                      leftPoints,
                                                      cv2.RHO)

            # Throw error if not enough good matches are found
            else:
                print("Not enough good matches. Cannot create panorama.\n")
                return

            # Warp right image using homography matrix
            warpedImage = cv2.warpPerspective(rightImage["image"], 
                                            h_matrix,
                                            (rightImage["image"].shape[1] + leftImage["image"].shape[1],
                                            rightImage["image"].shape[0]))


            # UNCOMMENT to display warped image
            '''
            plt.imshow(warpedImage)
            plt.show()
            '''

            # Get min height of left and right image
            min_height = min(warpedImage.shape[1], leftImage["image"].shape[1])

            # Stitch left image into right image
            stitchedImage = warpedImage
            stitchedImage[0:leftImage["image"].shape[0], 0:min_height] = leftImage["image"]

            # Crop black background from stitched image
            grey = cv2.cvtColor(stitchedImage, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(grey, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            stitchedImage = stitchedImage[y:y+h, x:x+w]

            # Show stitched image
            #plt.imshow(stitchedImage)
            #plt.show()

            # Clear some memory
            del leftImage
            if (i == 1):
                del imageDict["image_1"]
            else:
                del imageDict["stitched"]

    return stitchedImage
            

def initialize_image(images, imageDict, stitched, i=0):
    '''
    Initialize dictionary of images for panorama.

    Parameters:
    -----------
    images:         array of OpenCV images
        Array of images to be stitched together for panorama.

    imageDict:      dictionary
        Dictionary containing information for each image.

    stitched:       OpenCV image or None
        Result from previous iteration of panorama stitching.

    i:              int, default = 0
        Index of image position in panorama, or 0 if stitching has
        already begun.

    Returns: Dictionary containing initialized information for each image.
    '''

    # Check that image stitching hasn't happened yet, in which case
    # we will initialize all the individual images into a dictionary
    if (i != 0):

        # Name image dictionary based on index
        index = "image_" + str(i)

        # Create nested dictionary for current image
        imageDict[index] = {}

        # Resize image for faster processing
        images[i-1] = imutils.resize(images[i-1], width=400)

        # Convert original image to RGB (for matplotlib compatibility)
        images[i-1] = cv2.cvtColor(images[i-1], cv2.COLOR_BGR2RGB)

        # Create variables for current image to simplify code
        thisImage = imageDict[index]
        thisImage["image"] = images[i-1]

        # Convert RGB image to greyscale to detect keypoints
        thisImage["grey"] = cv2.cvtColor(thisImage["image"], cv2.COLOR_RGB2GRAY)

        # Find keypoints and descriptors
        feature_finder = cv2.SIFT_create()
        thisImage["keypoints"], thisImage["features"] = feature_finder.detectAndCompute(thisImage["grey"], None)
    
    # Initialize stitched image information
    else:
        # Find keypoints and descriptors
        feature_finder = cv2.SIFT_create()
        kp, feat = feature_finder.detectAndCompute(stitched, None)

        # Create dictionary entry for stitched image
        imageDict["stitched"] = {}
        thisImage = imageDict["stitched"]

        # Convert RGB image to greyscale to use as left image
        stitched_color = stitched
        stitched = cv2.cvtColor(stitched, cv2.COLOR_RGB2GRAY)

        # Initialize dictionary values
        thisImage["image"] = stitched_color
        thisImage["grey"] = stitched
        thisImage["keypoints"] = kp
        thisImage["features"] = feat

        # Save memory
        del kp, feat

    # Return initialized image dictionary
    return imageDict

def main():

    # Create array to hold sample images
    images = []

    # Read sample images from local directory
    images.append(cv2.imread(r'Images\Example_1\1.jpg'))
    images.append(cv2.imread(r'Images\Example_1\2.jpg'))
    images.append(cv2.imread(r'Images\Example_1\3.jpg'))
    images.append(cv2.imread(r'Images\Example_1\4.jpg'))

    # Build panorama using sample images
    result = build_panorama(images)

    # Display panorama
    plt.imshow(result)
    plt.show()

    # Convert RGB image to BGR (for opencv compatibility)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # Save panorama to file
    cv2.imwrite('result.jpg', result)

if __name__ == "__main__":
    main()
