import cv2 
from matplotlib import pyplot as plt
import numpy as np
import easyocr
import imutils

fig, axs = plt.subplots(3, 3, figsize=(15, 10)) #Create a figure with 3 rows and 3 columns

img = cv2.imread("t1.jpg") #read image
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to gray
axs[0, 1].imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)) #show processed image
axs[0, 1].set_title('Grey Image')

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (9, 9), 0) #Apply Gaussian blur, 5x5 kernel, sigma=0, to the gray image
axs[0, 2].imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
axs[0, 2].set_title('Blurred Image')

# Apply Adaptive Thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2) 
edged = cv2.Canny(thresh, 50, 250) #Edge detection

axs[1, 0].imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Edged Image')

keypoints = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Find contours 
contours = imutils.grab_contours(keypoints) #Grab contours 

print("Number of contours found: ", len(contours))

#filter contours based on area, perimeter, and aspect ratio
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)

    if (area > 800) and (perimeter > 100) and (aspect_ratio >= 1.5):
        filtered_contours.append(contour)

contourlist = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:15] #Sort contours

#Loop over our contours to find the best possible approximate contour of 10 contours
location = None
for contour in contourlist:
    epsilon = 0.02 * cv2.arcLength(contour, True) 
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4: 
        location = approx
        print("Contour area: ", cv2.contourArea(contour))
        print("Contour perimeter: ", cv2.arcLength(contour, True))
        break
     
print("Location: ", location)

if location is not None:
    mask = np.zeros(gray.shape, np.uint8) #create blank image with same dimensions as the original image
    new_image = cv2.drawContours(mask, [location], 0,255, -1) #Draw contours on the mask image
    new_image = cv2.bitwise_and(img, img, mask=mask) #Take bitwise AND between the original image and mask image

    axs[1, 1].imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)) #show the final image
    axs[1, 1].set_title('Final Image with Contours')

    (x,y) = np.where(mask==255) #Find the co-ordinates of the four corners of the document
    (x1, y1) = (np.min(x), np.min(y)) #Find the top left corner
    (x2, y2) = (np.max(x), np.max(y)) #Find the bottom right corner
    cropped_image = img[x1:x2+1, y1:y2+1] #Crop the image using the co-ordinates

    axs[1, 2].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)) #show the cropped image
    axs[1, 2].set_title('Cropped Image')

    # Sort the coordinates to ensure they are in the order: top-left, top-right, bottom-right, bottom-left
    location = location.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = location.sum(axis=1)
    rect[0] = location[np.argmin(s)]
    rect[2] = location[np.argmax(s)]

    diff = np.diff(location, axis=1)
    rect[1] = location[np.argmin(diff)]
    rect[3] = location[np.argmax(diff)]

    # Calculate the width and height of the new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Calculate the destination points to which the top-left, top-right, bottom-right, and bottom-left points will be mapped
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Calculate the perspective transform matrix and warp the perspective to grab the document
    M = cv2.getPerspectiveTransform(rect, dst)
    warped_image = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    axs[2, 0].imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)) #show the warped image
    axs[2, 0].set_title('Warped Image')

    reader = easyocr.Reader(['en']) #create an easyocr reader object with english as the language
    result = reader.readtext(warped_image) #read text from the warped image

    if result:
        text = result[0][-2] #Extract the text from the result
        font = cv2.FONT_HERSHEY_SIMPLEX
        res = cv2.putText(img, text=text, org=(int(rect[0][0]), int(rect[1][1])+60), fontFace=font, fontScale=1.2, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA) #put the text on the image
        res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3) #Draw a rectangle around the text

        axs[2, 1].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)) #show the final image with text
        axs[2, 1].set_title('Final Image with Text')
    else:
        print("No text detected by OCR.")
else:
    print("No suitable contour found.")

plt.tight_layout()
plt.show()