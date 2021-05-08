import cv2
import numpy as np
from random import randint

image = cv2.imread('ahte_test_binary_images/book1_page11.png')
# image = cv2.resize(image, (image.shape[1], image.shape[0]*2))
cv2.imwrite('FinalOutput/origin.png', image)
imgcpy = image.copy()
imgcpy1 = image.copy()
imgcpy2 = image.copy()
imgcpy3 = image.copy()

image = cv2.GaussianBlur(image, (5, 5), 1)
cv2.imwrite('FinalOutput/gaussian.png', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("FinalOutput/grey.jpg", gray)

gray = np.array(255 * (gray / 255)**1, dtype='uint8')
cv2.imwrite("FinalOutput/greyImgGamaCorrelation.jpg", gray)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('FinalOutput/otsus.png', thresh)

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 400:
        cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)
cv2.imwrite('FinalOutput/nonoise.png', thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,5))
line_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35,1))
close = cv2.morphologyEx(line_img, cv2.MORPH_CLOSE, close_kernel, iterations=3)
cv2.imwrite('FinalOutput/lineIMG.png', close)

kernel = np.ones((1,60), np.uint8)
close = cv2.erode(close,kernel,iterations=4)	
cv2.imwrite('FinalOutput/erosion.png',close)

blur = cv2.blur(close, (99, 1),0)
cv2.imwrite('FinalOutput/blureImg.tif', blur)

_, imgPart = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
cv2.imwrite('FinalOutput/imgPage.png', imgPart)

cnts = cv2.findContours(imgPart, cv2.RETR_EXTERNAL , 
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 300:
        cv2.drawContours(imgPart, [c], -1, (0,0,0), -1)
cv2.imwrite('FinalOutput/nonoise2.png',imgPart)


kernel = np.ones((3,1), np.uint8)
close =  cv2.dilate(imgPart,kernel, iterations=2)
cnts = cv2.findContours(close, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 10000:
        cv2.drawContours(close, [c], -1, (0,0,0), -1)
cv2.imwrite("FinalOutput/close.png", close)

kernel = np.ones((1,30), np.uint8)
close = cv2.erode(close,kernel,iterations=2)	
cv2.imwrite('FinalOutput/erosion2.png',close)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (140,1))
close = cv2.dilate(close,kernel, iterations=2)
cv2.imwrite("FinalOutput/dilation.png", close)

################################################# Semantic Labeling

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imwrite("v3_out/opening.png", opening)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imwrite("v3_out/sure_bg.png", sure_bg)

rett, labels = cv2.connectedComponents(sure_bg)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0
labeled_img = cv2.addWeighted(imgcpy,0.9,labeled_img,0.7,5)
cv2.imwrite("FinalOutput/labels.png", labeled_img)

##################################################################################

# Draw Rectangles around Lines
(contours, _) = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10000:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(imgcpy,(x-1,y-5),(x+w,y+h),(randint(0, 255),randint(0, 255),randint(0, 255)),5)
        cv2.putText(imgcpy,"Line",(x-50,y),cv2.FONT_HERSHEY_SIMPLEX,1,(209, 80, 0, 255),5)
cv2.imwrite("FinalOutput/imgContoure.png", imgcpy)


########################################################## Watershed
# noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(close,cv2.MORPH_OPEN,kernel, iterations = 2)
# cv2.imwrite("v3_out/opening.png", opening)

# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# cv2.imwrite("v3_out/sure_bg.png", sure_bg)

# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
# cv2.imwrite("v3_out/dist_transform.png", dist_transform)

# ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
# cv2.imwrite("v3_out/sure_fg.png", sure_fg)

# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# cv2.imwrite("v3_out/unknown.png", unknown)

# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# cv2.imwrite("v3_out/markers.png", markers)

# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1

# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
# markers = cv2.watershed(imgcpy,markers)
# cv2.imwrite("v3_out/unknown.png", markers)

# imgcpy[markers == -1] = [0,0,255]
# cv2.imwrite("v3_out/watershed.png", imgcpy)

#####################################################################################

pts = cv2.findNonZero(close)
ret = cv2.minAreaRect(pts)

(cx, cy), (w, h), ang = ret
if w > h:
    w, h = h, w
    ang += 90

## (4) Find rotated matrix, do rotation
M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
rotated = cv2.warpAffine(close, M, (close.shape[1], close.shape[0]))

## (5) find and draw the upper and lower boundary of each lines
hist = cv2.reduce(rotated, 1, cv2.REDUCE_AVG).reshape(-1)

th = 2
H, W = close.shape[:2]
uppers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
lowers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]
c=0
rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
for y in uppers:
    c+=1
    if c%2==0:
        cv2.line(imgcpy2, (0, y), (W, y), (255, 0, 0), 5)
    else:
        cv2.line(imgcpy2, (0, y), (W, y), (0, 0, 255), 5)
c=0
for y in lowers:
    c+=1
    if c%2==0:
        cv2.line(imgcpy2, (0, y), (W, y), (255, 0, 0), 5)
    else:
        cv2.line(imgcpy2, (0, y), (W, y), (0, 0, 255), 5)

cv2.imwrite('FinalOutput/rotated.png', imgcpy2)


##################################################### rlsa functions 

def iteration(image: np.ndarray, value: int) -> np.ndarray:
    """
    This method iterates over the provided image by converting 255's to 0's if the number of consecutive 255's are
    less the "value" provided
    """

    rows, cols = image.shape
    for row in range(0,rows):
        try:
            start = image[row].tolist().index(0) # to start the conversion from the 0 pixel
        except ValueError:
            start = 0 # if '0' is not present in that row

        count = start
        for col in range(start, cols):
            if image[row, col] == 0:
                if (col-count) <= value and (col-count) > 0:
                    image[row, count:col] = 0               
                count = col  
    return image 

def rlsa(image: np.ndarray, horizontal: bool = True, vertical: bool = True, value: int = 0) -> np.ndarray:
    """
    rlsa(RUN LENGTH SMOOTHING ALGORITHM) is to extract the block-of-text or the Region-of-interest(ROI) from the
    document binary Image provided. Must pass binary image of ndarray type.
    """
    
    if isinstance(image, np.ndarray): # image must be binary of ndarray type
        value = int(value) if value>=0 else 0 # consecutive pixel position checker value to convert 255 to 0
        try:
            # RUN LENGTH SMOOTHING ALGORITHM working horizontally on the image
            if horizontal:
                image = iteration(image, value)   

            # RUN LENGTH SMOOTHING ALGORITHM working vertically on the image
            if vertical:
                image = image.T
                image = iteration(image, value)
                image = image.T

        except (AttributeError, ValueError) as e:
            image = None
            print("ERROR: ", e, "\n")
            print('Image must be an np ndarray and must be in "binary". Use Opencv/PIL to convert the image to binary.\n')
            print("import cv2;\nimage=cv2.imread('path_of_the_image');\ngray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);\n\
                (thresh, image_binary) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)\n")
            print("method usage -- rlsa.rlsa(image_binary, True, False, 10)")
    else:
        print('Image must be an np ndarray and must be in binary')
        image = None
    return image
#########################################

image_rlsa_horizontal = rlsa(imgPart, True, False, 20)
cv2.imwrite('FinalOutput/rlsa.png', image_rlsa_horizontal)

close_kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (35,1))
close1 = cv2.morphologyEx(image_rlsa_horizontal, cv2.MORPH_CLOSE, close_kernel1, iterations=3)
kernel1 = np.ones((1,25), np.uint8)
close1 = cv2.erode(close1,kernel1,iterations=2)	
cv2.imwrite('FinalOutput/close1.png',close1)
#opening = cv2.morphologyEx(close1, cv2.MORPH_OPEN, np.ones((1,10), np.uint8))
#cv2.imwrite('FinalOutput/close11.png',opening)
dilation = cv2.dilate(close1,close_kernel1,iterations = 6)
cv2.imwrite('FinalOutput/close111.png',dilation)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(dilation,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imwrite("v3_out/opening2.png", opening)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imwrite("v3_out/sure_bg2.png", sure_bg)

rett, labels = cv2.connectedComponents(sure_bg)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0
labeled_img = cv2.addWeighted(imgcpy3,0.9,labeled_img,0.7,5)
cv2.imwrite("FinalOutput/labels2.png", labeled_img)

# Draw Rectangles around Lines
(contours1, _) = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours1:
    area = cv2.contourArea(cnt)
    if area > 11000:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(imgcpy1,(x-1,y-5),(x+w,y+h),(randint(0, 255),randint(0, 255),randint(0, 255)),5)
cv2.imwrite("FinalOutput/imgContoure1.png", imgcpy1)



################################################################### Projection Profile
# # im = cv2.imread('ahte_test_binary_images/book1_page11.png', cv2.IMREAD_GRAYSCALE)
# # GaussianFilter= cv2.GaussianBlur(im, (5,5), 0)
# # _, binarizedImage = cv2.threshold(GaussianFilter, 127, 255, cv2.THRESH_BINARY)
# close[close == 0] = 0
# close[close == 255] = 1
# horizontal_projection = np.sum(close, axis=1)
# print(horizontal_projection)
# height, width = close.shape
# blankImage = np.zeros((height, width, 3), np.uint8)
# for row in range(height):
#     cv2.line(blankImage, (0,row), (int(horizontal_projection[row]*width/height),row), (255,255,255), 1)
# blankImage = cv2.Canny(blankImage,10,10)
# lines = cv2.HoughLinesP(blankImage, 1, np.pi/180, 30, minLineLength=10, maxLineGap=250)
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(imgcpy, (x1, y1), (x2, y2), (255, 0, 0), 3)
# cv2.imwrite("FinalOutput/blank.png", imgcpy)
################################################################### 