import cv2 as cv
import numpy as np


img = cv.imread("monie.jpg")
img = cv.resize(img, (500, 500))


gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (3, 3), 0)
gray = gray.astype(np.float32)
gray =gray.reshape(-1, 1)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 2  
ret, labels, centers = cv.kmeans(gray, k, None, criteria, 3, cv.KMEANS_PP_CENTERS)
labels = labels.reshape(img.shape[0], img.shape[1])
labels = (labels * 255 / (k - 1)).astype(np.uint8)

contours,_=cv.findContours(labels,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
count=len(contours)
drawn=cv.drawContours(img,contours,-1,(255,0,0),3)
cv.putText(img, f"Nombre trouve: {count}", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


cv.imshow("Image Originale", img)
cv.imshow("Segmentation K-means", labels)


cv.waitKey(0)
cv.destroyAllWindows()
