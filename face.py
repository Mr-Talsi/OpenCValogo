import cv2 as cv
cap = cv.VideoCapture(0)
def changeRes(width,height):
    cap.set(3,width)
    cap.set(4,height)
    return width,height

width,height=changeRes(1000,1000)
while True:
    ret, frame = cap.read()
    frame=cv.flip(frame, 1)
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier('haar_faces.xml')
    gray=cv.GaussianBlur(gray,(31,31),0)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        cx, cy = x + w // 2, y + h // 2
        cv.putText(frame, f"X: {cx}, Y: {cy}", (x, y - 10),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv.imshow("Webcam", frame)
    # cv2.imshow("Masque", mask)
    # Quitter avec 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
