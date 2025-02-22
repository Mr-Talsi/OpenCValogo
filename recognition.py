
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_faces.xml')

people = ['Ahmed','yassmine']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
cap = cv.VideoCapture(0)
def changeRes(width,height):
#livevideo
    cap.set(3,width)
    cap.set(4,height)
    return width,height

width,height=changeRes(1000,1000)


while True:
    ret, frame = cap.read()
    frame=cv.flip(frame, 1)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # # Detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 1)
    for (x,y,w,h) in faces_rect:
        area = w*h
        if area > 13000 :
            faces_roi = gray[y:y+h,x:x+w]
            label, confidence = face_recognizer.predict(faces_roi)
            print(f'Label = {people[label]} with a confidence of {confidence}')
            if confidence < 48:
                cv.putText(frame, str(people[label]), (x,y-20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)

            cv.imshow('Detected Face', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv.destroyAllWindows()