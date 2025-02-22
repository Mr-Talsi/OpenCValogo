import cv2
import numpy as np

# Ouvrir la webcam
cap = cv2.VideoCapture(0)
def changeRes(width,height):
#livevideo
    cap.set(3,width)
    cap.set(4,height)
    return width,height

width,height=changeRes(1000,1000)


while True:
    ret, frame = cap.read()
    frame=cv2.flip(frame, 1)
    frame=cv2.GaussianBlur(frame,(21,21),0)

    if not ret:
        break

    # Convertir en HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Définir la plage de couleur pour l'orange
    # lower_bound = np.array([5, 100, 100])
    # upper_bound = np.array([15, 255, 255])
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([150, 255, 40])
    # lower_bound = np.array([100, 50, 50])
    # upper_bound = np.array([130, 255, 255])
    # Créer un masque pour détecter la couleur
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Trouver les contours de l'objet
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Variable pour stocker la plus grande orange
    max_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 13000 and area > max_area:  # Filtrer les petits bruits et trouver le plus grand contour
            max_contour = contour
            max_area = area

    if max_contour is not None:
        # Dessiner un rectangle autour de l'objet
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Dessiner un cercle au centre de l'objet
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Afficher les coordonnées de l'objet
        cv2.putText(frame, f"X: {cx}, Y: {cy}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher les résultats
    cv2.imshow("Webcam",frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
