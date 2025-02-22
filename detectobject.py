import cv2 as cv
import numpy as np

# Ouvrir la webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aplatir l'image en une liste de pixels RGB
    gray = frame.reshape((-1, 3))
    gray = gray.astype(np.float32)
    
    # Paramètres de K-means
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4  # Nombre de clusters
    ret, labels, centers = cv.kmeans(gray, k, None, criteria, 3, cv.KMEANS_PP_CENTERS)
    
    # Reshape labels et mise à l'échelle pour l'affichage
    labels = labels.reshape(frame.shape[0], frame.shape[1])
    labels=labels*255/(k-1)
    labels = (labels).astype(np.uint8)
    print(labels)
    # Trouver les contours sur l'image binaire
    _, binary = cv.threshold(labels, 10, 255, cv.THRESH_BINARY)  # Convertir en binaire
    contours, _ = cv.findContours(binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow("binary", binary)
    # Dessiner les contours et objets détectés
    for contour in contours:
        if cv.contourArea(contour) > 1000:  # Filtrer les petits bruits
            # Dessiner un rectangle autour de l'objet
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Dessiner un cercle au centre de l'objet
            cx, cy = x + w // 2, y + h // 2
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Afficher les coordonnées de l'objet
            cv.putText(frame, f"X: {cx}, Y: {cy}", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher l'image avec détection
    cv.imshow("Webcam", frame)

    # Quitter avec 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv.destroyAllWindows()
