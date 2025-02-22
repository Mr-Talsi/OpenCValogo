import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Initialisation de Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Indices des points clés des lèvres (Mediapipe)
LIPS_POINTS = [61, 291, 13, 14, 78, 308]  # Coins et centre des lèvres

# Seuil de détection du sourire (à ajuster selon le test)
SMILE_THRESHOLD = 0.46  

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion en RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Récupérer les coordonnées des points des lèvres
            lips_points = np.array([(face_landmarks.landmark[i].x * frame.shape[1], 
                                     face_landmarks.landmark[i].y * frame.shape[0]) for i in LIPS_POINTS])

            # Calcul des distances
            d1 = distance.euclidean(lips_points[2], lips_points[3])  # Distance verticale (milieu)
            d2 = distance.euclidean(lips_points[4], lips_points[5])  # Distance verticale (extrémités)
            d3 = distance.euclidean(lips_points[0], lips_points[1])  # Distance horizontale (coins)

            # Calcul du Lip Aspect Ratio (LAR)
            lar = (d1 + d2) / (2 * d3)

            # Affichage du LAR
            cv2.putText(frame, f"LAR: {lar:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Détection du sourire
            if lar > SMILE_THRESHOLD:
                cv2.putText(frame, "Sourire detecte !", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Dessiner les points des lèvres
            for point in lips_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    # Afficher l'image avec la détection
    cv2.imshow("Détection du Sourire", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
