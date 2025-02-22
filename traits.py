import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


LIPS_POINTS = [61, 291, 13, 14, 78, 308]  
MOUTH_POINTS = [13, 14, 78, 308]  

EXCLAMATION_THRESHOLD = 0.55  

SMILE_THRESHOLD = 0.46 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    black=np.zeros(frame.shape,dtype='uint8')

    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mouth_points = np.array([(face_landmarks.landmark[i].x * frame.shape[1], 
                                      face_landmarks.landmark[i].y * frame.shape[0]) for i in MOUTH_POINTS])

            vertical_distance = distance.euclidean(mouth_points[0], mouth_points[1])  # Distance verticale (ouverture de la bouche)
            horizontal_distance = distance.euclidean(mouth_points[2], mouth_points[3])  # Distance horizontale (largeur de la bouche)

            oar = vertical_distance / horizontal_distance

            
            cv2.putText(black, f"OAR: {oar:.2f}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            
            if oar > EXCLAMATION_THRESHOLD:
                cv2.putText(black, "Exclamation detecte", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            lips_points = np.array([(face_landmarks.landmark[i].x * frame.shape[1], 
                                     face_landmarks.landmark[i].y * frame.shape[0]) for i in LIPS_POINTS])

        
            d1 = distance.euclidean(lips_points[2], lips_points[3])  # Distance verticale (milieu)
            d2 = distance.euclidean(lips_points[4], lips_points[5])  # Distance verticale (extrémités)
            d3 = distance.euclidean(lips_points[0], lips_points[1])  # Distance horizontale (coins)

            lar = (d1 + d2) / (2 * d3)
            cv2.putText(black, f"LAR: {lar:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if lar > SMILE_THRESHOLD and oar < 0.2:
                cv2.putText(black, "Sourire detecte !", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                mp_drawing.draw_landmarks(
                black, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,  # Pas de points individuels
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                black, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    cv2.imshow("Détection du visage", frame)
    cv2.imshow("Détection des caractéristiques du visage", black)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

