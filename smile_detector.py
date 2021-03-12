import cv2

trained_face = cv2.CascadeClassifier('haarcascade.xml')
smile_face = cv2.CascadeClassifier('haarcascade_smile.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face.detectMultiScale(frame_grayscale)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 50), 10)

    the_face = frame[y:y+h, x:x+w]
    face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
    smile_coordinates = smile_face.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

    if len(smile_coordinates) > 0:
        cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    cv2.imshow('Python Smile Detector App', frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
    
webcam.release()
cv2.destroyAllWindows()
print("Code executed")
