import cv2
import mediapipe as mp

video = cv2.VideoCapture(2)
if not video.isOpened():
    print("No video")

while (True):
    ret, frame = video.read()
    if not ret:
        print("No frames")
        break

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
