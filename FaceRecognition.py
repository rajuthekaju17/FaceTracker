import cv2
from datetime import datetime
import argparse
import os

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_stream = cv2.VideoCapture(0)
face_present = False  # Flag to track face detection in the frame

while True:
    check, frame = video_stream.read()
    if frame is not None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10)
        
        for x, y, w, h in detected_faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            current_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S-%f')
            cv2.imwrite("detected_face_" + str(current_time) + ".jpg", frame)  # Save the frame
            
            # Update flag if a face is detected
            face_present = True

        cv2.imshow("Live Feed", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            if face_present:  # Proceed only if a face was detected
                ap = argparse.ArgumentParser()
                ap.add_argument("-ext", "--extension", required=False, default='jpg')
                ap.add_argument("-o", "--output", required=False, default='output.mp4')
                args = vars(ap.parse_args())

                dir_path = '.'
                image_extension = args['extension']
                video_output = args['output']

                image_files = [f for f in os.listdir(dir_path) if f.endswith(image_extension)]

                if image_files:  # Proceed if there are images to create the video
                    first_image_path = os.path.join(dir_path, image_files[0])
                    frame = cv2.imread(first_image_path)
                    img_height, img_width, channels = frame.shape

                    video_encoder = cv2.VideoWriter_fourcc(*'mp4v')
                    video_out = cv2.VideoWriter(video_output, video_encoder, 5.0, (img_width, img_height))

                    for image in image_files:
                        image_path = os.path.join(dir_path, image)
                        frame = cv2.imread(image_path)
                        video_out.write(frame)

                    video_out.release()  # Release the video writer
                    break  # Exit the loop
                
            else:
                print("No face detected. Exiting without creating the video.")
                break

video_stream.release()
cv2.destroyAllWindows()
