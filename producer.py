import pika
import cv2
import json
import threading
import time
import base64

# Connection to RabbitMQ Server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue='video')

def video_feed():
    FRAMES_PER_SECOND = 0.5
    # capture video from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # read frames
        ret, frame = cap.read()

        # convert frame to JPG
        ret, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # publish 
        channel.basic_publish(exchange='',
                              routing_key='video',
                              body=jpg_as_text)
        
        time.sleep(1/FRAMES_PER_SECOND) # Configure this according to your potential frame rate

# Run the video feed function on a separate thread
video_thread = threading.Thread(target=video_feed)
video_thread.start()