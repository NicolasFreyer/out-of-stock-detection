import pika
import cv2
import json
import numpy as np
import base64
from ultralytics import YOLO
from scipy.spatial import distance
import supervision as sv
import threading
from flask import Flask, jsonify, Response

app = Flask(__name__)

# Connection to RabbitMQ Server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue='video')


def get_distance(box1, box2):
    center1 = (box1[2] / 2, box1[3] / 2)
    center2 = (box2[2] / 2, box2[3] / 2)
    return distance.euclidean(center1, center2)

global_product_absences = []
global_frame = None
cap = cv2.VideoCapture(0)
model = YOLO("/Users/nicolasfreyer/out_of_stock_detection_project/weights_1.pt")
box_annotator = sv.BoxAnnotator(thickness = 2, text_thickness= 2, text_scale=1)

def frame_processor(ch, method, properties, body):
    # Load the jpg from message
    jpg_original = base64.b64decode(body)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)
    
    # start of your YOLO code
    global global_product_absences
    global_product_absences.clear()
    global global_frame #note the inclusion of the global keyword here
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]
    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels
    )
    
    global_frame = frame

    absence_labels = []
    product_labels = []
    for box, _, score, class_id, _ in detections:
        if class_id == 2 and score > 0.5:
            absence_labels.append((box, score))
        else:
            if score > 0.4:
                product_labels.append((box, class_id))

    for (absence_box, _) in absence_labels:
        if product_labels:
            best_match = min(product_labels, key=lambda product: get_distance(absence_box, product[0]))
            global_product_absences.append(model.names[best_match[1]])

    absences_set = set(global_product_absences)
    global_product_absences = list(absences_set)


def generate():
    while True:
        if global_frame is not None:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', global_frame)
            frame = buffer.tobytes()

            # Concatenate frame into video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/list", methods=["GET"])
def get_product_absences():
    result = list(global_product_absences)
    return jsonify(result)

channel.basic_consume(queue='video', on_message_callback=frame_processor, auto_ack=True)

if __name__ == "__main__":
    # Run the consumer function on a separate thread as flask will take the main thread.
    consumer_thread = threading.Thread(target=channel.start_consuming)
    consumer_thread.start()
    # Run Flask application
    app.run(port=5000)