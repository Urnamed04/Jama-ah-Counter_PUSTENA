import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("C:\Users\Ridersystem\Downloads\Compressed\gfdfytcgfvh\yolov3-wider_16000.weights","C:\Users\Ridersystem\Downloads\Compressed\gfdfytcgfvh\yolov3-face.cfg")

# Debug prints
print("Layer names:", net.getLayerNames())
print("Unconnected out layers:", net.getUnconnectedOutLayers())

layer_names = net.getLayerNames()
output_layers = [layer_names[int(i) - 1] for i in net.getUnconnectedOutLayers()]

# Buka kamera (kamera default biasanya index 0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Periksa apakah kamera terbuka dengan benar
if not cap.isOpened():
    print("Kamera tidak bisa dibuka")
    exit()

while True:
    # Ambil frame dari kamera
    ret, frame = cap.read()

    # Periksa apakah frame berhasil diambil
    if not ret:
        print("Gagal mengambil gambar")
        break

    height, width, channels = frame.shape

    # Create a blob and pass it through the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []

    # Process each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            confidence = scores[0]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Tampilkan frame yang dihasilkan
    cv2.imshow('Deteksi Wajah', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
