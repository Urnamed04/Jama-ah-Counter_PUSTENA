import cv2
import numpy as np

# ============================================
# SETUP
# ============================================

# GANTI dengan IP yang muncul di aplikasi DroidCam di HP kamu
droidcam_ip = "192.168.18.89"  # ← ganti sesuai dengan IP di aplikasi DroidCam kamu
video_url = f"http://{droidcam_ip}:4747/video"

# Load YOLOv3 pretrained untuk deteksi wajah (WIDER face dataset)
net = cv2.dnn.readNet(
    r"C:\Users\Ridersystem\Downloads\Compressed\gfdfytcgfvh\New folder\yolov3-wider_16000.weights",
    r"C:\Users\Ridersystem\Downloads\Compressed\gfdfytcgfvh\New folder\yolov3-face.cfg"
)

# Ambil nama layer output dari YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Buka video stream dari DroidCam IP
cap = cv2.VideoCapture(video_url)

# Periksa apakah kamera terbuka dengan benar
if not cap.isOpened():
    print("❌ Tidak bisa membuka stream dari DroidCam. Periksa IP dan koneksi WiFi.")
    exit()

print("✅ Streaming dimulai dari DroidCam...")

# ============================================
# LOOP UTAMA
# ============================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ Gagal mengambil frame dari stream.")
        break

    height, width, channels = frame.shape

    # Buat blob untuk input ke YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Parsing hasil deteksi
    boxes = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            confidence = scores[0]  # Hanya satu kelas (wajah)
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Non-maximum suppression untuk hapus bounding box tumpang tindih
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Gambar bounding box ke frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Wajah {int(confidences[i]*100)}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tampilkan hasil deteksi
    cv2.imshow('Deteksi Wajah dari DroidCam', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# AKHIR
# ============================================

cap.release()
cv2.destroyAllWindows()
