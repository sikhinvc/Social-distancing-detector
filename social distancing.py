import cv2
import numpy as np
import math
weightpath = "/home/shikhin/PycharmProjects/spark internship/object detection/yolov3.weights"
yolo3_path = "/home/shikhin/PycharmProjects/spark internship/object detection/yolov3.cfg"
net = cv2.dnn.readNet(weightpath, yolo3_path)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
cap = cv2.VideoCapture("pedestrians.mp4")
while 1:
    ret, img = cap.read()
    img = cv2.resize(img, (800, 800))
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    coordinatesx = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == "person":
                coordinatesx.append((int(x+w/2),int(y+h/2)))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    violation = 0
    for i,j in coordinatesx:
        for u, v in coordinatesx:
            distance = math.sqrt(((i - u) ** 2) + ((j - v) ** 2))
            if distance>0.0 and distance<50.0:
                q = coordinatesx.index((i, j))
                w = coordinatesx.index((u, v))
                cv2.line(img, (i, j), (u, v), color=(0, 0, 255), thickness= 2 )
                violation = violation+ 1
    text1 = "Total number of violations: " + str(int(violation / 2))
    cv2.putText(img, text1, (50, 50), font, 2, color, 2)
    cv2.imshow("Output", img)
    print(coordinatesx)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
