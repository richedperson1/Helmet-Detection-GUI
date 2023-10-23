import numpy as np
import torch
import cv2
import yaml

font = cv2.FONT_HERSHEY_COMPLEX
colors = np.random.uniform(0, 255, size=(100, 3))


class yolo_helmet_v4:
    def __init__(self) -> None:
        """
        Setting up the model for helmet detection
        """

        yolov4_model_weight_path = "helmetModel\\yolov3_tiny_custom_6000.weights"
        yolov4_model_config_path = "helmetModel\\testing_yolov3_tiny_custom.cfg"
        self.net = cv2.dnn.readNet(
            yolov4_model_weight_path, yolov4_model_config_path)

        self.classes = ["Helmet", "No-Helmet"]

    def detect_helmet_V4(self, img):
        """
        Detection of helmet using YOLO V3 Model
        """
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(
            img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

        self.net.setInput(blob)
        output_layers_names = self.net.getUnconnectedOutLayersNames()

        layerOutputs = self.net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []

        # print(layerOutputs)
        labelTxt = []
        for output in layerOutputs:
            for detection in output:
                #
                scores = detection[5:]
                class_id = np.argmax(scores)
                # print(class_id)
                confidence = scores[class_id]
                # print(scores)
                confidence_thr_v4 = 0.4
                if confidence > confidence_thr_v4:
                    # print(class_id)
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    xMax = x+w
                    yMax = y+h

                    boxes.append([x, y, w, h])
                    labelTxt.append(
                        [f"{class_id} {center_x/width} {center_y/height} {w/width} {h/height}\n"])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        flag = False
        if len(indexes) > 0:
            for i in indexes.flatten():
                flag = True
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence,
                            (x, y+10), font, 0.5, (255, 0, 0), 1)
                # cv2.imshow('Image3', img)

        return labelTxt


if __name__ == "__main__":
    obj = yolo_helmet_v4()
