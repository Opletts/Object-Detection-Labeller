import os
import re
import cv2

import torch
import torchvision
from torchvision.transforms import functional as F

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from params import *


def get_coords(event, x, y, flags, param):
    global mouseX1, mouseY1, mouseX2, mouseY2, flag
    if event == cv2.EVENT_LBUTTONDOWN:
        flag = 0
        mouseX1, mouseY1 = x, y

    if event == cv2.EVENT_LBUTTONUP:
        flag = 1
        mouseX2, mouseY2 = x, y


if enable_model:
    model = torchvision.models.detection.__dict__["fasterrcnn_resnet50_fpn"](pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 8)

    device = torch.device("cuda:0")
    model.to(device)

    model.eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])


mouseX1, mouseY1 = 0, 0
mouseX2, mouseY2 = 0, 0

img_list = os.listdir(img_path)
img_list.sort(key=lambda f: int(re.sub('\D', '', f)))

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_coords)

for name in img_list:
    save_str = ""
    num_boxes = 0
    img = cv2.imread(os.path.join(img_path, name))
    img_h, img_w, _ = img.shape

    img = cv2.resize(img, (img_w // 2, img_h // 2))
    img_h, img_w, _ = img.shape
    disp = img.copy()
    cpy = img.copy()

    if enable_model:
        img = F.to_tensor(img).unsqueeze(0)
        img = img.to(device)

        pred = model(img)
        bboxes = pred[0]['boxes'].cpu().detach().numpy()
        scores = pred[0]['scores'].cpu().detach().numpy()

        for bbox, score in zip(bboxes, scores):
            save = False
            cv2.putText(disp, "Model Prediction", (img_w // 3, 50), 0, 1, (0, 0, 255), 1)

            if score > confidence:
                x1, y1, x2, y2 = bbox

                while True:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), 255, 1)
                    cv2.imshow("Image", disp)
                    key = cv2.waitKey(1)

                    if key >= ord('0') and key <= ord('9'):
                        num_boxes += 1
                        val = key - 48
                        save = True
                        break

                    if key == ord('d'):
                        save = False
                        break

                    if key == ord('q'):
                        print("Stopped at ", name)
                        exit(0)

                if save:
                    cv2.rectangle(cpy, (x1, y1), (x2, y2), 255, 1)
                    disp = cpy.copy()
                    save_str += str(val - 1) + " " + str(x1 / img_w) + " " + str(y1 / img_h) + " " + str(x2 / img_w) + " " + str(y2 / img_h) + "\n"

    print("Add your own Boxes")
    disp = cpy.copy()
    cv2.putText(disp, "Manual", (img_w // 2, 50), 0, 1, (0, 0, 255), 1)
    cpy = disp.copy()

    while True:
        cv2.imshow("Image", disp)
        key = cv2.waitKey(1)

        if mouseX1 != 0 and mouseY1 != 0 and mouseX2 != 0 and mouseY2 != 0 and flag:
            disp = cpy.copy()
            cv2.rectangle(disp, (mouseX1, mouseY1), (mouseX2, mouseY2), (0, 0, 255), 1)

        if key >= ord('0') and key <= ord('9'):
            val = key - 48

            if mouseX1 != 0 and mouseY1 != 0 and mouseX2 != 0 and mouseY2 != 0:
                num_boxes += 1
                save_str += str(val - 1) + " " + str(mouseX1 / img_w) + " " + str(mouseY1 / img_h) + " " + str(mouseX2 / img_w) + " " + str(mouseY2 / img_h) + "\n"
                cv2.rectangle(disp, (mouseX1, mouseY1), (mouseX2, mouseY2), (0, 0, 255), 1)
                cv2.rectangle(cpy, (mouseX1, mouseY1), (mouseX2, mouseY2), (0, 0, 255), 1)
                mouseX1, mouseY1, mouseX1, mouseY2 = 0, 0, 0, 0

        if key == ord('s'):
            break

        if key == ord('q'):
            print("Stopped at ", name)
            exit(0)

    img_id = name[5:-4]

    with open(os.path.join(save_lbl, 'label' + img_id + '.txt'), 'w') as f:
        f.writelines(save_str)

    print("Boxes : ", num_boxes)
