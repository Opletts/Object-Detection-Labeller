import cv2

from params import img_path, vid_path

cap = cv2.VideoCapture(vid_path)

f_id = 0
every_n_frames = 1
k = 1

while True:
    ret, img = cap.read()
    img_h, img_w, _ = img.shape
    f_id += 1

    if f_id % every_n_frames == 0:
        cv2.imwrite(img_path + str(k) + '.png', img)

        cv2.imshow("img", cv2.resize(img, (img_w // 2, img_h // 2)))
        cv2.waitKey(1)

        print('image' + str(k))
        k += 1
