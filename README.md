# Object-Detection-Labeller
A labelling tool which can be used to label objects with the help of a trained model.

## Installation
Create a virtual environment and install the requirements.

```bash
virtualenv -p /usr/bin/python3.7 venv

source venv/bin/activate

pip3 install -r requirements.txt
```

## Requirements
*   Numpy
*   OpenCV
*   Torch
*   Torchvision
*   Six
*   Pillow

## Usage
[Link](https://entuedu-my.sharepoint.com/:u:/g/personal/n1904786f_e_ntu_edu_sg/ESTo9OCxJIlCrWibMQmCrtgBMFDZS0MboObCg_fGPwSzgg?e=nrSZkc) to a faster_rcnn_resnet50_fpn trained on the Bosch Dataset.

Modify the `img_path`, `save_lbl` and `model_path` in `params.py`.

`enable_model` flag lets you decide if you want a model to make bounding box predictions which you can choose to keep or discard.

Only predictions above the set `confidence` will be shown.

Change `resume` to the name of the image where you left off.

Run using :
```
python3 label_img.py
```

*   The model shows every predicted bounding box one by one.
*   Press `1-8` based on the `class.names` or `d` to discard the prediction.
*   Once all predictions are shown, it switches to manual labelling where the user can add their own bounding boxes and classes.
*   Manual mode is indicated by a red `Manual` displayed on top of the screen.
*   Click at the top left of the region of interest, drag to the bottom right and release. If you're not satisfied with the bounding box, you can draw again, discarding the previous box.
*   After you're satisfied with the box, press `1-8` for the class, which confirms it.
*   Repeat for multiple objects.
*   Press `s` after you've drawn your bounding boxes to save and go to the next image.
*   If the image directly goes manual mode, the model failed to make any predictions.
*   Press `q` to quit.


## Save Format
*   The co-ordinates are normalized and saved, so the values should be between `0` and `1`.
*   `Class_ID    X1  Y1  X2  Y2`
