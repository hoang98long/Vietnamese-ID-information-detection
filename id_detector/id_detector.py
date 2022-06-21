import tensorflow as tf
import cv2
import numpy as np
import os

IMAGE_SIZE = 380
os_dir = os.path.dirname(__file__)
with open(os.path.join(os_dir, 'EfficientNet_380_200_model.json'), 'r') as reader:
    model_json = reader.read()
model = tf.keras.models.model_from_json(model_json)
model.load_weights(os.path.join(os_dir, 'EffNetB2-380-200-0927-RSM-first-60-0.89661.h5'))


def resize_image(image, width, height, border_mode=cv2.BORDER_REPLICATE, centre=False):
    print(image.shape)
    assert type(width) == int
    assert type(height) == int
    h, w = image.shape[:2]
    pad_bot, pad_right = 0, 0
    if w / h > width / height:
        new_w = width
        new_h = int(h * (new_w / w))
        pad_bot = height - new_h
    else:
        new_h = height
        new_w = int(w * (new_h / h))
        pad_right = width - new_w
    image = cv2.resize(image, (new_w, new_h))
    if centre:
        res = cv2.copyMakeBorder(image, int(pad_bot / 2), int(pad_bot / 2), int(pad_right / 2), int(pad_right / 2),
                                 border_mode)
    else:
        res = cv2.copyMakeBorder(image, 0, pad_bot, 0, pad_right, border_mode)
    print(res.shape)
    return res, ((new_w + pad_right) / new_w, (new_h + pad_bot) / new_h)


def predict(image):
    shift = 0.5
    if type(image) == str:
        assert os.path.exists(image)
        image = cv2.imread(image)
    else:
        assert type(type(image) == np.ndarray)
        pass
    h, w = image.shape[:2]
    resized_image, scale_wh = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
    lb = model.predict(resized_image[np.newaxis, ...])[0][:, np.newaxis]
    lb = np.hstack((lb[:4], lb[4:]))
    lb += shift
    lb[:, 0] = lb[:, 0] * scale_wh[0] * w
    lb[:, 1] = lb[:, 1] * scale_wh[1] * h
    x_min = np.min(lb[:, 0])
    x_max = np.max(lb[:, 0])
    y_min = np.min(lb[:, 1])
    y_max = np.max(lb[:, 1])
    t = np.asarray([[0, 0], [x_max - x_min, 0], [x_max - x_min, y_max - y_min], [0, y_max - y_min]], dtype=np.float32)
    transformed = cv2.getPerspectiveTransform(lb, t)
    dst = cv2.warpPerspective(image, transformed, (x_max - x_min, y_max - y_min))
    return dst


def cropping(img):
    shift = 0.5
    h, w = img.shape[:2]
    img_test, scale_wh = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
    print(img_test[np.newaxis, ...].shape)
    lb = model.predict(img_test[np.newaxis, ...])[0][:, np.newaxis]
    lb = np.hstack((lb[:4], lb[4:]))
    lb += shift
    lb[:, 0] = lb[:, 0] * scale_wh[0] * w
    lb[:, 1] = lb[:, 1] * scale_wh[1] * h
    lb = np.float32(lb)
    perspective_transform_img = np.float32([[0, 0], [650, 0], [650, 400], [0, 400]])
    transformed_img = cv2.getPerspectiveTransform(lb, perspective_transform_img)
    cropped_img = cv2.warpPerspective(img, transformed_img, (650, 400))
    return cropped_img


img = cv2.imread("../test_image/a.jpg")
# cv2.imwrite("test_image/a_crop.jpg",cropping(img))
cv2.imshow("image", cropping(img))
cv2.waitKey(0)
cv2.destroyAllWindows()