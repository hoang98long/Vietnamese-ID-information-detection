import os
import tensorflow as tf
import warnings
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os_dir = os.path.dirname(__file__)
detect_fn = tf.saved_model.load(os.path.join(os_dir, 'saved_model'))
classifier = joblib.load(os.path.join(os_dir, 'random_forest.joblib'))


def text_sort(list_box, list_box_label, img_cropped):
    sorted_array = []
    for i in range(10):
        sorted_array.append([])
    for i in range(len(list_box_label)):
        sorted_array[list_box_label[i]].append(list_box[i])
    for i in range(10):
        if i == 1:
            sorted_array[i].sort(key=lambda x: x[0])
            diff = sorted_array[i][-1][0] - sorted_array[i][0][0]
            if diff < 20:
                sorted_array[i].sort(key=lambda x: x[1])
            else:
                upper_name = []
                lower_name = []
                for j in range(len(sorted_array[i])):
                    if sorted_array[i][j][0] < sorted_array[i][0][0] + int(diff/2):
                        upper_name.append(sorted_array[i][j])
                    else:
                        lower_name.append(sorted_array[i][j])
                upper_name.sort(key=lambda x: x[1])
                lower_name.sort(key=lambda x: x[1])
                sorted_array[i] = [*upper_name, *lower_name]
        else:
            sorted_array[i].sort(key=lambda x: x[1])
    for i in range(10):
        if sorted_array[i]:
            for j in range(len(sorted_array[i])):
                sorted_array[i][j] = \
                    img_cropped[sorted_array[i][j][0]:sorted_array[i][j][2], sorted_array[i][j][1]:sorted_array[i][j][3]]
    return sorted_array


def localizing(img_crop):
    height, width = img_crop.shape[:2]
    image_np = np.array(img_crop)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    scale_x = StandardScaler()
    list_box = []
    for i in range(len(detections['detection_scores'])):
        if detections['detection_scores'][i] > 0.3:
            list_box.append(
                [int(detections['detection_boxes'][i][0] * height), int(detections['detection_boxes'][i][1] * width)
                    , int(detections['detection_boxes'][i][2] * height),
                 int(detections['detection_boxes'][i][3] * width)])
    scaled_box = scale_x.fit_transform(list_box)
    list_box_label = classifier.predict(scaled_box)
    text_array = text_sort(list_box, list_box_label, img_crop)
    return text_array

# import cv2
# img = cv2.imread("c.jpg")
#
#
# localizing(img)