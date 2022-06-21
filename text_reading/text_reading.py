import tensorflow as tf
import numpy as np
import cv2
import os
from text_reading.model_gender.gender import gender_value

minimal_output_signature = {
    'predictions': 'AttentionOcr_v1/predicted_chars:0',
    'scores': 'AttentionOcr_v1/predicted_scores:0',
    'predicted_length': 'AttentionOcr_v1/predicted_length:0',
    'predicted_text': 'AttentionOcr_v1/predicted_text:0',
    'predicted_conf': 'AttentionOcr_v1/predicted_conf:0',
    'normalized_seq_conf': 'AttentionOcr_v1/normalized_seq_conf:0'
}
os_dir = os.path.dirname(__file__)
tf.compat.v1.reset_default_graph()
graph1 = tf.compat.v1.Graph()
with graph1.as_default():
    sess_number = tf.compat.v1.Session()
    graph_def_number = tf.compat.v1.saved_model.loader.load(
        sess=sess_number,
        tags=[tf.saved_model.SERVING],
        export_dir=os.path.join(os_dir, 'model_number'))

graph2 = tf.compat.v1.Graph()
with graph2.as_default():
    sess_text = tf.compat.v1.Session()
    graph_def_text = tf.compat.v1.saved_model.loader.load(
        sess=sess_text,
        tags=[tf.saved_model.SERVING],
        export_dir=os.path.join(os_dir, 'model_text'))

graph3 = tf.compat.v1.Graph()
with graph3.as_default():
    sess_dob = tf.compat.v1.Session()
    graph_def_dob = tf.compat.v1.saved_model.loader.load(
        sess=sess_dob,
        tags=[tf.saved_model.SERVING],
        export_dir=os.path.join(os_dir, 'model_dob'))

graph4 = tf.compat.v1.Graph()
with graph4.as_default():
    sess_date_expire = tf.compat.v1.Session()
    graph_def_date_expire = tf.compat.v1.saved_model.loader.load(
        sess=sess_date_expire,
        tags=[tf.saved_model.SERVING],
        export_dir=os.path.join(os_dir, 'model_date_of_expire'))

graph5 = tf.compat.v1.Graph()
with graph5.as_default():
    sess_resident = tf.compat.v1.Session()
    graph_def_resident = tf.compat.v1.saved_model.loader.load(
        sess=sess_resident,
        tags=[tf.saved_model.SERVING],
        export_dir=os.path.join(os_dir, 'model_resident'))

graph6 = tf.compat.v1.Graph()
with graph6.as_default():
    sess_poi = tf.compat.v1.Session()
    graph_def_poi = tf.compat.v1.saved_model.loader.load(
        sess=sess_poi,
        tags=[tf.saved_model.SERVING],
        export_dir=os.path.join(os_dir, 'model_poi'))


def read_image(image, new_h, max_w):
    input_image = np.asarray(image)
    im = input_image
    h, w, d = im.shape
    unpad_im = cv2.resize(im, (int(new_h * w / h), new_h), interpolation=cv2.INTER_AREA)
    if unpad_im.shape[1] > max_w:
        pad_im = cv2.resize(im, (max_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        pad_im = cv2.copyMakeBorder(unpad_im, 0, 0, 0, max_w - int(new_h * w / h), cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return pad_im


def ocr(image, new_height, max_width, key):
    result_string = ''
    input_image = read_image(image, new_height, max_width)[np.newaxis, :]
    feed_dict = {
        'original_image:0': input_image
    }
    if key == 0:
        results = sess_number.run(minimal_output_signature, feed_dict=feed_dict)
    elif key == 2:
        results = sess_dob.run(minimal_output_signature, feed_dict=feed_dict)
    elif key == 9:
        results = sess_date_expire.run(minimal_output_signature, feed_dict=feed_dict)
    elif key in {7, 8}:
        results = sess_resident.run(minimal_output_signature, feed_dict=feed_dict)
    elif key in {5, 6}:
        results = sess_resident.run(minimal_output_signature, feed_dict=feed_dict)
    else:
        results = sess_text.run(minimal_output_signature, feed_dict=feed_dict)
    for pr_byte in results['predicted_text'].tolist():
        result_string = result_string + pr_byte.decode('utf-8')
    result_string = result_string.split(b'\xe2\x96\x91'.decode('utf-8'))[0]
    return result_string


def full_image_reading(list_localized_image):
    result = []
    if len(list_localized_image) == 0:
        return
    for i in range(len(list_localized_image)):
        if i == 0:
            result.append(ocr(list_localized_image[i][0], 64, 500, i))
        else:
            card_information = []
            for j in range(len(list_localized_image[i])):
                if i in {2, 5, 6, 7, 8, 9}:
                    card_information.append(ocr(list_localized_image[i][j], 64, 320, i))
                    if i == 2:
                        break
                elif i == 3:
                    card_information.append(gender_value(list_localized_image[i][j]))
                    break
                else:
                    card_information.append(ocr(list_localized_image[i][j], 40, 100, i))
            result.append(card_information)
    return result


