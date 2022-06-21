from id_detector import id_detector
from text_localization import text_localization
from text_reading import text_reading
import cv2
import os
import codecs
os_dir = os.path.dirname(__file__)


def process_img(image_dir):
    img = cv2.imread(image_dir)
    img_cropped = id_detector.cropping(img)
    list_localized_img = text_localization.localizing(img_cropped)
    result = text_reading.full_image_reading(list_localized_img)
    cv2.imwrite(os.path.join(os_dir, "result_image/"+image_dir.rsplit('/')[-1]), img_cropped)
    f = codecs.open(os.path.join(os_dir, "result_image/"+image_dir.rsplit('/')[-1][0:-4]+".txt"), "w", "utf-8")
    key = -1
    for text in result:
        key = key + 1
        if key == 0:
            f.writelines(text + '\n')
        else:
            line = ''
            for word in text:
                line = line + word + ' '
            f.writelines(line+'\n')
    f.close()
    return result

print(process_img("D:/AIWork/MTA_OCR_project/MTA_OCR_2020/test_image/a.jpg"))