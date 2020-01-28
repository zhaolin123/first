import numpy as np
import time
import tensorflow as tf
import cv2

from utils import Read_Img_2_Tensor, Load_DeepFashion2_Yolov3, Draw_Bounding_Box

def Detect_Clothes(img, model_yolov3, eager_execution=True):
    """Detect clothes in an image using Yolo-v3 model trained on DeepFashion2 dataset"""
    img = tf.image.resize(img, (416, 416))

    t1 = time.time()
    if eager_execution==True:
        boxes, scores, classes, nums = model_yolov3(img)
        # change eager tensor to numpy array
        boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()
    else:
        boxes, scores, classes, nums = model_yolov3.predict(img)
    t2 = time.time()
    print('Yolo-v3 feed forward: {:.2f} sec'.format(t2 - t1))

    class_names = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
                  'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
                  'long_sleeve_dress', 'vest_dress', 'sling_dress']

    # Parse tensor
    list_obj = []
    for i in range(nums[0]):
        obj = {'label':class_names[int(classes[0][i])], 'confidence':scores[0][i]}
        obj['x1'] = boxes[0][i][0]
        obj['y1'] = boxes[0][i][1]
        obj['x2'] = boxes[0][i][2]
        obj['y2'] = boxes[0][i][3]
        list_obj.append(obj)

    return list_obj

def Detect_Clothes_and_Crop(img_tensor, model, threshold=0.5):
    list_obj = Detect_Clothes(img_tensor, model)

    img = np.squeeze(img_tensor.numpy())
    img_width = img.shape[1]
    img_height = img.shape[0]
    print(img.dtype)
    # crop out one cloth
    new_W = 400
    new_H = 500
    for obj in list_obj:
        if obj['label'] == 'short_sleeve_top' and obj['confidence']>threshold:
            img_crop = img[int(obj['y1']*img_height):int(obj['y2']*img_height), int(obj['x1']*img_width):int(obj['x2']*img_width), :]
            old_W = img_crop.shape[1]
            old_H = img_crop.shape[0]

            new_img_crop = np.zeros((new_H, new_W, img_crop.shape[2]), dtype=np.float32)

            if old_H/new_H > old_W/new_W:
                max_H = new_H
                max_W = int(new_W * new_H / old_H)
            else:
                max_W = new_W
                max_H = int(new_H * new_W / old_W)
            for i in range(max_H):
                for j in range(max_W):
                    new_img_crop[i][j] = img_crop[int(i / max_H * old_H)][int(j / max_W * old_W)]
    return new_img_crop

if __name__ == '__main__':
    img = Read_Img_2_Tensor('./images/test6.jpg')
    model = Load_DeepFashion2_Yolov3()
    list_obj = Detect_Clothes(img, model)
    img_with_boxes = Draw_Bounding_Box(img, list_obj)

#     cv2.imshow("Clothes detection", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./images/test6_clothes_detected.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)*255)
