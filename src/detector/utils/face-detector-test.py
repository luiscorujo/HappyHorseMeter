import numpy as np
import tensorflow as tf
import cv2 as cv
import PIL

# Read the graph.
with tf.gfile.FastGFile('saved_faster_23k/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    img_width = 100
    img_height = 100
    # scale factor for preprocessing
    picSize = 200
    rotation = True

    # Read and preprocess an image.
    original_img = cv.imread('test_image1.jpeg')
    height, width, channels = original_img.shape

    # Resize image
    ratio = picSize / height
    resized_image = cv.resize(original_img, None, fx=ratio, fy=ratio)
    inp = resized_image[:, :, [2, 1, 0]]  # BGR2RGB

    # gray image
    # gray = cv.cvtColor(inp, cv.COLOR_RGB2GRAY)
    # croped_image = gray

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    croped_image = None
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * width
            y = bbox[0] * height
            right = bbox[3] * width
            bottom = bbox[2] * height
            croped_image = original_img[int(y):int(bottom), int(x):int(right)]
            cv.rectangle(original_img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

cv.imshow('TensorFlow MobileNet-SSD', original_img)
cv.imshow("cropped!", croped_image)
cv.imwrite('croped.jpg', croped_image)
cv.imwrite('image_with_rectangel.jpg', original_img)

cv.waitKey()


##
"""
    cap = cv.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        rows = frame.shape[0]
        cols = frame.shape[1]
        #inp = cv.resize(frame, (300, 300))
        #inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': frame.reshape(1, frame.shape[0], frame.shape[1], 3)})

        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
        cv.imshow('TensorFlow MobileNet-SSD', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

#"""

##

"""
    # Read and preprocess an image.
    img = cv.imread('test_image1.jpeg')
    rows = img.shape[0]
    cols = img.shape[1]

    # resize image
    inp = cv.resize(img, (300, 300))

    croped_image = inp

    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            croped_image = img[int(y):int(bottom), int(x):int(right)]
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            cv.imshow("cropped!", croped_image)

cv.imshow('TensorFlow MobileNet-SSD', img)
cv.imshow("cropped!", croped_image)
cv.waitKey()

"""
