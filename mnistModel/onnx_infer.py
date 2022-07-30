import io

import numpy
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image


def pre_pic(picName):
    img = Image.open(picName)
    img.show()
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))  # 变为灰度图
    threshold = 50  # 阈值，将图片二值化操作
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]  # 进行反色处理
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)  # 类型转换
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)  # 把值变为0~1之间的数值
    return img_ready


ort_session = ort.InferenceSession("./model/mnist.onnx", providers=["CPUExecutionProvider"])
outputs = ort_session.get_outputs()

ret = ort_session.run(None,
                      input_feed={
                          ort_session.get_inputs()[0].name: pre_pic("./test/test.png").reshape([1, 1, 28, 28])
                      }
                      )
print(ret[0])
