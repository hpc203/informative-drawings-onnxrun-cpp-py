import argparse
import cv2
import numpy as np
import onnxruntime

class Informative_Drawings():
    def __init__(self, modelpath):
        try:
            cv_net = cv2.dnn.readNet(modelpath)
        except:
            print('opencv read onnx failed!!!')
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        # providers_list = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        providers_list = ['CPUExecutionProvider']
        self.net = onnxruntime.InferenceSession(modelpath, so,providers = providers_list)
        input_shape = self.net.get_inputs()[0].shape
        self.input_height = int(input_shape[2])
        self.input_width = int(input_shape[3])
        self.input_name = self.net.get_inputs()[0].name
        self.output_name = self.net.get_outputs()[0].name

    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(np.transpose(img.astype(np.float32), (2, 0, 1)), axis=0).astype(np.float32)
        outs = self.net.run([self.output_name], {self.input_name: blob})

        result = outs[0].squeeze()
        result *= 255
        result = cv2.resize(result.astype('uint8'), (srcimg.shape[1], srcimg.shape[0]))
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='images/2.jpg', help='image path')
    parser.add_argument("--modelpath", type=str, default='weights/opensketch_style_512x512.onnx', choices=["weights/opensketch_style_512x512.onnx", "weights/anime_style_512x512.onnx", "weights/contour_style_512x512.onnx"], help='onnx filepath')
    args = parser.parse_args()

    mynet = Informative_Drawings(args.modelpath)
    srcimg = cv2.imread(args.imgpath)
    result = mynet.detect(srcimg)

    cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
    cv2.imshow('srcimg', srcimg)
    winName = 'Deep learning in onnxruntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()