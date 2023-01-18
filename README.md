# informative-drawings-onnxrun-cpp-py
使用ONNXRuntime部署Informative-Drawings生成素描画，包含C++和Python两个版本的程序。
起初，我想使用opencv做部署的，但是opencv的dnn模块读取onnx文件出错， 无赖只能使用onnxruntime做部署了。


程序的原始paper是cvpr2022的一篇文章《Learning to generate line drawings that convey geometry and semantics》
，链接在 https://carolineec.github.io/informative_drawings/
