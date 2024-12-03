# from ultralytics import YOLO

# from IPython.display import display, Image

# from roboflow import Roboflow
# rf = Roboflow(api_key="rSg3JUtMCujZpQaOoU00")
# project = rf.workspace("ntnst").project("lemon-disease-detection")
# version = project.version(2)
# dataset = version.download("yolov8")
                

# from roboflow import Roboflow
# rf = Roboflow(api_key="rSg3JUtMCujZpQaOoU00")
# project = rf.workspace("ruang-saya").project("tanaman-hkx9z")
# version = project.version(15)
# dataset = version.download("yolov8")
                

# from roboflow import Roboflow
# rf = Roboflow(api_key="rSg3JUtMCujZpQaOoU00")
# project = rf.workspace("sdgp-project").project("tea-leaf-diseases-6el9p-qgcji")
# version = project.version(2)

# dataset = version.download("yolov8")



# from roboflow import Roboflow
# rf = Roboflow(api_key="rSg3JUtMCujZpQaOoU00")
# project = rf.workspace("ruang-saya").project("tanaman-hkx9z")
# version = project.version(15)
# dataset = version.download("yolov11")
                
                

# from roboflow import Roboflow
# rf = Roboflow(api_key="rSg3JUtMCujZpQaOoU00")
# project = rf.workspace("tarumt-wdovj").project("citrus-eq7f8")
# version = project.version(1)
# dataset = version.download("multiclass")


import kagglehub

# Download latest version
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

print("Path to dataset files:", path)