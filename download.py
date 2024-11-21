
from roboflow import Roboflow
rf = Roboflow(api_key="rSg3JUtMCujZpQaOoU00")
project = rf.workspace("sdgp-project").project("tea-leaf-diseases-6el9p-qgcji")
version = project.version(2)
dataset = version.download("yolov8")
