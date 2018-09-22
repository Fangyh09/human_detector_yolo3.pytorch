# human_detector_yolo3.pytorch 
## Run
```python
from yolo.yolo_wrapper import YoloEstimator

model = YoloEstimator()
print(model.predict(["yolo/dog-cycle-car.png"]))
```

## Format
If no human bboxes detected, return [].


Forked from https://github.com/ayooshkathuria/pytorch-yolo-v3

