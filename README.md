# AIGIDetectionZOO

## Introduction 🕵
AIGIDetectionZOO provides APIs of extensive AIGI detectors in research literature, ranging from naive CNN to state-of-the-art. Contributions are always welcome!
<br>


## How to use AIGIDetectionZOO
### Installation
`pip install git+https://github.com/kylin0421/AIGIDetectionZOO`<br>


### Load Detectors

Import and load detector with

```
import aigidetection_zoo as zoo

detector = zoo.load_detector("cnnspot")
```

Currently, only [CNNSpot](https://github.com/PeterWang512/CNNDetection), [UnivFD](https://github.com/WisconsinAIVision/UniversalFakeDetect) and [PatchCraft](https://github.com/Ekko-zn/AIGCDetectBenchmark) are supported. You can check available detectors by

`print zoo.available_models()`<br>


### Detect a single image
```
from PIL import Image


img = Image.open("fake.png")
print(detector.detect_image(img))
```

This returns a single value between 0 and 1 where closer to 1 indicates likely being AI-generated and 0 indicates likely being real.
<br>


### Detect all images in a dataloader

coming soon......
<br>

### Train detectors on your own dataset

coming soon......
<br><br>
## Acknowledgments

This repo is developed based on [CNNSpot](https://github.com/PeterWang512/CNNDetection), [UnivFD](https://github.com/WisconsinAIVision/UniversalFakeDetect), [PatchCraft](https://github.com/Ekko-zn/AIGCDetectBenchmark). Thanks for their sharing codes and models. 


<br><br><br>
**🚧 This repository is still under construction... Stay tuned!**

If you are interested in collaborating, feel free to contact me at su_linxiang@126.com.
