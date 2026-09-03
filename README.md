# FishCount

FishCount uses the YOLO model to count fish in images or videos.

## 1. Install Python
Ensure Python is installed on your computer. You can work in a local environment such as IDLE, Visual Studio, PyCharm, or use Google Colab or Jupyter Notebook. This guide focuses on a local setup, but instructions for Colab are also provided.

To install Python:
- **Windows**: Download and install the latest version from [Python's official website](https://www.python.org/downloads/).
- **Linux**: Open a terminal and run:
  ```bash
  sudo apt-get update && sudo apt-get upgrade
  sudo apt-get install python3 -y
  ```

## 2. Set Up Your Working Directory 
Create a new folder for your project. Example:

- **Windows**:
  ```bash
  cd Desktop\fishdetect
  ```
- **Linux**:
  ```bash
  cd Desktop/fishdetect
  ```
or else you can directly clone this repo to your computer 
```bash
git clone https://github.com/dpkpandey/fishdetect
cd fishdetect
```
Then folow follow step 3 and 4 and do 
```bash
pip install -r requirements.txt
```
so you can skip installing other dependancies as explained in step 4

## 3. Create a Virtual Environment
To manage dependencies, create and activate a virtual environment:

- **Windows**:
  ```bash
  python -m venv myenv
  myenv\Scripts\activate
  ```
- **Linux**:
  ```bash
  python3 -m venv myenv
  source myenv/bin/activate
  ```

To exit the virtual environment, type:
```bash
deactivate
```

## 4. Install Dependencies
Before proceeding, upgrade `pip` and install required libraries:
```bash
python3 -m pip install --upgrade pip
```
If you have not clone repo and want to manaully then you should install required dependencies, install all as required.
```bash
pip install ultralytics opencv-python numpy openpyxl torch
```

## 5. Enable GPU Acceleration (Optional)
For better performance with large datasets, configure GPU support:

1. Install the compatible CUDA version for your GPU. Check [Nvidia's website](https://developer.nvidia.com/cuda-downloads).
2. Install PyTorch with CUDA support from [PyTorch's official website](https://pytorch.org/).
3. Run the recommended command in your terminal to install PyTorch with CUDA.
   
If you have downloaded "cuda==2.16" stable version then run this code for both Windows and Linux system.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## 6. Prepare Dataset
1. Sign up or log in to [Roboflow](https://roboflow.com/).
2. Upload and annotate images.
3. Download the dataset and organize files:
   - Place images in the `images` folder.
   - Place annotation `.txt` files in the `labels` folder inside your `fishdetect` directory.

## 7. Create YAML Configuration File
Create `fishdetect.yaml` to define the dataset path and classes:
```yaml
path: C:\Users\YourComputer\Desktop\fishdetect  #This is path of your current directory you can do pwd to see location and can copy that
train: images
val: images
names:
  0: fish
```
 
### For multiple classes (e.g., algae detection):  $\textcolor{red}{You}$ $\textcolor{red}{can}$ $\textcolor{red}{skip}$ $\textcolor{red}{this}$ $\textcolor{red}{one}$  $\textcolor{red}{here,}$ $\textcolor{red}{this}$ $\textcolor{red}{is}$ $\textcolor{red}{just}$ $\textcolor{red}{an}$  $\textcolor{red}{example}$ $\textcolor{red}{of}$ $\textcolor{red}{how}$ $\textcolor{red}{to}$ $\textcolor{red}{use}$  $\textcolor{red}{multiple}$ $\textcolor{red}{classes}$

 
```yaml
path: C:\Users\YourComputer\Desktop\algaedetect  #This is the path of your current directory
train: images
val: images
nc: 15
names: ['Ceratium', 'Chaetoceros', 'Cyclotella', 'Cyanobacteria', 'Euglenoid Eutreptiella', 'Gymnodinium', 'Microcystis', 'New', 'Oocystis', 'Oocytis', 'Oscillatoria', 'Pleurosigma sp.', 'Pseudo-nitzschia', 'Pseudo-nitzschia sp.', 'macro-algae']
```

## 8. Train the Model
Create `main.py` with the following content:
```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.yaml").load("yolo11m.pt")  # Train from scratch
# model = YOLO("yolo11m.pt")  # Load pre-trained model (recommended)

# Train the model
results = model.train(data="fishdetect.yaml", batch=8, epochs=500)
```
- Adjust batch size according to your GPU memory.
- Run the script:
  ```bash
  python main.py
  ```

## 9. Run Detection
After training, see the run/train/weight folder and you will see last.pt and best.pt files, use those file and rename it like here I renamed " lastsmall1000all.pt"  and keep in your working directory name as detectfish.  
We use the model to detect fish in images or videos:
```python
import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("lastsmall1000all.pt") #here I have lastY11m500.pt and bestY11m500.pt files both are very good model those output result footage is done in this model.
results = model.predict("outputfile.mp4", show=True, save=True, conf=0.30, tracker ="botsort.yaml") # you can choose your tracker= "bytetrack.yaml" or other.
cv2.imshow("Detection", result.plot())
if cv2.waitKey(1) & OxFF ==ord('q'):
    break
cap.release()
cv2.destroyAllWindows()
# The result for this code will be saved in run/predict folder
```

## 10. Count
Congratulations! You've successfully set up and trained a YOLO model to count whatever you want, this program simply suggests you can modify this for any detection.
Here you can fine tune sort.py file such that you can easily get desirable results. In classes.txt, we 
define classes name. If you are using multiple classes just include in that and make a change in count_fish.py as well.
If you are going to count other object then go for it. It will work. Just remember you will need to have pytorch model i.e., .pt file. Thats all.
Now download all the files in same folder and start counting. Good luck.


 ## 11. Measure length and Weight
To measure length and weight of the fish we use $\textcolor{red}{dpk-length}$ approximation as explained in the code more documentations will be provided on request on above youtube channel. 
This approximation works fine for Barramundi fish for any size becasue of emprical results on large number of fish. This approximation to pull length from detection is only 
applicable for detection, if you want to work on OBB detection then you can direclty impliment length and width as in detection no need to use " $\textcolor{red}{dpk-length}$ $\textcolor{red}{ transformation"}$. However,
simple detection is faster for real time detection and counting as compare to OBB with small computational power.

## 12. Optimize with Tensorrt for faster 

If you need to count or detect in very fast moving motion, then your yolo.pt file may not be fast enough to work in high FPS in real time. However, using nvidia cuda toolkit and pytorch, now we can install tensorrt.
```bash
pip3 install tensorrt
```

After installation, we can generate optimize file which is balanced and optimized for fast operation. 

At first we need to generate .engine file using above command 

```bash
from ultralytics import YOLO

model= YOLO("last.pt") #whaterver you have used as a model
model.export( format="engine", task="detect", half="True) #this creates last.engine, task means what you want to do with this optimization. And with half =True select FP16 (half precisison) or you can use 'optimize=True' 

tensorrt_model = YOLO("last.engine")
```
You will get optimized file "last.engine" which is almost equivalent to the .pt file in term of performance and it is 2 times faster than yolo.pt model.
Now I have uplodaded Mainfishcount.py file as well which is robust and more powerful interms of fish counting. You can change your last1.engine file according to output of your optimized or still you can use bestYm500.pt model as well. Choice is yours. For this if you use https://youtu.be/pXhthoGPRkg link it will work better because of high FPS. 
## References

```bash
title: Ultralytics YOLO
message: >-
  If you use this software, please cite it using the
  metadata from this file.
type: software
authors:
  - given-names: Glenn
    family-names: Jocher
    affiliation: Ultralytics
    orcid: 'https://orcid.org/0000-0001-5950-6979'
  - family-names: Qiu
    given-names: Jing
    affiliation: Ultralytics
    orcid: 'https://orcid.org/0000-0003-3783-7069'
  - given-names: Ayush
    family-names: Chaurasia
    affiliation: Ultralytics
    orcid: 'https://orcid.org/0000-0002-7603-6750'
repository-code: 'https://github.com/ultralytics/ultralytics'
url: 'https://ultralytics.com'
license: AGPL-3.0
version: 8.0.0
date-released: '2023-01-10'
```


For improvements or contributions, feel free to submit a pull request!

