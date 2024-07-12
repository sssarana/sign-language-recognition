# ASL alphabet recognition model
Using this dataset from Kaggle: https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download
we can create a model to recognise ASL letters in gestures. To do so, follow the steps below.

## Clone the repository and install requirements
```sh 
git clone 
pip install -r requirements.txt
```
## Adjust paths
Go to scripts/ and adjust paths in each file. Your directory sctructure should be the same as mine, but path to the project directory itself might need to be adjusted. I use \\ in my paths because I am using Windows PC, backslashes are doubled so that it doesn't indicate that the next character is the escape character.
## Prepare the data
Create data/ folder and create 3 folders inside it: raw, processed, models
Dowload the dataset and unzip it in /data/raw folder so that it contains asl_alphabet_test and asl_alphabet_train.
Navigate to your project directory and run the script:
```sh 
python3 scripts/prep_data.py
```
## Train the model
You will need to allow the model some time to train. This will create .h5 file that wa trained using your processed data. Run the script and wait. I had to wait for approximately 22 hours. This time can be reduced by optimising the script to make it use GPU, which will be done in future commits.
```sh 
python3 scripts/train.py
```
## Convert the model
To convert the .h5 weights to tflite model we need to save it as a 'saved model' first, after defining model architecture. Then it can be converted to tflite.
Run:
```sh 
python3 scripts/to_tflite.py
```
## Run 
The inf_laptop.py uses OpenCV to read/write the image from camera 0, if you want to use a camera with differen index, modify the camera_config.yaml.
```sh 
python3 scripts/inf_laptop.py
```
## Running the script on RB5 platform
Running the script on Qualcomm's RB5 platform requires inference modifications. If you had it running on your laptop/PC, then you can just move necessary files to the platform and don't need to train the model on it. This is better approach because we don't need to waste space on RB5 and don't need to leave it working and heating for a long time. 

### Move files
```sh 
adb push sign-lang-recognition/config/camera_config.yaml /data/sign-lang-recognition
adb push sign-lang-recognition/models/sign_lang.tflite /data/sign-lang-recognition
adb push sign-lang-recognition/requirements.txt /data/sign-lang-recognition
adb push sign-lang-recognition/data/processed/label_names.txt /data/sign-lang-recognition
```
Alternatively, create a folder with necessary file on your PC and just push it to RB5
```sh 
adb push sign-lang-recognition-rb5 /data
```
### Modify the inference script and move it to RB5
```sh 
adb push sign-lang-recognition/scripts/inf.py /data/sign-lang-recognition
```
### Connect to the RB5 using adb
```sh 
adb shell
su
oelinux123
```
### Install requirements
```sh 
apt-get install python3-pip libopencv-dev
pip install -r requirements.txt
```
### Run 
inf.py prints results from test images, modify the script to use it for what you want to.
```sh 
cd /data/sign-lang-recognition
python3 inf.py
```

### Run using TCP
If you don't want to write a new inference script, you can use TCP to get output from RB5 camera.
1. Start TCP server on RB5
We can use sample apps repository to start stream. Replace rb5_address available_port with your actual address, for example:
```sh 
./tcp_server 0 112.168.132.120 34808
```
If it doesn't start properly, use netstat to find a free port and try use this instead of the one you used before. 
```sh 
cd /data
git clone https://github.com/quic/sample-apps-for-robotics-platforms.git
cd RB5/linux_kernel_5_x/GStreamer-apps/cpp/gst_streaming/
make
./tcp_server 0 rb5_address available_port
```
Make sure it is linux_kernel_5_x and not linux_kernel_4_x.
2. Use OpenCV to connect to it
Open inf_tcp.py and change this:
```sh 
cap = cv2.VideoCapture("tcp://112.168.132.120:34808")
```
to your actual address and port. Run the script:
```sh 
python inf_tcp.py
```