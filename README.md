

Vehicle Cut-in Detection




Overview
Vehicle Cut-in Detection is a project that aims to detect and visualize vehicles cutting in front of a target vehicle using computer vision techniques. This can be useful for advanced driver assistance systems (ADAS) and autonomous driving applications.
Table of Contents
* Features
* Installation
* Usage
* Dataset
* Contributing
* License
* Acknowledgements
Features
* Real-time vehicle detection using Haar Cascades and YOLO.
* Cut-in event detection based on vehicle trajectories.
* Visualization of cut-in events on video streams.
* Modular and easy-to-extend codebase.
Installation
Prerequisites
* Python 3.7+
* OpenCV
* Numpy
* Matplotlib
* MoviePy
Install Dependencies
bash
Copy code
pip install -r requirements.txt
Usage
Running the Detection
To run the vehicle cut-in detection on a sample video, use the following command:
bash
Copy code
python detect_cutin.py --input path/to/input/video.mp4 --output path/to/output/video.mp4
Arguments
* --input: Path to the input video file.
* --output: Path to save the output video with detected cut-ins.
Example
bash
Copy code
python detect_cutin.py --input sample_videos/highway.mp4 --output output_videos/highway_output.mp4
Dataset (Dataset can be downloaded from https://idd.insaan.iiit.ac.in/dataset/download/)
Make sure to format the dataset properly and update the paths in the configuration files.
Contributing
Contributions are welcome! Please read the CONTRIBUTING.md for more information on how to contribute.
Steps to Contribute
1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -am 'Add some feature').
4. Push to the branch (git push origin feature/your-feature).
5. Create a new Pull Request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgements
* OpenCV
* YOLO (The weights can be downloaded from https://pjreddie.com/media/files/yolov3.weights)
* MoviePy
* Special thanks to the open-source community for providing valuable resources and tools.
