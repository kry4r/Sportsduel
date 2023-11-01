import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from serverui import Ui_MainWindow
from qfluentwidgets import setThemeColor,ThemeColor
import datetime
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np
# import tqdm
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose
import poseembedding as pe  # 姿态关键点编码模块
import poseclassifier as pc  # 姿态分类器
import resultsmooth as rs  # 分类结果平滑
import counter  # 动作计数器
import visualizer as vs  # 可视化模块
import socket
import cv2
import datetime
import threading

#python -m PyQt5.uic.pyuic duel.ui -o duel.py


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self,path,port):
        super().__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.FramelessWindowHint)   
        #self.close_button.clicked.connect(self.close_window)
        self.start_button.clicked.connect(self.process)
        self.sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = (path,port)
        self.sock.bind(self.ip)     
        
    def mouseMoveEvent(self, e: QMouseEvent):
        if self._tracking:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)
 
    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._startPos = QPoint(e.x(), e.y())
            self._tracking = True
 
    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.LeftButton:
            self._tracking = False
            self._startPos = None
            self._endPos = None
    
    def process_recv(self):
        self.sock.listen(5)
        conn,adress = self.sock.accept()
        while True:
            conn.recv(1024)
            nparr = np.fromstring(data, dtype='uint8', sep='')  # 化为数组
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.enemy.setPixmap(QPixmap.fromImage(img))
            self.enemy.setScaledContents(True)
    
    def process_send(self,flag):
        mkfile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
        if not os.path.exists('./video-output'):
            os.mkdir('./video-output')
        if flag == 1:
            class_name = 'push_down'
        elif flag == 2:
            class_name = 'squat_down'
        elif flag == 3:
            class_name = 'pull_up'
        cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        video_cap = cv2.VideoCapture(0)

        # Get some video parameters to generate output video with classificaiton.
        # video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = 24
        video_width = 640
        video_height = 480

        # Initilize tracker, classifier and counter.
        # Do that before every video as all of them have state.

        # Folder with pose class CSVs. That should be the same folder you using while
        # building classifier to output CSVs.
        pose_samples_folder = 'pose'

        # Initialize tracker.
        pose_tracker = mp_pose.Pose()

        # Initialize embedder.
        pose_embedder = pe.FullBodyPoseEmbedder()

        # Initialize classifier.
        # Check that you are using the same parameters as during bootstrapping.
        pose_classifier = pc.PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            class_name=class_name,
            pose_embedder=pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)

        # # Uncomment to validate target poses used by classifier and find outliers.
        # outliers = pose_classifier.find_pose_sample_outliers()
        # print('Number of pose sample outliers (consider removing them): ', len(outliers))

        # Initialize EMA smoothing.
        pose_classification_filter = rs.EMADictSmoothing(
            window_size=10,
            alpha=0.2)

        # Initialize counter.
        repetition_counter = counter.RepetitionCounter(
            class_name=class_name,
            enter_threshold=5,
            exit_threshold=4)

        # Initialize renderer.
        pose_classification_visualizer = vs.PoseClassificationVisualizer(
            class_name=class_name,
            # plot_x_max=video_n_frames,
            # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
            plot_y_max=10)

        # Run classification on a video.

        # Open output video.

        # frame_idx = 0
        output_frame = None
        # with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
        while video_cap.isOpened():
            # Get next frame of the video.
            success, input_frame = video_cap.read()
            if not success:
                break

            # Run pose tracker.
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            result = pose_tracker.process(image=input_frame)
            pose_landmarks = result.pose_landmarks

            # Draw pose prediction.
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)

            if pose_landmarks is not None:
                # Get landmarks.
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                           for lmk in pose_landmarks.landmark], dtype=np.float32)
                assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks)

                # Smooth classification using EMA.
                pose_classification_filtered = pose_classification_filter(pose_classification)

                # Count repetitions.
                repetitions_count = repetition_counter(pose_classification_filtered)
            else:
                # No pose => no classification on current frame.
                pose_classification = None

                # Still add empty classification to the filter to maintaing correct
                # smoothing for future frames.
                pose_classification_filtered = pose_classification_filter(dict())
                pose_classification_filtered = None

                # Don't update the counter presuming that person is 'frozen'. Just
                # take the latest repetitions count.
                repetitions_count = repetition_counter.n_repeats

            # Draw classification plot and repetition counter.
            output_frame = pose_classification_visualizer(
                frame=output_frame,
                pose_classification=pose_classification,
                pose_classification_filtered=pose_classification_filtered,
                repetitions_count=repetitions_count)

            # 实时输出检测画面
            #cv2.imshow('video', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
            self.self.setPixmap(QPixmap.fromImage(output_frame))
            self.self.setScaledContents(True)
            img_encode = cv2.imencode('.jpg', output_frame)[1]
            data = np.array(img_encode)  # 转化为矩阵
            byte_encode = data.tobytes()  # 编码格式转为字节格式
            data_len = str(len(byte_encode))
            self.sock.send(byte_encode)
            # Save the output frame.
            # 按键盘的q或者esc退出
            #if cv2.waitKey(1) in [ord('q'), 27]:
                #break



        # Close output video.
        video_cap.release()
        cv2.destroyAllWindows()

        # Release MediaPipe resources.
        pose_tracker.close()
        
    def process(self):
        t1 = threading.Thread(target=self.process_recv(),args=(1, 1))
        t2 = threading.Thread(target=self.process_send(1),args=(2, 2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
app = QApplication(sys.argv)
window = MainWindow('192.168.0.103',7788)
window.show()
sys.exit(app.exec_())