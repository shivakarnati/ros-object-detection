""""
To run the script:

-$ colcon build
-$ source ./install/setup.bash
-$ ros2 run object_det object_det_node --ros-args -p image_subscription:="/oakd/rgb/preview/image_raw"

To run the rosbag file:
ros2 bag play -s mcap rec1_0.mcap
"""
# ros packages
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
import cv_bridge

# ml packages
import cv2
import numpy as np
import torch

# supervision
import supervision as sv
from supervision.detection.core import Detections
from supervision.annotators.core import BoundingBoxAnnotator

# detectron
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.config import global_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes

# yolo
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.solutions import object_counter

# mediapipe
from collections import defaultdict
import mediapipe as mp

class ObjectDetNode(Node):
    def __init__(self):
        super().__init__("obj_det_node")
        
        self.get_logger().info("starting object dection node...")
        
        # Declare subscription parameter
        self.declare_parameter("img_subscription", Parameter.Type.STRING)
        self.img_subscription = self.get_parameter("img_subscription").value
        self.bridge = cv_bridge.CvBridge()
        self.last_robot_image = np.zeros((480,640,3),np.uint8)

        # Create the image subscription 
        self.img_subscription_values = self.create_subscription(
            Image,
            self.img_subscription,
            self.store_img,
            10
        )
        self.start_recording("recorded_detectron")
        
        """ 

        # self.vid_cap = cv2.VideoCapture(0)
        # assert self.vid_cap.isOpened()
        
        #w, h, fps = (int(self.vid_cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Define region points
        #region_points = [(400, 20), ( 404,1080), (360,1080), (360,20)]

        # Video writer
        # video_writer = cv2.VideoWriter("object_counting_output.avi",
        #                     cv2.VideoWriter_fourcc(*'mp4v'),
        #                     fps,
        #                     (w, h))
        
        # Init Object Counter
        # counter = object_counter.ObjectCounter()
        # counter.set_args(view_img=True,
        #          reg_pts=region_points,
        #          classes_names=self.model('yolo').names,
        #          draw_tracks=False)
        # count = 0
        # while self.vid_cap.isOpened():
        #     success, im0 = self.vid_cap.read()
        #     if not success:
        #         print("Video frame is empty or video processing has been successfully completed.")
        #         break
        #     tracks = self.model('yolo').track(im0, persist=True, show=False,classes=56)
        #     im0 = counter.start_counting(im0, tracks)
        #     video_writer.write(im0)
        #     count = count+20
        #     self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
                    
        # self.vid_cap.release()
        # video_writer.release()
        # cv2.destroyAllWindows()
        
        # supervision
        tracker = sv.ByteTrack()
        box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        if success:
            results = self.model('yolo')(self.open_cv_image)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            
            labels = [
                f"#{tracker_id} {results.names[class_id]}"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id) 
            ]
            
            annotated_frame = box_annotator.annotate(
                self.open_cv_image.copy(), detections= detections)
            
            labelled_frame = label_annotator.annotate(annotated_frame,detections=detections, labels=labels)
            count = count+20
            self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            cv2.imshow('tracking', labelled_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.save_frame(labelled_frame)
                
            if success:
                results = self.model('yolo').track(self.open_cv_image, persist=True,classes=56, show= False)
                
                self.open_cv_image = counter.start_counting(self.open_cv_image, results)
                
                # boxes = results[0].boxes.xywh.cpu()
                # track_ids = results[0].boxes.id.int().cpu().tolist()
                
                # annotated_frame = results[0].plot()

                # for box, track_id in zip(boxes, track_ids):
                #     x, y, w, h = box
                #     track = track_history[track_id]
                #     track.append((float(x), float(y)))
                #     if len(track) > 30:
                #         track.pop(0)
                
                count = count+20
                self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES,count)
                # cv2.imshow('tracking',self.open_cv_image)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
                self.save_frame(self.open_cv_image)
        
            self.vid_cap.release()
            cv2.destroyAllWindows()
            
        if ret:  
            self.image_msg = self.bridge.cv2_to_imgmsg(self.open_cv_image, "bgr8")
            self.store_img(self.image_msg)
            count = count + 20
            self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES,count)            
    """ 

    # Defining multiple models
    def model(self, model):
        if model == "yolo":
            self.detection_model = YOLO("yolov8n.pt")
            return self.detection_model
        elif model == "detectron2":
            self.cfg = get_cfg()
            self.cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
            self.cfg.MODEL.DEVICE= 'cpu'
            self.detection_model = DefaultPredictor(self.cfg)
            return self.detection_model
        else:
            BaseOptions = mp.tasks.BaseOptions
            ObjectDetector = mp.tasks.vision.ObjectDetector
            ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            options = ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path='./src/image_processing/image_processing/efficientdet_lite0.tflite'),
                max_results=10,
                running_mode=VisionRunningMode.IMAGE, category_allowlist=["chair"], 
                score_threshold = 0.5
            )
            self.detection_model = ObjectDetector.create_from_options(options)
            return self.detection_model
                
    # To select custom class in detectron
    def only_keep_chair_class(self,outputs):
        oim = self.last_robot_image        
        cls = outputs['instances'].pred_classes
        scores = outputs["instances"].scores 
        boxes = outputs['instances'].pred_boxes

        indx_to_keep = (cls == 56).nonzero().flatten().tolist()
            
        cls1 = torch.tensor(np.take(cls.cpu().numpy(), indx_to_keep))
        scores1 = torch.tensor(np.take(scores.cpu().numpy(), indx_to_keep))
        boxes1 = Boxes(torch.tensor(np.take(boxes.tensor.cpu().numpy(), indx_to_keep, axis=0)))
        
        obj = detectron2.structures.Instances(image_size=(oim.shape[0], oim.shape[1]))
        obj.set('pred_classes', cls1)
        obj.set('scores', scores1)
        obj.set('pred_boxes',boxes1)
        return obj
        
        
    def store_img(self, img: Image):
        try:
            self.last_robot_image = self.bridge.imgmsg_to_cv2(img_msg=img)
        except:
            self.get_logger().error("Unable to convert the image!")
            return 
        
        #results = self.model('yolo').track(self.last_robot_image, conf=0.3 )
        
        original_outputs = self.model("detectron2")(self.last_robot_image) 
        modified_outputs = self.only_keep_chair_class(original_outputs)
        
        v = Visualizer(self.last_robot_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0)
        out = v.draw_instance_predictions(modified_outputs.to("cpu"))
        cv2.imwrite('detections.png',out.get_image())
        self.save_frame(out.get_image())
        
    def start_recording(self, video_file_name):
        self.video_file = video_file_name
        self.video_file_name = video_file_name + '.avi'
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.output_video = cv2.VideoWriter(self.video_file_name, self.codec,5,(250,250))
        if self.output_video.isOpened():
            self.get_logger().info("Video writer is opened successfully!")
        else:
            self.get_logger().error("Error in writing the video!")
        
        print("initialized {}".format(self.video_file))
    
    def save_frame(self, frame):
        self.output_video.write(frame[:,:,::-1])
        self.get_logger().info("saving the video...")
        
    def end_recording(self):
        self.output_video.release()
        

def main():
    rclpy.init()
    node = ObjectDetNode()
    try:
        rclpy.spin(node)
    except:
        node.end_recording()
        rclpy.shutdown()

if __name__=='__main__':
    main()