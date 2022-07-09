import numpy
import os
from transformers import AutoFeatureExtractor, DetrForSegmentation
import torch
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import String
from cv_bridge import CvBridge

ALGO_VERSION = os.getenv("MODEL_NAME")

if not ALGO_VERSION:
    ALGO_VERSION = '<default here>'


def predict(image: Image):
    feature_extractor = AutoFeatureExtractor.from_pretrained(ALGO_VERSION)
    model = DetrForSegmentation.from_pretrained(ALGO_VERSION)

    inputs = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs)

    # Convert output to be between 0 and 1
    sizes = torch.tensor([tuple(reversed(image.size))])
    result = feature_extractor.post_process_segmentation(output, sizes)
    
    return result[0]


class RosIO(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.declare_parameter('pub_image', False)
        self.declare_parameter('pub_boxes', True)
        self.image_subscription = self.create_subscription(
            Image,
            '/<name>/sub/image_raw',
            self.listener_callback,
            10
        )

        self.image_publisher = self.create_publisher(
            String,
            '/<name>/pub/image',
            1
        )
    
        self.detection_publisher = self.create_publisher(
            String,
            '/<name>/pub/detection_boxes',
            1
        )

    def get_detection_arr(self, result):
        dda = Detection2DArray()

        detections = []
        self.counter += 1

        ## Insert ROS Type Here 
        ## Output from HF
        '''
        [{'labels': tensor([93, 17, 75, 75, 63, 17]),
  'masks': tensor([[[1, 1, 1,  ..., 1, 1, 1],
           [1, 1, 1,  ..., 1, 1, 1],
           [0, 0, 1,  ..., 1, 1, 1],
           ...,
           [1, 1, 1,  ..., 1, 1, 1],
           [1, 1, 1,  ..., 1, 1, 1],
           [1, 1, 1,  ..., 1, 1, 1]],
  
          [[0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           ...,
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0]],
  
          [[0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           ...,
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0]],
  
          [[0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           ...,
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0]],
  
          [[1, 1, 1,  ..., 1, 1, 1],
           [1, 1, 1,  ..., 1, 1, 1],
           [1, 1, 1,  ..., 1, 1, 1],
           ...,
           [1, 1, 1,  ..., 1, 1, 1],
           [1, 1, 1,  ..., 1, 1, 1],
           [1, 1, 1,  ..., 1, 1, 1]],
  
          [[0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           ...,
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0],
           [0, 0, 0,  ..., 0, 0, 0]]]),
  'scores': tensor([0.9094, 0.9941, 0.9987, 0.9995, 0.9722, 0.9994],
         grad_fn=<IndexBackward0>)}]
        '''


    def listener_callback(self, msg: Image):
        bridge = CvBridge()
        cv_image: numpy.ndarray = bridge.imgmsg_to_cv2(msg)
        converted_image = PilImage.fromarray(numpy.uint8(cv_image), 'RGB')
        result = predict(converted_image)
        print(f'Predicted Bounding Boxes')

        if self.get_parameter('pub_image').value:
            self.image_publisher.publish(msg)

        if self.get_parameter('pub_boxes').value:
            detections = self.get_detection_arr(result)
            self.detection_publisher.publish(detections)

        


def main(args=None):
    print('<name> Started')

    rclpy.init(args=args)

    minimal_subscriber = RosIO()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
