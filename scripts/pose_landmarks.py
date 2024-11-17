#!/usr/bin/env python
"""

"""

import rospy
import cv2
from cv_bridge import (
    CvBridge,
    CvBridgeError,
)
import mediapipe as mp
import pyrealsense2 as rs2
import numpy as np
import math
from copy import (
    copy,
    deepcopy,
)
import warnings

from std_msgs.msg import (
    Bool,
    Float64MultiArray,
)
from sensor_msgs.msg import (
    Image,
    CameraInfo,
)
from geometry_msgs.msg import (Point)

from data_collector.msg import (UpperBodyKeypoints)


class PoseLandmarks:
    """
    
    """

    def __init__(
        self,
        node_name,
        camera_name,
        image_rotation,
        depth_averaging_enable,
        depth_averaging_count,
        coordinates_averaging_enable,
        coordinates_averaging_count,
        enable_imshow,
    ):
        """
        
        """

        # # Private constants:
        self.__NODE_NAME = node_name
        self.__CAMERA_NAME = camera_name
        self.__IMAGE_ROTATION = image_rotation
        self.__DEPTH_AVERAGING = {
            'enable': depth_averaging_enable,
            'count': depth_averaging_count,
        }
        self.__COORDINATES_AVERAGING = {
            'enable': coordinates_averaging_enable,
            'count': coordinates_averaging_count,
        }
        self.__ENABLE_IMSHOW = enable_imshow

        self.__BRIDGE = CvBridge()

        # Pose landmarks detection:
        self.__MP_DRAWING = mp.solutions.drawing_utils
        self.__MP_POSE = mp.solutions.pose
        # Settings: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
        self.__POSE = self.__MP_POSE.Pose(
            model_complexity=2,  # 0 - Lite, 1 - Full, 2 - Heavy.
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.85,
        )

        # # Public constants:

        # # Private variables:
        self.__cv_color_image = None
        self.__cv_depth_image = None
        self.__cv_depth_image_snapshot = None
        self.__instristics = None

        self.__physical_workload = 0

        self.__pose_landmarks = None
        self.__pose_landmarks_frame = None
        self.__keypoints = list()

        self.__depths = [[], [], [], [], [], [], [], []]
        self.__coordinates = [
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        ]
        self.__enable_imshow = self.__ENABLE_IMSHOW

        self.__rotation_matrix = None
        self.__rotated_width = None
        self.__rotated_height = None

        # # Public variables:

        # # Initialization and dependency status topics:
        self.__is_initialized = False
        self.__dependency_initialized = False

        self.__node_is_initialized = rospy.Publisher(
            f'{self.__NODE_NAME}/is_initialized',
            Bool,
            queue_size=1,
        )

        # NOTE: Specify dependency initial False initial status.
        self.__dependency_status = {}

        # NOTE: Specify dependency is_initialized topic (or any other topic,
        # which will be available when the dependency node is running properly).
        self.__dependency_status_topics = {}

        self.__dependency_status['image_topic'] = False
        self.__dependency_status_topics['image_topic'] = rospy.Subscriber(
            f'/{self.__CAMERA_NAME}/color/image_raw',
            Image,
            self.__image_topic_callback,
        )

        self.__dependency_status['align_depth_topic'] = False
        self.__dependency_status_topics['align_depth_topic'] = rospy.Subscriber(
            f'/{self.__CAMERA_NAME}/aligned_depth_to_color/image_raw',
            Image,
            self.__realsense_alighed_depth_callback,
        )

        # # Service provider:

        # # Service subscriber:

        # # Topic publisher:
        self.__pose_landmarks_image = rospy.Publisher(
            f'{self.__NODE_NAME}/image',
            Image,
            queue_size=1,
        )
        self.__upperbody_keypoints = rospy.Publisher(
            f'/physical_workload/upperbody_keypoints',
            UpperBodyKeypoints,
            queue_size=1,
        )

        # # Topic subscriber:
        rospy.Subscriber(
            f'/{self.__CAMERA_NAME}/color/image_raw',
            Image,
            self.__image_topic_callback,
        )
        rospy.Subscriber(
            f'/{self.__CAMERA_NAME}/aligned_depth_to_color/image_raw',
            Image,
            self.__realsense_alighed_depth_callback,
        )
        rospy.Subscriber(
            f'/{self.__CAMERA_NAME}/aligned_depth_to_color/camera_info',
            CameraInfo,
            self.__realsense_alighed_depth_info_callback,
        )

        rospy.Subscriber(
            '/physical_workload',
            Float64MultiArray,
            self.__physical_workload_callback,
        )

    # # Dependency status callbacks:
    # NOTE: each dependency topic should have a callback function, which will
    # set __dependency_status variable.

    # # Service handlers:

    # # Topic callbacks:
    def __image_topic_callback(self, message):
        """
        
        """

        cv_image = None

        try:
            cv_image = self.__BRIDGE.imgmsg_to_cv2(
                message,
                'bgr8',
            )

        except CvBridgeError as e:
            rospy.logerr(
                (
                    f'{self.__NODE_NAME}:'
                    f' an error occured while converting from Image message to cv2. \n'
                    f'{e} \n'
                ),
            )
            return

        if self.__IMAGE_ROTATION != 0:
            # Calculate rotation matrix and new width and height to avoid image
            # shrinking and distortion.
            if not self.__dependency_status['image_topic']:
                original_height, original_width = cv_image.shape[:2]

                # getRotationMatrix2D needs coordinates in reverse order (width,
                # height) compared to shape.
                image_center = (original_width // 2, original_height // 2)

                self.__rotation_matrix = cv2.getRotationMatrix2D(
                    image_center,
                    self.__IMAGE_ROTATION,
                    1.0,
                )

                # Rotation calculates the cos and sin, taking absolutes of
                # those.
                abs_cos = abs(self.__rotation_matrix[0, 0])
                abs_sin = abs(self.__rotation_matrix[0, 1])

                # Find the new width and height bounds.
                self.__rotated_width = int(
                    original_height * abs_sin + original_width * abs_cos
                )
                self.__rotated_height = int(
                    original_height * abs_cos + original_width * abs_sin
                )

                # Subtract old image center (bringing image back to origo) and
                # adding the new image center coordinates.
                (self.__rotation_matrix[0, 2]
                ) += (self.__rotated_width / 2 - image_center[0])
                (self.__rotation_matrix[1, 2]
                ) += (self.__rotated_height / 2 - image_center[1])

            # Rotate the image.
            cv_image = cv2.warpAffine(
                cv_image,
                self.__rotation_matrix,
                (self.__rotated_width, self.__rotated_height),
            )

        self.__cv_color_image = cv_image

        if not self.__is_initialized:
            self.__dependency_status['image_topic'] = True

    def __realsense_alighed_depth_callback(self, message):
        """

        """

        if not self.__is_initialized:
            self.__dependency_status['align_depth_topic'] = True
            return

        cv_depth_image = self.__BRIDGE.imgmsg_to_cv2(
            img_msg=message,
            desired_encoding=message.encoding,
        )

        # Apply the rotation to the image.
        if self.__IMAGE_ROTATION != 0:
            cv_depth_image = cv2.warpAffine(
                cv_depth_image,
                self.__rotation_matrix,
                (self.__rotated_width, self.__rotated_height),
            )

        self.__cv_depth_image = cv_depth_image

    def __realsense_alighed_depth_info_callback(self, message):
        """

        """

        # Set the variable once on the first topic callback.
        if self.__instristics:
            return

        self.__intrinsics = rs2.intrinsics()
        self.__intrinsics.width = message.width
        self.__intrinsics.height = message.height
        self.__intrinsics.ppx = message.K[2]
        self.__intrinsics.ppy = message.K[5]
        self.__intrinsics.fx = message.K[0]
        self.__intrinsics.fy = message.K[4]

        if message.distortion_model == 'plumb_bob':
            self.__intrinsics.model = rs2.distortion.brown_conrady

        elif message.distortion_model == 'equidistant':
            self.__intrinsics.model = rs2.distortion.kannala_brandt4

        self.__intrinsics.coeffs = [i for i in message.D]

    def __physical_workload_callback(self, message: Float64MultiArray):
        """
        
        """

        self.__physical_workload = message.data[0]

    # # Timer functions:

    # # Private methods:
    def __check_initialization(self):
        """Monitors required criteria and sets is_initialized variable.

        Monitors nodes' dependency status by checking if dependency's
        is_initialized topic has at most one publisher (this ensures that
        dependency node is alive and does not have any duplicates) and that it
        publishes True. If dependency's status was True, but get_num_connections
        is not equal to 1, this means that the connection is lost and emergency
        actions should be performed.

        Once all dependencies are initialized and additional criteria met, the
        nodes is_initialized status changes to True. This status can change to
        False any time to False if some criteria are no longer met.
        
        """

        self.__dependency_initialized = True

        for key in self.__dependency_status:
            if self.__dependency_status_topics[key].get_num_connections() != 1:
                if self.__dependency_status[key]:
                    rospy.logerr(
                        (f'{self.__NODE_NAME}: '
                         f'lost connection to {key}!')
                    )

                    # # Emergency actions on lost connection:
                    # NOTE (optionally): Add code, which needs to be executed if
                    # connection to any of dependencies was lost.

                    if key == 'realsense_camera':
                        self.__instristics = None

                self.__dependency_status[key] = False

            if not self.__dependency_status[key]:
                self.__dependency_initialized = False

        if not self.__dependency_initialized:
            waiting_for = ''
            for key in self.__dependency_status:
                if not self.__dependency_status[key]:
                    waiting_for += f'\n- waiting for {key} node...'

            rospy.logwarn_throttle(
                15,
                (
                    f'{self.__NODE_NAME}:'
                    f'{waiting_for}'
                    # f'\nMake sure those dependencies are running properly!'
                ),
            )

        # NOTE: Add more initialization criterea if needed.
        if (self.__dependency_initialized):
            if not self.__is_initialized:
                rospy.loginfo(f'\033[92m{self.__NODE_NAME}: ready.\033[0m',)

                self.__is_initialized = True

        else:
            if self.__is_initialized:
                # NOTE (optionally): Add code, which needs to be executed if the
                # nodes's status changes from True to False.
                pass

            self.__is_initialized = False

        self.__node_is_initialized.publish(self.__is_initialized)

    def __pose_estimation(self):
        """
        The output contains the following normalized coordinates (Landmarks):
        - x and y: Landmark coordinates normalized between 0.0 and 1.0 by the
        image width (x) and height (y). 

        - z: The landmark depth, with the depth
        at the midpoint of the hips as the origin. The smaller the value, the
        closer the landmark is to the camera. The magnitude of z uses roughly
        the same scale as x. 
        
        - visibility: The likelihood of the landmark being
        visible within the image.

        X is horizontal, Y is vertical. Top left is (0.0, 0.0), bottom right is
        (1.0, 1.0).

        0 - nose
        1 - left eye (inner)
        2 - left eye
        3 - left eye (outer)
        4 - right eye (inner)
        5 - right eye
        6 - right eye (outer)
        7 - left ear
        8 - right ear
        9 - mouth (left)
        10 - mouth (right)
        11 - left shoulder
        12 - right shoulder
        13 - left elbow
        14 - right elbow
        15 - left wrist
        16 - right wrist
        17 - left pinky
        18 - right pinky
        19 - left index
        20 - right index
        21 - left thumb
        22 - right thumb
        23 - left hip
        24 - right hip
        25 - left knee
        26 - right knee
        27 - left ankle
        28 - right ankle
        29 - left heel
        30 - right heel
        31 - left foot index
        32 - right foot index

        https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#image_2
        https://www.hackersrealm.net/post/realtime-human-pose-estimation-using-python
        
        """

        pose_results = None

        # If no color image is available.
        if self.__cv_color_image is None:
            return

        frame = self.__cv_color_image

        # Snapshot of corresponding depth image.
        self.__cv_depth_image_snapshot = deepcopy(self.__cv_depth_image)

        # If no depth image is available.
        if self.__cv_depth_image_snapshot is None:
            return

        try:

            # Process the frame for pose detection.
            pose_results = self.__POSE.process(frame)

            # If landmarks were detected.
            if pose_results.pose_landmarks:

                # Draw pose landmarks on the frame.
                self.__MP_DRAWING.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    self.__MP_POSE.POSE_CONNECTIONS,
                )

                self.__pose_landmarks = pose_results
                self.__update_upperbody_keypoints(
                    self.__pose_landmarks.pose_landmarks.landmark
                )

        except Exception as e:
            print(e)

        # Display physical workload.
        cv2.putText(
            frame,
            text=str(int(self.__physical_workload)),
            org=(690, 415),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
            color=(0, 0, 255),
            thickness=6,
        )

        self.__pose_landmarks_frame = frame

    def __publish_landmarks_image(self):
        """
        
        """

        if self.__pose_landmarks_frame is None:
            return

        image_message = self.__BRIDGE.cv2_to_imgmsg(
            self.__pose_landmarks_frame,
            encoding='bgr8',
        )
        self.__pose_landmarks_image.publish(image_message)

    def __publish_upperbody_keypoints(self):
        """
        
        """

        keypoints = UpperBodyKeypoints()
        keypoints.keypoints = self.__keypoints

        self.__upperbody_keypoints.publish(keypoints)

    def __get_3d_coordinates(self, pixel_x, pixel_y, keypoint_i):
        """
        https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a

        """

        x_min = pixel_x - 5
        x_max = pixel_x + 5
        y_min = pixel_y - 5
        y_max = pixel_y + 5

        # I expect to see RuntimeWarnings in this block.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            depth = np.mean(
                self.__cv_depth_image_snapshot[y_min:y_max, x_min:x_max]
            )

        if self.__DEPTH_AVERAGING['enable']:
            # Initialize values.
            if len(self.__depths[keypoint_i]) == 0:
                self.__depths[keypoint_i] = (
                    [depth] * self.__DEPTH_AVERAGING['count']
                )

            self.__depths[keypoint_i].pop(0)
            self.__depths[keypoint_i].append(depth)

            depth_averaged = sum(self.__depths[keypoint_i]
                                ) / len(self.__depths[keypoint_i])

            depth = copy(depth_averaged)

        coordinates_3d = rs2.rs2_deproject_pixel_to_point(
            self.__intrinsics,
            [pixel_x, pixel_y],
            depth,
        )

        # Protection against out of bounds keypoints. Landmark detection model
        # can predict a landmark location outside of the image. However, there
        # are no corresponding depth image values, hence, the 3d coordinate
        # values will be nan.

        # FIXME: Need to add check for the outside of the image keypoints on the
        # previous steps.
        if any(math.isnan(x) for x in coordinates_3d):
            return False

        if self.__COORDINATES_AVERAGING['enable']:
            # Initialize values.
            if len(self.__coordinates[keypoint_i]) == 0:
                for _ in range(self.__COORDINATES_AVERAGING['count']):
                    self.__coordinates[keypoint_i] = np.append(
                        self.__coordinates[keypoint_i],
                        np.array(
                            [
                                coordinates_3d[0],
                                coordinates_3d[1],
                                coordinates_3d[2],
                            ]
                        ),
                    )

                self.__coordinates[keypoint_i] = (
                    self.__coordinates[keypoint_i]
                ).reshape(-1, 3)

            # Update the list.
            self.__coordinates[keypoint_i] = np.delete(
                self.__coordinates[keypoint_i], 0, axis=0
            )

            self.__coordinates[keypoint_i] = np.append(
                self.__coordinates[keypoint_i],
                np.array(
                    [
                        coordinates_3d[0],
                        coordinates_3d[1],
                        coordinates_3d[2],
                    ]
                )
            ).reshape(-1, 3)

            average_coordinates = np.mean(
                self.__coordinates[keypoint_i],
                axis=0,
            )

            coordinates_3d = copy(average_coordinates)

        keypoint = Point()

        # mm to m.
        keypoint.y = -coordinates_3d[0] / 1000
        keypoint.z = -coordinates_3d[1] / 1000
        keypoint.x = coordinates_3d[2] / 1000

        return keypoint

    def __update_upperbody_keypoints(self, pose_landmarks):
        """
        """

        keypoints = list()
        height, width = self.__pose_landmarks_frame.shape[:2]

        # 0 - left_shoulder, 1 - right_shoulder,
        # 2 - left_elbow, 3 - right_elbow,
        # 4 - left_wrist, 5 - right_wrist.
        for i in range(11, 17):

            keypoint = self.__get_3d_coordinates(
                pixel_x=int(pose_landmarks[i].x * width),
                pixel_y=int(pose_landmarks[i].y * height),
                keypoint_i=i - 11,
            )

            keypoints.append(keypoint)

        # 6 - chest.
        keypoint = self.__get_3d_coordinates(
            pixel_x=int(
                ((pose_landmarks[11].x + pose_landmarks[12].x) / 2) * width
            ),
            pixel_y=int(
                ((pose_landmarks[11].y + pose_landmarks[12].y) / 2) * height
            ),
            keypoint_i=6,
        )

        keypoints.append(keypoint)

        # 7 - waist.
        keypoint = self.__get_3d_coordinates(
            pixel_x=int(
                ((pose_landmarks[23].x + pose_landmarks[24].x) / 2) * width
            ),
            pixel_y=int(
                ((pose_landmarks[23].y + pose_landmarks[24].y) / 2) * height
            ),
            keypoint_i=7,
        )

        keypoints.append(keypoint)

        # Protection against out of bounds keypoints.
        if any(item is False for item in keypoints):
            return

        self.__keypoints = keypoints

    def __imshow(self):
        """
        
        """

        # Optionally show the frame.
        if self.__enable_imshow:
            cv2.imshow(
                'self.__cv_image',
                self.__pose_landmarks_frame,
            )

            if (
                cv2.waitKey(1) & 0xFF == ord('q') or
                cv2.getWindowProperty('self.__cv_image',
                                      cv2.WND_PROP_VISIBLE) < 1
            ):
                cv2.destroyAllWindows()
                self.__enable_imshow = False

                rospy.loginfo(f'{self.__NODE_NAME}: imshow window was closed.',)

    # # Public methods:
    def main_loop(self):
        """
        
        """

        self.__check_initialization()

        if not self.__is_initialized:
            return

        # NOTE: Add code (function calls), which has to be executed once the
        # node was successfully initialized.
        self.__pose_estimation()

        self.__publish_landmarks_image()
        self.__publish_upperbody_keypoints()

        if self.__pose_landmarks_frame is not None:
            self.__imshow()

    def node_shutdown(self):
        """
        
        """

        rospy.loginfo_once(f'{self.__NODE_NAME}: node is shutting down...',)

        # NOTE: Add code, which needs to be executed on nodes' shutdown here.
        # Publishing to topics is not guaranteed, use service calls or
        # set parameters instead.

        rospy.loginfo_once(f'{self.__NODE_NAME}: node has shut down.',)


def main():
    """
    
    """

    # # Default node initialization.
    # This name is replaced when a launch file is used.
    rospy.init_node(
        'pose_landmarks',
        log_level=rospy.INFO,  # rospy.DEBUG to view debug messages.
    )

    rospy.loginfo('\n\n\n\n\n')  # Add whitespaces to separate logs.

    # # ROS parameters:
    node_name = rospy.get_name()

    node_frequency = rospy.get_param(
        param_name=f'{rospy.get_name()}/node_frequency',
        default=100,
    )

    camera_name = rospy.get_param(
        param_name=f'{node_name}/camera_name',
        default='camera',
    )
    image_rotation = rospy.get_param(
        param_name=f'{node_name}/image_rotation',
        default=90,
    )
    depth_averaging_enable = rospy.get_param(
        param_name=f'{node_name}/depth_averaging_enable',
        default=True,
    )
    depth_averaging_count = rospy.get_param(
        param_name=f'{node_name}/depth_averaging_count',
        default=10,
    )
    coordinates_averaging_enable = rospy.get_param(
        param_name=f'{node_name}/coordinates_averaging_enable',
        default=True,
    )
    coordinates_averaging_count = rospy.get_param(
        param_name=f'{node_name}/coordinates_averaging_count',
        default=5,
    )
    enable_imshow = rospy.get_param(
        param_name=f'{node_name}/enable_imshow',
        default=False,
    )

    pose_landmarks = PoseLandmarks(
        node_name=node_name,
        camera_name=camera_name,
        image_rotation=image_rotation,
        depth_averaging_enable=depth_averaging_enable,
        depth_averaging_count=depth_averaging_count,
        coordinates_averaging_enable=coordinates_averaging_enable,
        coordinates_averaging_count=coordinates_averaging_count,
        enable_imshow=enable_imshow,
    )

    rospy.on_shutdown(pose_landmarks.node_shutdown)
    node_rate = rospy.Rate(node_frequency)

    while not rospy.is_shutdown():
        pose_landmarks.main_loop()
        node_rate.sleep()


if __name__ == '__main__':
    main()
