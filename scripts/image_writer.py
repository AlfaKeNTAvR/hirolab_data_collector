#!/usr/bin/env python
"""

Author(s):

TODO:
    1. Protection against empty output_folder_path and output_file_name.

"""

# # Standart libraries:
import rospy
import os
import cv2
from cv_bridge import (
    CvBridge,
    CvBridgeError,
)

# # Third party libraries:

# # Standart messages and services:
from std_msgs.msg import (
    Bool,
    String,
)
from sensor_msgs.msg import (Image)
from std_srvs.srv import (Empty)

# # Third party messages and services:


class ImageWriter:
    """
    
    """

    def __init__(
        self,
        node_name,
        image_topic,
        output_file_path,
        image_writing_period,
        image_rotation,
        enable_imshow,
    ):
        """
        
        """

        # # Private CONSTANTS:
        # NOTE: By default all new class CONSTANTS should be private.
        self.__NODE_NAME = node_name
        self.__IMAGE_TOPIC = image_topic
        self.__OUTPUT_FILE_PATH = output_file_path
        self.__IMAGE_WRITING_PERIOD = image_writing_period
        self.__IMAGE_ROTATION = image_rotation
        self.__ENABLE_IMSHOW = enable_imshow

        self.__BRIDGE = CvBridge()

        # # Public CONSTANTS:

        # # Private variables:
        # NOTE: By default all new class variables should be private.
        self.__cv_image = None
        self.__output_file_path = None
        self.__recorder_status = 'finished'  # 'paused', 'recording'
        self.__elapsed_time = 0.0

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
        self.__dependency_status_topics['image_topic'] = (
            rospy.Subscriber(
                f'{self.__IMAGE_TOPIC}',
                Image,
                self.__image_topic_callback,
            )
        )

        # # Service provider:
        rospy.Service(
            f'{self.__NODE_NAME}/resume_recording',
            Empty,
            self.__resume_recording_handler,
        )
        rospy.Service(
            f'{self.__NODE_NAME}/pause_recording',
            Empty,
            self.__pause_recording_handler,
        )
        rospy.Service(
            f'{self.__NODE_NAME}/finish_recording',
            Empty,
            self.__finish_recording_handler,
        )

        # # Service subscriber:

        # # Topic publisher:
        self.__status = rospy.Publisher(
            f'{self.__NODE_NAME}/recorder_status',
            String,
            queue_size=1,
        )

        # # Topic subscriber:
        rospy.Subscriber(
            f'{self.__IMAGE_TOPIC}',
            Image,
            self.__image_topic_callback,
        )

        # # Timers:
        rospy.Timer(
            rospy.Duration(self.__IMAGE_WRITING_PERIOD),
            self.__write_image_timer,
        )

        # # Node parameters:
        rospy.loginfo(
            f'{self.__NODE_NAME}:'
            '\nNode parameters:'
            f'\n- image_topic: {self.__IMAGE_TOPIC}'
            f'\n- output_file_path: {self.__OUTPUT_FILE_PATH}'
            f'\n- image_writing_period: {self.__IMAGE_WRITING_PERIOD}'
            f'\n- image_rotation: {self.__IMAGE_ROTATION}'
            f'\n- enable_imshow: {self.__ENABLE_IMSHOW}'
            '\n'
        )

    # # Dependency status callbacks:
    # NOTE: each dependency topic should have a callback function, which will

    # # Service handlers:
    def __resume_recording_handler(self, request):
        """
        
        """

        # Start a new recording.
        if self.__recorder_status == 'finished':
            # Get a unique directory for image writing:
            directory, filename = os.path.split(self.__OUTPUT_FILE_PATH)
            new_directory = self.__get_unique_directory(directory)
            self.__output_file_path = os.path.join(new_directory, filename)

            # Reset elapsed time.
            self.__elapsed_time = 0.0

            rospy.logwarn(
                f'{self.__NODE_NAME}: '
                f'\n{new_directory} recording has started.'
            )

        # Resume the previous recording:
        elif self.__recorder_status == 'paused':
            rospy.logwarn(f'{self.__NODE_NAME}: recording has resumed.')

        self.__recorder_status = 'recording'

        return []

    def __pause_recording_handler(self, request):
        """
        
        """

        if self.__recorder_status == 'recording':
            self.__write_image()

            rospy.logwarn(f'{self.__NODE_NAME}: recording has paused.')

            self.__recorder_status = 'paused'

        elif self.__recorder_status == 'paused':
            rospy.logwarn(
                f'{self.__NODE_NAME}: '
                f'\nThe current recording has been already paused!'
            )

        elif self.__recorder_status == 'finished':
            rospy.logwarn(
                f'{self.__NODE_NAME}: Nothing to pause.'
                f'\nThe previous recording has been already finished!'
            )

        return []

    def __finish_recording_handler(self, request):
        """
        
        """

        if self.__recorder_status != 'finished':
            self.__write_image()

            rospy.logwarn(f'{self.__NODE_NAME}: recording has finished.')

            self.__recorder_status = 'finished'

        else:
            rospy.logwarn(
                f'{self.__NODE_NAME}: Nothing to finish.'
                f'\nThe previous recording has been already finished!'
            )

        return []

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

        self.__cv_image = cv_image

        if not self.__is_initialized:
            self.__dependency_status['image_topic'] = True

    # # Timer callbacks:
    def __write_image_timer(self, event):
        """
        
        """

        if self.__recorder_status == 'recording':
            self.__write_image()

    # # Private methods:
    # NOTE: By default all new class methods should be private.
    def __check_initialization(self):
        """Monitors required criteria and sets is_initialized variable.

        Monitors nodes' dependency status by checking if dependency's
        is_initialized topic has at most one publisher (this ensures that
        dependency node is alive and does not have any duplicates) and that it
        publishes True. If dependency's status was True, but get_num_connections
        is not equal to 1, this means that the connection is lost and emergency
        actions should be performed.

        Once all dependencies are initialized and additional criteria met, the
        nodes' is_initialized status changes to True. This status can change to
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

        # NOTE (optionally): Add more initialization criterea if needed.
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

    def __get_unique_directory(self, directory):
        """
        Generates a unique directory name based on the provided directory. 
        If the directory exists, appends a counter to the directory name.
        
        :param directory: The base directory path.
        :return: A unique directory path.
        """

        counter = 1
        image_folder = os.path.basename(directory)
        parent_directory = os.path.dirname(directory)
        new_directory = directory

        while os.path.exists(new_directory):
            new_folder_name = f'{image_folder}({counter})'
            new_directory = os.path.join(parent_directory, new_folder_name)

            try:
                if not os.path.exists(new_directory):
                    os.makedirs(new_directory)
                    break
            except OSError as e:
                rospy.logwarn(
                    f'{self.__NODE_NAME}:'
                    f'an error occured while creating directory {new_directory}.\n'
                    f'{e}'
                )
                break

            counter += 1

        if os.path.exists(directory):
            rospy.logwarn(
                f'{self.__NODE_NAME}: the directory {directory} already exists.'
                f'\nA new directory was created: {new_directory}'
            )

        return new_directory

    def __write_image(self):
        """
        
        """

        try:
            # Ensure the directory exists.
            directory = os.path.dirname(self.__output_file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            directory, filename = os.path.split(self.__output_file_path)
            file_base, file_extension = os.path.splitext(filename)
            new_filename = f'{file_base}_{round(self.__elapsed_time, 2)}{file_extension}'
            new_output_file_path = os.path.join(directory, new_filename)

            cv2.imwrite(new_output_file_path, self.__cv_image)

        except Exception as e:
            rospy.logerr(
                (
                    f'{self.__NODE_NAME}: '
                    f'an error occured while writing an image.\n'
                    f'{e} \n'
                ),
            )
            return

        # rospy.loginfo(f'{self.__NODE_NAME}: the image was saved.')
        self.__elapsed_time += self.__IMAGE_WRITING_PERIOD

    def __imshow(self):
        """
        
        """

        # Optionally show the frame.
        if self.__enable_imshow:
            cv2.imshow(
                'self.__cv_image',
                self.__cv_image,
            )

            if (
                cv2.waitKey(1) & 0xFF == ord('q') or
                cv2.getWindowProperty('self.__cv_image',
                                      cv2.WND_PROP_VISIBLE) < 1
            ):
                cv2.destroyAllWindows()
                self.__enable_imshow = False

    # # Public methods:
    # NOTE: By default all new class methods should be private.
    def main_loop(self):
        """
        
        """

        self.__check_initialization()

        if not self.__is_initialized:
            return

        # NOTE: Add code (function calls), which has to be executed once the
        # node was successfully initialized.
        self.__status.publish(self.__recorder_status)

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
        'image_writer',
        log_level=rospy.INFO,  # rospy.DEBUG to view debug messages.
    )

    rospy.loginfo('\n\n\n\n\n')  # Add whitespaces to separate logs.

    # # ROS launch file parameters:
    node_name = rospy.get_name()

    node_frequency = rospy.get_param(
        param_name=f'{rospy.get_name()}/node_frequency',
        default=100,
    )
    image_topic = rospy.get_param(
        param_name=f'{rospy.get_name()}/image_topic',
        default='/camera/color/image_raw',
    )
    output_folder_path = rospy.get_param(
        param_name=f'{rospy.get_name()}/output_folder_path',
        # default='/home/alfakentavr/data',
    )
    output_file_name = rospy.get_param(
        param_name=f'{rospy.get_name()}/output_file_name',
        # default='experiment-name_p0_m0_t0.png',
    )
    camera_name = rospy.get_param(
        param_name=f'{rospy.get_name()}/camera_name',
        default='camera',
    )
    image_writing_period = rospy.get_param(
        param_name=f'{rospy.get_name()}/image_writing_period',
        default=0.5,
    )
    image_rotation = rospy.get_param(
        param_name=f'{rospy.get_name()}/image_rotation',
        default=0,
    )
    enable_imshow = rospy.get_param(
        param_name=f'{node_name}/enable_imshow',
        default=True,
    )

    # Get absolute output file path:
    sub_path = output_file_name.split('.')[0].split('_')

    experiment_name = sub_path[0]
    participant_number = sub_path[1]
    mode_number = sub_path[2]
    trial_number = sub_path[3]

    output_file_path = (
        f'{output_folder_path}'
        f'/{experiment_name}/{participant_number}/{mode_number}/{trial_number}/{camera_name}_images'
        f'/{output_file_name}'
    )

    class_instance = ImageWriter(
        node_name=node_name,
        image_topic=image_topic,
        output_file_path=output_file_path,
        image_writing_period=image_writing_period,
        image_rotation=image_rotation,
        enable_imshow=enable_imshow,
    )

    rospy.on_shutdown(class_instance.node_shutdown)
    node_rate = rospy.Rate(node_frequency)

    while not rospy.is_shutdown():
        class_instance.main_loop()
        node_rate.sleep()


if __name__ == '__main__':
    main()
