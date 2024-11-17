#!/usr/bin/env python
"""

Author(s):

TODO:
    - Add data_writer dependency on topic_subscriber(s).
    - Add launch file parameters parsing.

"""

# # Standart libraries:
import rospy
import rostopic
import pandas as pd
import os
from datetime import (datetime)

# # Third party libraries:

# # Standart messages and services:
from std_msgs.msg import (
    Bool,
    String,
)
from std_srvs.srv import (Empty)

# # Third party messages and services:


class TopicSubscriber:
    """
    
    """

    def __init__(
        self,
        node_name,
        topic_name,
        topic_specifier,
    ):
        """
        
        """

        # # Private CONSTANTS:
        # NOTE: By default all new class CONSTANTS should be private.
        self.__NODE_NAME = node_name
        self.__TOPIC_NAME = topic_name
        self.__TOPIC_SPECIFIER = topic_specifier
        self.__MESSAGE_TYPE, _, _ = rostopic.get_topic_class(self.__TOPIC_NAME)

        # # Public CONSTANTS:

        # # Private variables:
        # NOTE: By default all new class variables should be private.

        # # Public variables:
        self.message_data = None

        # # Initialization and dependency status topics:

        # # Service provider:

        # # Service subscriber:

        # # Topic publisher:

        # # Topic subscriber:
        if self.__MESSAGE_TYPE is None:
            rospy.logerr(
                (
                    f'{self.__NODE_NAME}:'
                    f'could not determine message type for topic {self.__TOPIC_NAME}'
                ),
            )
            return

        rospy.Subscriber(
            f'{self.__TOPIC_NAME}',
            self.__MESSAGE_TYPE,
            self.__generic_callback,
        )

        # # Timers:

    # # Dependency status callbacks:

    # # Service handlers:

    # # Topic callbacks:
    def __generic_callback(self, message):
        """

        """

        try:
            self.message_data = self.__message_to_dictionary(
                eval(f'message{self.__TOPIC_SPECIFIER}')
            )

        except Exception as e:
            rospy.logerr(
                (
                    f'{self.__NODE_NAME}:'
                    f' an error occured in a callback for {self.__TOPIC_NAME}{self.__TOPIC_SPECIFIER}. \n'
                    f'{e} \n'
                ),
            )

    # # Timer callbacks:

    # # Private methods:
    def __message_to_dictionary(self, message):
        """Converts a ROS message to a Python dictionary.
         
        This function handles both simple and complex ROS message types. For
        complex types, it recursively processes each field and converts it to a
        corresponding dictionary entry. For simple data types like float, int,
        or string, it returns the value directly.

        If a message consists of a single field, the value of that field is
        returned directly rather than in a dictionary. For lists contained
        within message fields, each element is also converted to a dictionary if
        it is a complex type.

        Parameters: 
            - message (various ROS message types): The ROS message to
            convert. This can be a complex message with multiple fields or a
            simple message with a single data type.

        Returns: 
            - dict or single value: A dictionary representation of the ROS
            message if it is a complex type with multiple fields. If the ROS
            message is a simple type or consists of a single field, the value of
            that field is returned directly.

        """

        # Check if msg is a simple data type (like float, int, string):
        if not hasattr(message, '__slots__'):
            return message

        message_dict = {}
        for field in message.__slots__:
            value = getattr(message, field)

            # If value is a list:
            if isinstance(value, list):
                message_dict[field] = [
                    self.__message_to_dictionary(item)
                    if hasattr(item, '_type') else item for item in value
                ]

            # If the value is another message:
            elif hasattr(value, '_type'):
                message_dict[field] = self.__message_to_dictionary(value)

            else:
                message_dict[field] = value

        # If there's only one field, return its value directly (e.g. 'data' for
        # standart messages):
        if len(message_dict) == 1:
            return next(iter(message_dict.values()))

        return message_dict

    # # Public methods:


class DataWriter:
    """
    
    """

    def __init__(
        self,
        node_name,
        output_file_path,
        data_recording_period,
        file_writing_period,
        topic_subscriber_instances,
        column_names,
    ):
        """
        
        """

        # # Private CONSTANTS:
        self.__NODE_NAME = node_name
        self.__DATA_RECORDING_PERIOD = data_recording_period
        self.__FILE_WRITING_PERIOD = file_writing_period
        self.__TOPIC_SUBSCRIBER_INSTANCES = topic_subscriber_instances
        self.__COLUMN_NAMES = column_names
        self.__COLUMN_NAMES.insert(0, 'current_datetime')
        self.__COLUMN_NAMES.insert(1, 'elapsed_time')

        # # Public CONSTANTS:

        # # Private variables:
        self.__output_file_path = output_file_path
        self.__recorder_status = 'finished'  # 'paused', 'recording'

        # # Public variables:
        self.data_frame = pd.DataFrame(columns=self.__COLUMN_NAMES)
        self.elapsed_time = 0.0

        # # Initialization and dependency status topics:
        self.__is_initialized = False
        self.__dependency_initialized = False

        self.__node_is_initialized = rospy.Publisher(
            f'{self.__NODE_NAME}/is_initialized',
            Bool,
            queue_size=1,
        )

        # NOTE: Specify dependency initial False initial status.
        self.__dependency_status = {
            # 'dependency_node_name': False,
        }

        # NOTE: Specify dependency is_initialized topic (or any other topic,
        # which will be available when the dependency node is running properly).
        self.__dependency_status_topics = {}

        # self.__dependency_status_topics['<dependency_node_name>'] = (
        #     rospy.Subscriber(
        #         f'/<dependency_node_name>/is_initialized',
        #         Bool,
        #         self.__dependency_name_callback,
        #     )
        # )

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

        # # Timers:
        rospy.Timer(
            rospy.Duration(self.__DATA_RECORDING_PERIOD),
            self.__update_dataframe_timer,
        )
        rospy.Timer(
            rospy.Duration(self.__FILE_WRITING_PERIOD),
            self.__write_to_csv_timer,
        )

    # # Dependency status callbacks:

    # # Service handlers:
    def __resume_recording_handler(self, request):
        """
        
        """

        # Start a new recording.
        if self.__recorder_status == 'finished':
            self.__output_file_path = self.__get_unique_filepath(
                self.__output_file_path
            )

            # Reset the DataFrame and the elapsed time.
            self.data_frame = pd.DataFrame(columns=self.__COLUMN_NAMES)
            self.elapsed_time = 0.0

            rospy.logwarn(
                f'{self.__NODE_NAME}: '
                f'\n{self.__output_file_path} recording has started.'
            )

        # Resume the previous recording:
        elif self.__recorder_status == 'paused':
            rospy.logwarn(
                f'{self.__NODE_NAME}:'
                f'\n{self.__output_file_path} recording has resumed.'
            )

        self.__recorder_status = 'recording'

        return []

    def __pause_recording_handler(self, request):
        """
        
        """

        if self.__recorder_status == 'recording':
            self.write_to_csv()

            rospy.logwarn(
                f'{self.__NODE_NAME}:'
                f'\n{self.__output_file_path} recording has paused.'
            )

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
            self.write_to_csv()

            rospy.logwarn(
                f'{self.__NODE_NAME}:'
                f'\n{self.__output_file_path} recording has finished.'
            )

            self.__recorder_status = 'finished'

        else:
            rospy.logwarn(
                f'{self.__NODE_NAME}: Nothing to finish.'
                f'\nThe previous recording has been already finished!'
            )

        return []

    # # Topic callbacks:

    # # Timer callbacks:
    def __update_dataframe_timer(self, event):
        """
        
        """

        if self.__recorder_status == 'recording':
            self.update_dataframe()

    def __write_to_csv_timer(self, event):
        """
        
        """

        if self.__recorder_status == 'recording':
            self.write_to_csv()

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

    def __get_unique_filepath(self, file_path):
        """
        Appends a counter to the filename if the file already exists, while
        keeping the directory path unchanged.
        """
        counter = 1
        directory, filename = os.path.split(file_path)
        file_base, file_extension = os.path.splitext(filename)
        new_filepath = file_path

        # Check if file exists and update the filename with a counter if needed
        while os.path.exists(new_filepath):
            new_filename = f"{file_base}({counter}){file_extension}"
            new_filepath = os.path.join(directory, new_filename)
            counter += 1

        if os.path.exists(file_path):
            rospy.logwarn(
                (
                    f'{self.__NODE_NAME}: {file_path} already exists.'
                    f'\nThe file path was changed to {new_filepath}'
                ),
            )

        return new_filepath

    # # Public methods:
    def main_loop(self):
        """
        
        """

        self.__check_initialization()

        if not self.__is_initialized:
            return

        # NOTE: Add code (function calls), which has to be executed once the
        # node was successfully initialized.
        self.__status.publish(self.__recorder_status)

    def node_shutdown(self):
        """
        
        """

        rospy.loginfo_once(f'{self.__NODE_NAME}: node is shutting down...',)

        # NOTE: Add code, which needs to be executed on nodes' shutdown here.
        # Publishing to topics is not guaranteed, use service calls or
        # set parameters instead.

        if self.__recorder_status != 'finished':
            self.write_to_csv()

            rospy.logwarn(
                f'{self.__NODE_NAME}:'
                f'\n{self.__output_file_path} recording has finished.'
            )

        rospy.loginfo_once(f'{self.__NODE_NAME}: node has shut down.',)

    def update_dataframe(self):
        """
        
        """

        topics_data = {}
        temp_data_frame = pd.DataFrame()

        topics_data['current_datetime'] = datetime.now()
        topics_data['elapsed_time'] = round(self.elapsed_time, 2)

        for i in range(len(self.__COLUMN_NAMES) - 2):
            (topics_data[self.__COLUMN_NAMES[i + 2]]) = [
                (self.__TOPIC_SUBSCRIBER_INSTANCES[i].message_data)
            ]

        new_row = pd.DataFrame(data=topics_data)

        try:
            temp_data_frame = pd.concat(
                [self.data_frame, new_row],
                ignore_index=True,
            )

        except Exception as e:
            rospy.logerr(
                (
                    f'{self.__NODE_NAME}:'
                    f' an error occured while adding a new_row to the data_frame. \n'
                    f'{e} \n'
                ),
            )

        self.data_frame = temp_data_frame
        self.elapsed_time += self.__DATA_RECORDING_PERIOD

    def write_to_csv(self):
        """
        
        """

        try:
            # Ensure the directory exists.
            directory = os.path.dirname(self.__output_file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save to a .tmp file first to avoid corrupting the original file in
            # case of a failure:
            temp_file_path = self.__output_file_path + '.tmp'
            self.data_frame.to_csv(
                temp_file_path,
                index=False,
            )

        except Exception as e:
            rospy.logerr(
                (
                    f'{self.__NODE_NAME}:'
                    f' an error occured while writing to the .tmp file. \n'
                    f'{e} \n'
                ),
            )
            return

        # Atomic operation (extremely small chance of a failure).
        try:
            os.rename(temp_file_path, self.__output_file_path)

        except Exception as e:
            rospy.logerr(
                (
                    f'{self.__NODE_NAME}:'
                    f' an error occured while renaming the .tmp file. \n'
                    f'{e} \n'
                ),
            )
            return

        rospy.loginfo(f'{self.__NODE_NAME}: data was saved.')


def main():
    """
    
    """

    # # Default node initialization.
    # This name is replaced when a launch file is used.
    rospy.init_node(
        'data_writer',
        log_level=rospy.INFO,  # rospy.DEBUG to view debug messages.
    )

    rospy.loginfo('\n\n\n\n\n')  # Add whitespaces to separate logs.

    # # ROS launch file parameters:
    node_name = rospy.get_name()

    node_frequency = rospy.get_param(
        param_name=f'{rospy.get_name()}/node_frequency',
        default=1000,
    )

    output_folder_path = rospy.get_param(
        param_name=f'{rospy.get_name()}/output_folder_path',
        # default='',
    )
    output_file_name = rospy.get_param(
        param_name=f'{rospy.get_name()}/output_file_name',
        # default='experimentname_p0_m0_t0.csv',
    )
    data_recording_period = rospy.get_param(
        param_name=f'{rospy.get_name()}/data_recording_period',
        default=2,
    )
    file_writing_period = rospy.get_param(
        param_name=f'{rospy.get_name()}/file_writing_period',
        default=10,
    )
    topics_specifiers_columns = rospy.get_param(
        param_name=f'{rospy.get_name()}/topics_specifiers_columns',
    )

    # Split the string and remove whitespaces.
    topics_specifiers_columns = [
        s for s in topics_specifiers_columns.split(' ') if s != ''
    ]

    if len(topics_specifiers_columns) % 3 != 0:
        raise ValueError(
            'The length of topics_specifiers_columns should be divisible by 3.'
        )

    topic_names = []
    topic_specifiers = []
    column_names = []

    for i in range(int(len(topics_specifiers_columns) / 3)):
        topic_names.append(topics_specifiers_columns[3 * i])

        if topics_specifiers_columns[3 * i + 1] == '.':
            topic_specifiers.append('')

        else:
            topic_specifiers.append(topics_specifiers_columns[3 * i + 1])

        column_names.append(topics_specifiers_columns[3 * i + 2])

    topic_subscriber_instances = []

    for i in range(len(topic_names)):
        topic_subscriber = TopicSubscriber(
            node_name=node_name,
            topic_name=topic_names[i],
            topic_specifier=topic_specifiers[i],
        )

        topic_subscriber_instances.append(topic_subscriber)

    # Get absolute output file path:
    sub_path = output_file_name.split('.')[0].split('_')

    experiment_name = sub_path[0]
    participant_number = sub_path[1]
    mode_number = sub_path[2]
    trial_number = sub_path[3]

    output_file_path = (
        f'{output_folder_path}'
        f'/{experiment_name}/{participant_number}/{mode_number}/{trial_number}'
        f'/{output_file_name}'
    )

    data_writer = DataWriter(
        node_name=node_name,
        output_file_path=output_file_path,
        data_recording_period=data_recording_period,
        file_writing_period=file_writing_period,
        topic_subscriber_instances=topic_subscriber_instances,
        column_names=column_names,
    )

    rospy.on_shutdown(data_writer.node_shutdown)
    node_rate = rospy.Rate(node_frequency)

    while not rospy.is_shutdown():
        data_writer.main_loop()
        node_rate.sleep()


if __name__ == '__main__':
    main()
