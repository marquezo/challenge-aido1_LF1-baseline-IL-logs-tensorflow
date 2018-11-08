#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import collections
import rosbag
import cv_bridge
import cv2
from copy import copy
from extract_data_functions import image_preprocessing, synchronize_data

# A collection of ros messages coming from a single topic.
MessageCollection = collections.namedtuple("MessageCollection", ["topic", "type", "messages"])

def extract_messages(path, requested_topics):

    # check if path is string and requested_topics a list
    assert isinstance(path, str)
    assert isinstance(requested_topics, list)

    bag = rosbag.Bag(path)

    _, available_topics = bag.get_type_and_topic_info()

    # check if the requested topics exist in bag's topics and if yes extract the messages only for them
    extracted_messages = {}
    for topic in requested_topics:
        if topic not in available_topics:
            raise ValueError("Could not find the requested topic (%s) in the bag %s" % (topic, path))
        extracted_messages[topic] = MessageCollection(topic=topic, type=available_topics[topic].msg_type, messages=[])

    for msg in bag.read_messages():
        topic = msg.topic
        if topic not in requested_topics:
            continue
        extracted_messages[topic].messages.append(msg)
    bag.close()

    return extracted_messages

def main():

    # define the list of topics that you want to extract
    ros_topics = [
                # the duckiebot name can change from one bag file to the other, so define
                # the topics WITHOUT the duckiebot name in the beginning
                "/camera_node/image/compressed",
                # "/lane_controller_node/car_cmd"
                "/wheels_driver_node/wheels_cmd"
                ]


    # define the bags_directory in order to extract the data
    bags_directory = os.path.join(os.getcwd(), "data", "bag_files")

    # define data_directory
    data_directory = 'data'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # define train and test directories inside the data directory
    test_dir = os.path.join(data_directory, "test")
    train_dir = os.path.join(data_directory, "train")
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    cvbridge_object = cv_bridge.CvBridge()

    # create a dataframe to store the data for all bag files
    # df_all = pd.DataFrame()

    first_time = True

    for file in os.listdir(bags_directory):
        if not file.endswith(".bag"):
            continue

        # extract bag_ID to include it in the data for potential future use (Useful in case of weird data distributions
        # or final results, since you will be able to associate the data with the bag files)
        bag_ID = file.partition(".bag")[0]

        # extract the duckiebot name to complete the definition of the nodes
        duckiebot_name = file.partition("_")[2].partition(".bag")[0]

        # complete the topics names with the duckiebot name in the beginning
        ros_topics_temp = copy(ros_topics)
        for num, topic in enumerate(ros_topics_temp):
            ros_topics_temp[num] = "/" + duckiebot_name + topic

        # define absolute path of the bag_file
        abs_path = os.path.abspath(os.path.join(bags_directory, file))

        print("Extract data for {} file.".format(file))
        try:
            msgs = extract_messages(abs_path, ros_topics_temp)
        except rosbag.bag.ROSBagException:
            print("Failed to open {}".format(abs_path))
            continue

                         ######## This following part is implementation specific ########

        # The composition of the ros messages is different (e.g. different names in the messages) and also different
        # tools are used to handle the different extracted data (e.g. cvbridge for images). As a result, the following
        # part of the script can be used as a basis to extract the data, but IT HAS TO BE MODIFIED based on your topics.

        # easy way to find the structure of your ros messages : print dir(msgs[name_of_topic])


        # extract the images and car_cmds messages
        ext_images = msgs["/" + duckiebot_name + "/camera_node/image/compressed"].messages
        # ext_car_cmds = msgs["/" + duckiebot_name + "/lane_controller_node/car_cmd"].messages
        ext_car_cmds = msgs["/" + duckiebot_name + "/wheels_driver_node/wheels_cmd"].messages

        # create dataframe with the images and the images' timestamps
        for num, img in enumerate(ext_images):

            # get the rgb image
            #### direct conversion to CV2 ####
            # np_arr = np.fromstring(img.data, np.uint8)
            # img = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
            # print("img", img, img.shape)
            img = cvbridge_object.compressed_imgmsg_to_cv2(img.message)
            img = image_preprocessing(img)  # -> each image is of dimensions (1, 48x96=4608)

            # hack to get the timestamp of each image in <float 'secs.nsecs'> format instead of <int 'rospy.rostime.Time'>
            temp_timestamp = ext_images[num].timestamp
            img_timestamp = temp_timestamp.secs + temp_timestamp.nsecs *10 ** -len(str(temp_timestamp.nsecs))

            temp_df = pd.DataFrame({
                'img': [img],
                'img_timestamp': [img_timestamp]
            })

            if num == 0:
                df_imgs = temp_df.copy()
            else:
                df_imgs = df_imgs.append(temp_df, ignore_index=True)


        # create dataframe with the car_cmds and the car_cmds' timestamps
        for num, cmd in enumerate(ext_car_cmds):

            # read wheel commands messages
            cmd_msg = cmd.message

            # hack to get the timestamp of each image in <float 'secs.nsecs'> format instead of <int 'rospy.rostime.Time'>
            temp_timestamp = ext_car_cmds[num].timestamp
            vel_timestamp = temp_timestamp.secs + temp_timestamp.nsecs * 10 ** -len(str(temp_timestamp.nsecs))

            temp_df = pd.DataFrame({
                'vel_timestamp': [vel_timestamp],
                'vel_left': [cmd_msg.vel_left],
                'vel_right': [cmd_msg.vel_right]
            })

            if num == 0:
                df_cmds = temp_df.copy()
            else:
                df_cmds = df_cmds.append(temp_df, ignore_index=True)

        # synchronize data
        print("Starting synchronization of data for {} file.".format(file))

        temp_synch_data, temp_synch_imgs = synchronize_data(df_imgs, df_cmds, bag_ID)

        if first_time:
            synch_data = copy(temp_synch_data)
            synch_imgs = copy(temp_synch_imgs)
            first_time = False

        else:
            synch_data = np.vstack((synch_data, temp_synch_data))
            synch_imgs = np.vstack((synch_imgs, temp_synch_imgs))
        
        print("\nShape of total data: {} , shape of total images: {}\n".format(synch_data.shape, synch_imgs.shape))

    print("Synchronization of all data is finished.\n")


    # define the names of the train and test .h5 files
    dataset_name = os.path.join(train_dir, 'dataset.npz')

    # check if these two files exist in the data directory and if yes remove them before saving the new files
    if os.path.isfile(dataset_name):
        os.remove(dataset_name)

    np.savez_compressed(dataset_name, synch_data=synch_data, synch_imgs=synch_imgs)

    print("Saved data and images to {}".format(dataset_name))

    #print("\nThe total {} data were split into {} training and {} test datasets and saved in {} "
    #      "directory.".format(synch_data.shape[0], df_data_train.shape[0], df_data_test.shape[0], data_directory))

if __name__ == "__main__":
    main()
