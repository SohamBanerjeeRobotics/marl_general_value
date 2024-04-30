import cv2
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import argparse
from cv_bridge import CvBridge
from pathlib import Path
import csv


def stamp_to_nanosec(stamp):
    return int(stamp.sec * 1e9) + stamp.nanosec


def load_bag(rosbag_path, dataset_path):
    cvbridge = CvBridge()
    storage_options = StorageOptions(uri=rosbag_path, storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = reader.get_all_topics_and_types()
    # Create a map for quicker lookup
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    topic_export_meta = {}
    for topic_name in type_map.keys():
        path = Path(dataset_path) / topic_name.lstrip("/")
        path.mkdir(parents=True, exist_ok=True)
        topic_export_meta[topic_name] = {
            "path": path,
            "video": None,
            "img_t": None,
            "pos": None,
            "tf": None,
            "lidar": None,
            "wheel_enc": None,
            "rl_tuples" : None,
            "index": 0,
        }

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg_type = get_message(type_map[topic])
        msg = deserialize_message(data, msg_type)
        if False and type_map[topic] == "freyja_msgs/msg/CurrentState":
            meta = topic_export_meta[topic]
            meta["index"] += 1
            if meta["pos"] is None:
                meta["pos"] = []
            meta["pos"].append(
                {
                    "t_rec": t,
                    "t_msg": stamp_to_nanosec(msg.header.stamp),
                    "pn": msg.state_vector[0],
                    "pe": msg.state_vector[1],
                    #"pd": msg.state_vector[2],
                    "vn": msg.state_vector[3],
                    "ve": msg.state_vector[4],
                    "yaw": msg.state_vector[8],
                }
            )
        elif (
            type_map[topic] == "sensor_msgs/msg/Image"
            or type_map[topic] == "sensor_msgs/msg/CompressedImage"
        ):
            if type_map[topic] == "sensor_msgs/msg/CompressedImage":
                img = cvbridge.compressed_imgmsg_to_cv2(msg)
            else:
                img = cvbridge.imgmsg_to_cv2(msg, "bgr8")
            meta = topic_export_meta[topic]
            if meta["img_t"] is None:
                meta["img_t"] = []
            if meta["video"] is None:
                meta["video"] = cv2.VideoWriter(
                    str((meta["path"].parent / f"{meta['path'].name}.mp4").resolve()),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    15,  # FPS
                    (img.shape[1], img.shape[0]),
                )
            filename = f"{meta['index']:05d}.jpg"
            meta["img_t"].append(
                {
                    "t_rec": t,
                    "t_msg": stamp_to_nanosec(msg.header.stamp),
                    "file": filename,
                }
            )
            meta["video"].write(img)
            cv2.imwrite(str((meta["path"] / filename).resolve()), img)
            meta["index"] += 1

        elif type_map[topic] == "sensor_msgs/msg/LaserScan":
            meta = topic_export_meta[topic]
            if meta["lidar"] is None:
                meta["lidar"] = []
            meta["index"] += 1
            meta["lidar"].append(
                {
                    "t_rec": t,
                    "t_msg": stamp_to_nanosec(msg.header.stamp),
                    "ranges": list(msg.ranges),
                }
            )
        elif type_map[topic] == "robomaster_msgs/msg/WheelSpeed":
            meta = topic_export_meta[topic]
            if meta["wheel_enc"] is None:
                meta["wheel_enc"] = []
            meta["index"] += 1
            meta["wheel_enc"].append(
                {
                    "t_rec": t,
                    "t_msg": stamp_to_nanosec(msg.header.stamp),
                    "fl": msg.fl,
                    "fr": msg.fr,
                    "rl": msg.rl,
                    "rr": msg.rr,
                }
            )
        elif type_map[topic] == "freyja_msgs/msg/RLStatesActions":
            meta = topic_export_meta[topic]
            meta["index"] += 1
            if meta["rl_tuples"] is None:
                meta["rl_tuples"] = []
            meta["rl_tuples"].append(
                {
                    "t_rec": t,
                    "t_msg": stamp_to_nanosec(msg.header.stamp),
                    "prev_state.pn":   msg.prev_state_pos.x,
                    "prev_state.pe":   msg.prev_state_pos.y,
                    "prev_state.yaw":  msg.prev_state_pos.z,
                    "prev_state.vn" :  msg.prev_state_vel.x,
                    "prev_state.ve" :  msg.prev_state_vel.y,
                    "prev_state.yawr": msg.prev_state_vel.z,
                    #"pd": msg.state_vector[2],
                    "curr_state.pn":   msg.curr_state_pos.x,
                    "curr_state.pe":   msg.curr_state_pos.y,
                    "curr_state.yaw":  msg.curr_state_pos.z,
                    "curr_state.vn":   msg.curr_state_vel.x,
                    "curr_state.ve":   msg.curr_state_vel.y,
                    "curr_state.yawr": msg.curr_state_vel.z,
                    "prev_action.n" :  msg.prev_action.x,
                    "prev_action.e" :  msg.prev_action.y,
                    "prev_action.yaw": msg.prev_action.z,
                }
            )
        else:
            continue

    for meta in topic_export_meta.values():
        #print(meta)
        if meta["video"] is not None:
            meta["video"].release()
        if meta["pos"] is not None:
            with open(meta["path"] / "pos.csv", "w", newline="") as csvfile:
                fieldnames = ["t_rec", "t_msg", "pn", "pe", "vn", "ve", "yaw"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(meta["pos"])
        if meta["img_t"] is not None:
            with open(meta["path"].parent / "t.csv", "w", newline="") as csvfile:
                fieldnames = ["t_rec", "t_msg", "file"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(meta["img_t"])
        if meta["lidar"] is not None:
            with open(meta["path"] / "lidar.csv", "w", newline="") as csvfile:
                fieldnames = ["t_rec", "t_msg", "ranges"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(meta["lidar"])
        if meta["wheel_enc"] is not None:
            with open(meta["path"] / "enc.csv", "w", newline="") as csvfile:
                fieldnames = ["t_rec", "t_msg", "fl", "fr", "rl", "rr"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(meta["wheel_enc"])
        if meta["rl_tuples"] is not None:
            with open(meta["path"] / "rl_tuples.csv", "w", newline="") as csvfile:
                fieldnames = ["t_rec", "t_msg", "prev_state.pn", "prev_state.pe", "prev_state.yaw", "prev_state.vn", "prev_state.ve", "prev_state.yawr", "curr_state.pn", "curr_state.pe", "curr_state.yaw", "curr_state.vn", "curr_state.ve", "curr_state.yawr", "prev_action.n", "prev_action.e", "prev_action.yaw"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(meta["rl_tuples"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag_path")
    parser.add_argument("--dataset_path")
    parser.add_argument("process_whole_folder", action="store_true")

    args = parser.parse_args()

    """
    if args.process_whole_folder:
        root_rosbag_path = Path(args.rosbag_path)
        for path_rosbag in root_rosbag_path.glob("*"):
            path_rosbag_data = os.path.join(args.rosbag_path, path_rosbag.name)
            path_dataset = os.path.join(args.dataset_path, path_rosbag.name)
            print(
                f"Load Rosbag from:\n\t\t{path_rosbag_data}\nSave into: \n\t\t{path_dataset}"
            )
            if not os.path.exists(path_dataset):
                os.mkdir(path_dataset)
            load_bag(path_rosbag_data, path_dataset)
    else:
    """
    load_bag(args.rosbag_path, args.dataset_path)


if __name__ == "__main__":
    main()
