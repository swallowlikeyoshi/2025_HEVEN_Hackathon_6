import rospy
import numpy as np
import cv2

class BrainLogger():
    def __init__(self, brain):
        self.brain = brain
        self.db = brain.db
        cv2.namedWindow("Lidar")
        cv2.namedWindow("Global Path")

    def log_data(self):
        # Pose
        pose = self.db.pose_data
        rospy.loginfo(f"Pose: x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f}")
        rospy.loginfo(f"Current Mission: {self.db.current_mission}")

        # Traffic
        traffic = self.db.traffic_light
        remaining = self.db.traffic_remaining_time
        rospy.loginfo(f"Traffic light: {traffic}, remaining_time: {remaining}")

        # Parking
        parking_info = self.db.parking_list
        target_park = self.db.target_park
        rospy.loginfo(f"Parking: {parking_info}, target: {target_park}")

        # Visualization
        cv2.imshow("Lidar", self.visualize_lidar(self.db.lidar_data))
        cv2.imshow("Global Path", self.visualize_path(self.db.global_path, pose))
        cv2.waitKey(1)

    def visualize_lidar(self, lidar_data):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        center = (200, 200)
        max_range = 10.0
        for i, d in enumerate(lidar_data):
            if d > 0:
                angle = np.deg2rad(i) + np.pi
                x = int(center[0] + d / max_range * 180 * np.cos(angle))
                y = int(center[1] + d / max_range * 180 * np.sin(angle))
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        return img

    def visualize_path(self, path, pose):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cx, cy = 100, 300
        scale = 20
        for pt in path:
            x = int(cx + pt[0] * scale)
            y = int(cy - pt[1] * scale)
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        px = int(cx + pose[0] * scale)
        py = int(cy - pose[1] * scale)
        cv2.circle(img, (px, py), 5, (0, 0, 255), -1)
        return img