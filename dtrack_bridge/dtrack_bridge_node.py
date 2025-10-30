#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import re
import socket
import threading
from typing import Dict, Any, List, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster

# ---------- Mathe-Helfer ----------

def mat_to_quaternion(R: List[List[float]]) -> Tuple[float, float, float, float]:
    m00,m01,m02 = R[0]; m10,m11,m12 = R[1]; m20,m21,m22 = R[2]
    t = m00 + m11 + m22
    if t > 0.0:
        S = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / S; x = 0.25 * S; y = (m01 + m10) / S; z = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / S; x = (m01 + m10) / S; y = 0.25 * S; z = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / S; x = (m02 + m20) / S; y = (m12 + m21) / S; z = 0.25 * S
    n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
    return (w/n, x/n, y/n, z/n)

def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(roll/2.0); sr = math.sin(roll/2.0)
    cp = math.cos(pitch/2.0); sp = math.sin(pitch/2.0)
    cy = math.cos(yaw/2.0); sy = math.sin(yaw/2.0)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return (w, x, y, z)

def quat_multiply(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    )

# ---------- Parser (passt zu deinem Sniffer-Beispiel) ----------

def parse_dtrack_packet(data: bytes) -> Dict[str, Any]:
    text = data.decode("ascii", errors="ignore")
    lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]

    fr = ts = None
    sixd_line = None
    for ln in lines:
        if ln.startswith("fr "):
            try: fr = int(ln.split()[1])
            except: pass
        elif ln.startswith("ts ") or ln.startswith("nts "):
            try: ts = float(ln.split()[1])
            except: pass
        elif ln.startswith("6d "):
            sixd_line = ln

    if sixd_line is None:
        m = re.search(r"6d\s+\d+.*", text, flags=re.S)
        sixd_line = m.group(0).strip() if m else None

    result = {"frame": fr, "timestamp": ts, "count": None, "objects": []}
    if not sixd_line:
        return result

    parts = sixd_line.split(maxsplit=2)
    try:
        count = int(parts[1])
    except:
        return result
    result["count"] = count
    if count == 0 or len(parts) < 3:
        return result

    groups = re.findall(r"\[([^\]]+)\]", parts[2])
    idx = 0
    for _ in range(count):
        if idx + 2 >= len(groups):
            break
        id_q = groups[idx].split();            idx += 1
        posabc = groups[idx].split();          idx += 1
        Rvals = [float(v) for v in groups[idx].split()]; idx += 1
        if len(id_q) < 2 or len(posabc) < 6 or len(Rvals) < 9:
            continue
        x,y,z,a,b,c = map(float, posabc[:6])
        R = [[Rvals[0], Rvals[1], Rvals[2]],
             [Rvals[3], Rvals[4], Rvals[5]],
             [Rvals[6], Rvals[7], Rvals[8]]]
        result["objects"].append({
            "id": int(id_q[0]),
            "quality": float(id_q[1]),
            "x": x, "y": y, "z": z,
            "a": a, "b": b, "c": c,
            "R": R,
        })
    return result

# ---------- ROS-Node ----------

class DTrackBridge(Node):
    def __init__(self):
        super().__init__("dtrack_bridge")

        # Parameter
        self.declare_parameter("bind_address", "0.0.0.0")
        self.declare_parameter("port", 2222)
        self.declare_parameter("topic", "/mocap/pose")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("object_id", 1)      # welcher Marker/Body
        self.declare_parameter("position_scale", 0.001)  # mm -> m
        # optionaler Ausrichtungs-Offset (Grad), falls DTrack-Achsen nicht mit REP-103 übereinstimmen
        self.declare_parameter("rot_correction_rpy_deg", [0.0, 0.0, 0.0])
        self.declare_parameter("translation_offset", [0.0, 0.0, 0.0])

        bind = self.get_parameter("bind_address").value
        port = int(self.get_parameter("port").value)
        topic = self.get_parameter("topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.object_id = int(self.get_parameter("object_id").value)
        self.scale = float(self.get_parameter("position_scale").value)
        rpy_corr_deg = self.get_parameter("rot_correction_rpy_deg").value
        self.rot_corr = quat_from_rpy(*(math.radians(x) for x in rpy_corr_deg))
        self.t_offset = self.get_parameter("translation_offset").value

        self.pose_pub = self.create_publisher(PoseStamped, topic, 10)
        self.tf_br = TransformBroadcaster(self) if self.publish_tf else None

        # UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.get_logger().info(f"Listening on {bind}:{port}/udp")

        # separater Lese-Thread (blockierendes recvfrom)
        self._run = True
        self.thread = threading.Thread(target=self.reader_loop, daemon=True)
        self.thread.start()

    def reader_loop(self):
        while rclpy.ok() and self._run:
            try:
                data, _ = self.sock.recvfrom(8192)
            except Exception as e:
                self.get_logger().warn(f"recvfrom error: {e}")
                continue

            parsed = parse_dtrack_packet(data)
            if not parsed.get("objects"):
                continue
            # wähle gewünschte ID (oder nimm das erste Objekt)
            obj = None
            for o in parsed["objects"]:
                if o["id"] == self.object_id:
                    obj = o; break
            if obj is None:
                obj = parsed["objects"][0]

            # Position (mm -> m) + Offset
            x = obj["x"] * self.scale + self.t_offset[0]
            y = obj["y"] * self.scale + self.t_offset[1]
            z = obj["z"] * self.scale + self.t_offset[2]

            # Orientierung aus 3x3 -> Quaternion und ggf. Korrektur multiplizieren
            q = mat_to_quaternion(obj["R"])
            # erst Korrektur * dann Messung (linke Multiplikation = fester Welt/Frame-Offset)
            q = quat_multiply(self.rot_corr, q)

            now = self.get_clock().now().to_msg()

            # Pose publishen
            msg = PoseStamped()
            msg.header.stamp = now
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            msg.pose.orientation.w = q[0]
            msg.pose.orientation.x = q[1]
            msg.pose.orientation.y = q[2]
            msg.pose.orientation.z = q[3]
            self.pose_pub.publish(msg)

            # optional TF map->base_link
            if self.tf_br is not None:
                t = TransformStamped()
                t.header.stamp = now
                t.header.frame_id = self.frame_id
                t.child_frame_id = self.child_frame_id
                t.transform.translation.x = x
                t.transform.translation.y = y
                t.transform.translation.z = z
                t.transform.rotation.w = q[0]
                t.transform.rotation.x = q[1]
                t.transform.rotation.y = q[2]
                t.transform.rotation.z = q[3]
                self.tf_br.sendTransform(t)

    def destroy_node(self):
        self._run = False
        try:
            self.sock.close()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = DTrackBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
