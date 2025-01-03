import collections
import enum
import time

import cv2
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from utils.sort import *


def main():

    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="bin/movenet_256.tflite")
    interpreter.allocate_tensors()

    interpreter_act = tf.lite.Interpreter(model_path="bin/act_movenet.tflite")
    interpreter_act.allocate_tensors()

    def movenet(input_image):
        """Runs detection on an input image.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the predicted keypoint
          coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])
        return keypoints_with_scores

    def act(keypoints):
        input_details = interpreter_act.get_input_details()
        output_details = interpreter_act.get_output_details()
        interpreter_act.set_tensor(
            input_details[0]["index"], keypoints.astype(np.float32)[None, ...]
        )
        interpreter_act.invoke()
        return interpreter_act.get_tensor(output_details[0]["index"])

    n = 0
    sum_process_time = 0
    sum_inference_time = 0
    n_frames = 30
    # fps_counter = avg_fps_counter(n_frames)
    d = {}
    mot_tracker = Sort(max_age=60, min_hits=5)
    inference_time = 0

    crop_region = init_crop_region(INPUT_SIZE, INPUT_SIZE)

    # gif = cv2.VideoCapture("dance.gif")
    # ret, frame = gif.read()  # ret=True if it finds a frame else False.
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image_height, image_width, _ = frame.shape

        input_image = tf.expand_dims(frame, axis=0)
        # input_image = tf.image.resize_with_pad(input_image, INPUT_SIZE, INPUT_SIZE)

        input_image = crop_and_resize(input_image, crop_region, crop_size=SRC_SIZE)
        # Run model inference.
        outputs = movenet(input_image)
        # Update the coordinates.
        for idx in range(17):
            outputs[0, 0, idx, 0] = (
                crop_region["y_min"] * image_height
                + crop_region["height"] * image_height * outputs[0, 0, idx, 0]
            ) / image_height
            outputs[0, 0, idx, 1] = (
                crop_region["x_min"] * image_width
                + crop_region["width"] * image_width * outputs[0, 0, idx, 1]
            ) / image_width

        start_time = time.monotonic()

        # Grab pose data for inference
        X_frame = []
        dets = []

        if outputs is not None:
            for pose in outputs:  # poses in image
                score = pose[:, -1].mean()
                if score > 0.3:
                    X_frame.append(pose)  # poses
                    dets.append(
                        [
                            np.min(pose[:, 0]),
                            np.min(pose[:, 1]),
                            np.max(pose[:, 0]),
                            np.max(pose[:, 1]),
                            score,
                        ]
                    )

            if X_frame:  # get pose ID from sort
                bb_ids = mot_tracker.update(np.asarray(dets))
            else:
                bb_ids = mot_tracker.update(np.empty((0, 5)))
            print(len(bb_ids))

            if bb_ids.size > 0:
                for i, idd in enumerate(bb_ids[:, 4]):
                    if int(idd) in d:
                        d[idd] = np.concatenate((d[idd], X_frame[i]), axis=0)

                        if d[idd].shape[0] == n_frames:

                            t = time.process_time()

                            pose_dist = act(preprocess(d[idd]))[0]
                            pose_dist = np.exp(pose_dist) / np.sum(np.exp(pose_dist))

                            inference_time = time.process_time() - t

                            pose_class = np.argmax(pose_dist)
                            print(
                                f"ID: {int(idd)} - Predicted Class: {CLASSES[pose_class]} {round(pose_dist[pose_class]*100,2)}%",
                            )

                            d[idd] = d[idd][1:, ...]

                    else:  # new idd
                        d[idd] = X_frame[i]

                print(d.keys())

            # clean old IDs
            for k in list(d.keys()):
                if k not in bb_ids[:, 4].astype(int):
                    # print(d)
                    del d[k]

        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time * 1000

        # avg_inference_time = sum_inference_time / n
        # text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f Nposes %d' % (
        #     avg_inference_time, 1000 / (avg_inference_time + 1e-6), next(fps_counter), len(outputs)
        # )
        # print(text_line)

        # Visualize the predictions with image.
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        display_image = tf.expand_dims(frame, axis=0)
        output_overlay = draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), outputs
        )

        # Display the captured frame
        cv2.imshow("Camera", output_overlay)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()


def _keypoints_and_edges_for_display(
    keypoints_with_scores, height, width, keypoint_threshold=0.11
):
    """Returns high confidence keypoints and edges for visualization.

    Args:
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      height: height of the image in pixels.
      width: width of the image in pixels.
      keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
      A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1
        )
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :
        ]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (
                kpts_scores[edge_pair[0]] > keypoint_threshold
                and kpts_scores[edge_pair[1]] > keypoint_threshold
            ):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image,
    keypoints_with_scores,
    crop_region=None,
    close_figure=False,
    output_image_height=None,
):
    """Draws the keypoint predictions on image.

    Args:
      image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
      keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
      crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
      output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
      A numpy array with shape [out_height, out_width, channel] representing the
      image overlaid with keypoint predictions.
    """
    height, width, _ = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis("off")

    _ = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle="solid")
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color="#FF1493", zorder=3)

    (keypoint_locs, keypoint_edges, edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width
    )

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region["x_min"] * width, 0.0)
        ymin = max(crop_region["y_min"] * height, 0.0)
        rec_width = min(crop_region["x_max"], 0.99) * width - xmin
        rec_height = min(crop_region["y_max"], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            rec_width,
            rec_height,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape((2400, -1, 4))
    plt.close(fig)
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv2.resize(
            image_from_plot,
            dsize=(output_image_width, output_image_height),
            interpolation=cv2.INTER_CUBIC,
        )
    return image_from_plot


def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)


def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(
        dwg.text(
            text,
            insert=(x + 1, y + 1),
            fill="black",
            font_size=font_size,
            style="font-family:sans-serif",
        )
    )
    dwg.add(
        dwg.text(
            text,
            insert=(x, y),
            fill="white",
            font_size=font_size,
            style="font-family:sans-serif",
        )
    )


def draw_pose(dwg, pose, src_size, color="yellow", threshold=0.1):
    box_x, box_y, box_w, box_h = (0, 0, INPUT_SIZE, INPUT_SIZE)
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold:
            continue
        # Offset and scale to source coordinate space.
        kp_x = int((keypoint.point[0] - box_x) * scale_x)
        kp_y = int((keypoint.point[1] - box_y) * scale_y)

        xys[label] = (kp_x, kp_y)
        dwg.add(
            dwg.circle(
                center=(int(kp_x), int(kp_y)),
                r=5,
                fill="cyan",
                fill_opacity=keypoint.score,
                stroke=color,
            )
        )

    for a, b in EDGES:
        if a not in xys or b not in xys:
            continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))


def draw_bbox(dwg, bbox, src_size, idd, lab="loading...", prob=0.00):
    box_x, box_y, box_w, box_h = (0, 0, INPUT_SIZE, INPUT_SIZE)
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h

    x_ul = int((bbox[0] - box_x) * scale_x)
    y_ul = int((bbox[1] - box_y) * scale_y) - 30
    x_lr = int((bbox[2] - box_x) * scale_x)
    y_lr = int((bbox[3] - box_y) * scale_y)

    bb_ins = (x_ul, y_ul)
    # print(bb_ins)
    bb_size = (x_lr - x_ul, y_lr - y_ul)
    if lab == "loading...":
        col = "yellow"
    else:
        col = "#A90690"
    dwg.add(
        dwg.rect(insert=bb_ins, size=bb_size, fill="none", stroke_width=5, stroke=col)
    )
    dwg.add(
        dwg.text(
            idd + " - " + lab + " " + str(round((prob * 100), 2)) + "%",
            insert=bb_ins,
            stroke="#500",
            fill=col,
            stroke_width=1,
            font_size="30px",
            font_weight="bold",
            font_family="Arial",
        )
    )


def torso_visible(keypoints):
    """Checks whether there are enough torso keypoints.

    This function checks whether the model is confident at predicting one of the
    shoulders/hips which is required to determine a good crop region.
    """
    return (
        keypoints[0, 0, KEYPOINT_DICT["left_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
        or keypoints[0, 0, KEYPOINT_DICT["right_hip"], 2] > MIN_CROP_KEYPOINT_SCORE
    ) and (
        keypoints[0, 0, KEYPOINT_DICT["left_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
        or keypoints[0, 0, KEYPOINT_DICT["right_shoulder"], 2] > MIN_CROP_KEYPOINT_SCORE
    )


def determine_torso_and_body_range(keypoints, target_keypoints, center_y, center_x):
    """Calculates the maximum distance from each keypoints to the center location.

    The function returns the maximum distances from the two sets of keypoints:
    full 17 keypoints and 4 torso keypoints. The returned information will be
    used to determine the crop size. See determineCropRegion for more detail.
    """
    torso_joints = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for joint in torso_joints:
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for joint in KEYPOINT_DICT.keys():
        if keypoints[0, 0, KEYPOINT_DICT[joint], 2] < MIN_CROP_KEYPOINT_SCORE:
            continue
        dist_y = abs(center_y - target_keypoints[joint][0])
        dist_x = abs(center_x - target_keypoints[joint][1])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y

        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]


def init_crop_region(image_height, image_width):
    """Defines the default crop region.

    The function provides the initial crop region (pads the full image from both
    sides to make it a square image) when the algorithm cannot reliably determine
    the crop region from the previous frame.
    """
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        "y_min": y_min,
        "x_min": x_min,
        "y_max": y_min + box_height,
        "x_max": x_min + box_width,
        "height": box_height,
        "width": box_width,
    }


def determine_crop_region(keypoints, image_height, image_width):
    """Determines the region to crop the image for the model to run inference on.

    The algorithm uses the detected joints from the previous frame to estimate
    the square region that encloses the full body of the target person and
    centers at the midpoint of two hip joints. The crop size is determined by
    the distances between each joints and the center point.
    When the model is not confident with the four torso joint predictions, the
    function returns a default crop which is the full image padded to square.
    """
    target_keypoints = {}
    for joint in KEYPOINT_DICT.keys():
        target_keypoints[joint] = [
            keypoints[0, 0, KEYPOINT_DICT[joint], 0] * image_height,
            keypoints[0, 0, KEYPOINT_DICT[joint], 1] * image_width,
        ]

    if torso_visible(keypoints):
        center_y = (
            target_keypoints["left_hip"][0] + target_keypoints["right_hip"][0]
        ) / 2
        center_x = (
            target_keypoints["left_hip"][1] + target_keypoints["right_hip"][1]
        ) / 2

        (max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange) = (
            determine_torso_and_body_range(
                keypoints, target_keypoints, center_y, center_x
            )
        )

        crop_length_half = np.amax(
            [
                max_torso_xrange * 1.9,
                max_torso_yrange * 1.9,
                max_body_yrange * 1.2,
                max_body_xrange * 1.2,
            ]
        )

        tmp = np.array(
            [center_x, image_width - center_x, center_y, image_height - center_y]
        )
        crop_length_half = np.amin([crop_length_half, np.amax(tmp)])

        crop_corner = [center_y - crop_length_half, center_x - crop_length_half]

        if crop_length_half > max(image_width, image_height) / 2:
            return init_crop_region(image_height, image_width)
        else:
            crop_length = crop_length_half * 2
            return {
                "y_min": crop_corner[0] / image_height,
                "x_min": crop_corner[1] / image_width,
                "y_max": (crop_corner[0] + crop_length) / image_height,
                "x_max": (crop_corner[1] + crop_length) / image_width,
                "height": (crop_corner[0] + crop_length) / image_height
                - crop_corner[0] / image_height,
                "width": (crop_corner[1] + crop_length) / image_width
                - crop_corner[1] / image_width,
            }
    else:
        return init_crop_region(image_height, image_width)


def crop_and_resize(image, crop_region, crop_size):
    """Crops and resize the image to prepare for the model input."""
    boxes = [
        [
            crop_region["y_min"],
            crop_region["x_min"],
            crop_region["y_max"],
            crop_region["x_max"],
        ]
    ]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size
    )
    return output_image


C1 = 1
C2 = 2
M1 = 7
M2 = 8
H = [0, 1, 2, 3, 4]
RF = [15]
LF = [16]

INPUT_SIZE = 256
SRC_SIZE = (INPUT_SIZE, INPUT_SIZE)

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}


class KeypointType(enum.IntEnum):
    """Pose kepoints."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.NOSE, KeypointType.LEFT_EAR),
    (KeypointType.NOSE, KeypointType.RIGHT_EAR),
    (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
)

CLASSES = [
    "standing",
    "check watch",
    "cross arms",
    "scratch head",
    "sit down",
    "get up",
    "turn around",
    "walking",
    "wave1",
    "boxing",
    "kicking",
    "pointing",
    "pick up",
    "bending",
    "hands clapping",
    "wave2",
    "jogging",
    "jumping",
    "pjump",
    "running",
]

MIN_CROP_KEYPOINT_SCORE = 0.1


def remove_confidence(X):
    Xr = X[..., :-1]
    return Xr


def invert_xy(X):
    Xi = X[..., ::-1]
    return Xi


def reduce_keypoints(X):
    to_prune = []
    for group in [H, RF, LF]:
        if len(group) > 1:
            to_prune.append(group[1:])
    to_prune = [item for sublist in to_prune for item in sublist]
    X[:, H[0], :] = np.true_divide(X[:, H].sum(1), (X[:, H] != 0).sum(1) + 1e-9)
    X[:, RF[0], :] = np.true_divide(X[:, RF].sum(1), (X[:, RF] != 0).sum(1) + 1e-9)
    X[:, LF[0], :] = np.true_divide(X[:, LF].sum(1), (X[:, LF] != 0).sum(1) + 1e-9)
    Xr = np.delete(X, to_prune, 1)
    return Xr


def add_velocity(X):
    T, K, C = X.shape
    v1, v2 = np.zeros((T + 1, K, C)), np.zeros((T + 1, K, C))
    v1[1:] = X
    v2[:T] = X
    vel = (v2 - v1)[:-1]
    Xv = np.concatenate((X, vel), axis=-1)
    return Xv


def scale_and_center(X):
    pose_list = []
    for pose in X:
        zero_point = (pose[C1, :2] + pose[C2, :2]) / 2
        module_keypoint = (pose[M1, :2] + pose[M2, :2]) / 2
        scale_mag = np.linalg.norm(zero_point - module_keypoint)
        if scale_mag < 1:
            scale_mag = 1
        pose[:, :2] = (pose[:, :2] - zero_point) / scale_mag
        pose_list.append(pose)
    Xn = np.stack(pose_list)
    return Xn


def flatten_features(X):
    Xf = X.reshape(X.shape[0], -1)
    return Xf


def preprocess(X):
    X = remove_confidence(X)
    X = invert_xy(X)
    X = reduce_keypoints(X)
    X = add_velocity(X)
    X = scale_and_center(X)
    X = flatten_features(X)
    return X


if __name__ == "__main__":
    main()
