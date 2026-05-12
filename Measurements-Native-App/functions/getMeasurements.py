import json

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from firebase_functions import https_fn
from firebase_functions.options import MemoryOption

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

KNOWN_OBJECT_WIDTH_CM = 21.0
FOCAL_LENGTH = 600
DEFAULT_HEIGHT_CM = 152.0


_depth_model = None
_holistic = None


def load_depth_model():
	model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
	model.eval()
	return model


def get_depth_model():
	global _depth_model
	if _depth_model is None:
		_depth_model = load_depth_model()
	return _depth_model


def get_holistic():
	global _holistic
	if _holistic is None:
		_holistic = mp_holistic.Holistic()
	return _holistic


def calibrate_focal_length(image, real_width_cm, detected_width_px):
	return (detected_width_px * FOCAL_LENGTH) / real_width_cm if detected_width_px else FOCAL_LENGTH


def detect_reference_object(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150)
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		largest_contour = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(largest_contour)
		focal_length = calibrate_focal_length(image, KNOWN_OBJECT_WIDTH_CM, w)
		scale_factor = KNOWN_OBJECT_WIDTH_CM / w
		return scale_factor, focal_length
	return 0.05, FOCAL_LENGTH


def _resize_for_depth(image, max_dim):
	if not max_dim:
		return image

	height, width = image.shape[:2]
	long_edge = max(height, width)
	if long_edge <= max_dim:
		return image

	scale = max_dim / long_edge
	new_width = max(1, int(width * scale))
	new_height = max(1, int(height * scale))
	return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def estimate_depth(image, max_dim=None):
	resized = _resize_for_depth(image, max_dim)
	input_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
	input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
	input_tensor = F.interpolate(input_tensor, size=(384, 384), mode="bilinear", align_corners=False)

	depth_model = get_depth_model()
	with torch.no_grad():
		depth_map = depth_model(input_tensor)

	return depth_map.squeeze().numpy()


def calculate_distance_using_height(landmarks, image_height, user_height_cm):
	top_head = landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height
	bottom_foot = max(
		landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
		landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
	) * image_height

	person_height_px = abs(bottom_foot - top_head)
	distance = (user_height_cm * FOCAL_LENGTH) / person_height_px
	scale_factor = user_height_cm / person_height_px

	return distance, scale_factor


def get_body_width_at_height(frame, height_px, center_x):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	_, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

	if height_px >= frame.shape[0]:
		height_px = frame.shape[0] - 1

	horizontal_line = thresh[height_px, :]
	center_x = int(center_x * frame.shape[1])
	left_edge, right_edge = center_x, center_x

	for i in range(center_x, 0, -1):
		if horizontal_line[i] == 0:
			left_edge = i
			break

	for i in range(center_x, len(horizontal_line)):
		if horizontal_line[i] == 0:
			right_edge = i
			break

	width_px = right_edge - left_edge
	min_width = 0.1 * frame.shape[1]
	if width_px < min_width:
		width_px = min_width

	return width_px


def calculate_measurements(
	results,
	scale_factor,
	image_width,
	image_height,
	depth_map,
	frame=None,
	user_height_cm=None,
):
	landmarks = results.pose_landmarks.landmark

	if user_height_cm:
		_, scale_factor = calculate_distance_using_height(landmarks, image_height, user_height_cm)

	scale_y = 384 / image_height if depth_map is not None else None
	scale_x = 384 / image_width if depth_map is not None else None

	def pixel_to_cm(value):
		return round(value * scale_factor, 2)

	def calculate_circumference(width_px, depth_ratio=1.0):
		width_cm = width_px * scale_factor
		estimated_depth_cm = width_cm * depth_ratio * 0.7
		half_width = width_cm / 2
		half_depth = estimated_depth_cm / 2
		return round(2 * np.pi * np.sqrt((half_width**2 + half_depth**2) / 2), 2)

	measurements = {}

	left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
	right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
	shoulder_width_px = abs(left_shoulder.x * image_width - right_shoulder.x * image_width)
	shoulder_width_px *= 1.1
	measurements["shoulder_width"] = pixel_to_cm(shoulder_width_px)

	chest_y_ratio = 0.15
	chest_y = left_shoulder.y + (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y - left_shoulder.y) * chest_y_ratio

	chest_correction = 1.15
	chest_width_px = abs((right_shoulder.x - left_shoulder.x) * image_width) * chest_correction

	if frame is not None:
		chest_y_px = int(chest_y * image_height)
		center_x = (left_shoulder.x + right_shoulder.x) / 2
		detected_width = get_body_width_at_height(frame, chest_y_px, center_x)
		if detected_width > 0:
			chest_width_px = max(chest_width_px, detected_width)

	chest_depth_ratio = 1.0
	if depth_map is not None:
		chest_x = int(((left_shoulder.x + right_shoulder.x) / 2) * image_width)
		chest_y_px = int(chest_y * image_height)
		chest_y_scaled = int(chest_y_px * scale_y)
		chest_x_scaled = int(chest_x * scale_x)
		if 0 <= chest_y_scaled < 384 and 0 <= chest_x_scaled < 384:
			chest_depth = depth_map[chest_y_scaled, chest_x_scaled]
			max_depth = float(np.max(depth_map))
			if max_depth > 0:
				chest_depth_ratio = 1.0 + 0.5 * (1.0 - chest_depth / max_depth)

	measurements["chest_width"] = pixel_to_cm(chest_width_px)
	measurements["chest_circumference"] = calculate_circumference(chest_width_px, chest_depth_ratio)

	left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
	right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

	waist_y_ratio = 0.35
	waist_y = left_shoulder.y + (left_hip.y - left_shoulder.y) * waist_y_ratio

	if frame is not None:
		waist_y_px = int(waist_y * image_height)
		center_x = (left_hip.x + right_hip.x) / 2
		detected_width = get_body_width_at_height(frame, waist_y_px, center_x)
		if detected_width > 0:
			waist_width_px = detected_width
		else:
			waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9
	else:
		waist_width_px = abs(right_hip.x - left_hip.x) * image_width * 0.9

	waist_width_px *= 1.16

	waist_depth_ratio = 1.0
	if depth_map is not None:
		waist_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
		waist_y_px = int(waist_y * image_height)
		waist_y_scaled = int(waist_y_px * scale_y)
		waist_x_scaled = int(waist_x * scale_x)
		if 0 <= waist_y_scaled < 384 and 0 <= waist_x_scaled < 384:
			waist_depth = depth_map[waist_y_scaled, waist_x_scaled]
			max_depth = float(np.max(depth_map))
			if max_depth > 0:
				waist_depth_ratio = 1.0 + 0.5 * (1.0 - waist_depth / max_depth)

	measurements["waist_width"] = pixel_to_cm(waist_width_px)
	measurements["waist"] = calculate_circumference(waist_width_px, waist_depth_ratio)

	hip_width_px = abs(left_hip.x * image_width - right_hip.x * image_width) * 1.35

	if frame is not None:
		hip_y_offset = 0.1
		hip_y = left_hip.y + (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y - left_hip.y) * hip_y_offset
		hip_y_px = int(hip_y * image_height)
		center_x = (left_hip.x + right_hip.x) / 2
		detected_width = get_body_width_at_height(frame, hip_y_px, center_x)
		if detected_width > 0:
			hip_width_px = max(hip_width_px, detected_width)

	hip_depth_ratio = 1.0
	if depth_map is not None:
		hip_x = int(((left_hip.x + right_hip.x) / 2) * image_width)
		hip_y_px = int(left_hip.y * image_height)
		hip_y_scaled = int(hip_y_px * scale_y)
		hip_x_scaled = int(hip_x * scale_x)
		if 0 <= hip_y_scaled < 384 and 0 <= hip_x_scaled < 384:
			hip_depth = depth_map[hip_y_scaled, hip_x_scaled]
			max_depth = float(np.max(depth_map))
			if max_depth > 0:
				hip_depth_ratio = 1.0 + 0.5 * (1.0 - hip_depth / max_depth)

	measurements["hip_width"] = pixel_to_cm(hip_width_px)
	measurements["hip"] = calculate_circumference(hip_width_px, hip_depth_ratio)

	neck = landmarks[mp_pose.PoseLandmark.NOSE.value]
	left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
	neck_width_px = abs(neck.x * image_width - left_ear.x * image_width) * 2.0
	measurements["neck"] = calculate_circumference(neck_width_px, 1.0)
	measurements["neck_width"] = pixel_to_cm(neck_width_px)

	left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
	sleeve_length_px = abs(left_shoulder.y * image_height - left_wrist.y * image_height)
	measurements["arm_length"] = pixel_to_cm(sleeve_length_px)

	shirt_length_px = abs(left_shoulder.y * image_height - left_hip.y * image_height) * 1.2
	measurements["shirt_length"] = pixel_to_cm(shirt_length_px)

	thigh_y_ratio = 0.2
	left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
	thigh_y = left_hip.y + (left_knee.y - left_hip.y) * thigh_y_ratio

	thigh_width_px = hip_width_px * 0.5 * 1.2

	if frame is not None:
		thigh_y_px = int(thigh_y * image_height)
		thigh_x = left_hip.x * 0.9
		detected_width = get_body_width_at_height(frame, thigh_y_px, thigh_x)
		if 0 < detected_width < hip_width_px:
			thigh_width_px = detected_width

	thigh_depth_ratio = 1.0
	if depth_map is not None:
		thigh_x = int(left_hip.x * image_width)
		thigh_y_px = int(thigh_y * image_height)
		thigh_y_scaled = int(thigh_y_px * scale_y)
		thigh_x_scaled = int(thigh_x * scale_x)
		if 0 <= thigh_y_scaled < 384 and 0 <= thigh_x_scaled < 384:
			thigh_depth = depth_map[thigh_y_scaled, thigh_x_scaled]
			max_depth = float(np.max(depth_map))
			if max_depth > 0:
				thigh_depth_ratio = 1.0 + 0.5 * (1.0 - thigh_depth / max_depth)

	measurements["thigh"] = pixel_to_cm(thigh_width_px)
	measurements["thigh_circumference"] = calculate_circumference(thigh_width_px, thigh_depth_ratio)

	left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
	trouser_length_px = abs(left_hip.y * image_height - left_ankle.y * image_height)
	measurements["trouser_length"] = pixel_to_cm(trouser_length_px)

	return measurements


def validate_front_image(image_np):
	try:
		rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
		image_height, image_width = image_np.shape[:2]

		with mp_holistic.Holistic(
			static_image_mode=True,
			model_complexity=1,
			enable_segmentation=False,
			refine_face_landmarks=False,
		) as holistic_instance:
			results = holistic_instance.process(rgb_frame)

		if not hasattr(results, "pose_landmarks") or not results.pose_landmarks:
			return False, "No person detected. Please make sure you're clearly visible in the frame."

		minimum_landmarks = [
			mp_holistic.PoseLandmark.NOSE,
			mp_holistic.PoseLandmark.LEFT_SHOULDER,
			mp_holistic.PoseLandmark.RIGHT_SHOULDER,
			mp_holistic.PoseLandmark.LEFT_ELBOW,
			mp_holistic.PoseLandmark.RIGHT_ELBOW,
			mp_holistic.PoseLandmark.RIGHT_KNEE,
			mp_holistic.PoseLandmark.LEFT_KNEE,
		]

		missing_upper = []
		for landmark in minimum_landmarks:
			landmark_data = results.pose_landmarks.landmark[landmark]
			if (
				landmark_data.visibility < 0.5
				or landmark_data.x < 0
				or landmark_data.x > 1
				or landmark_data.y < 0
				or landmark_data.y > 1
			):
				missing_upper.append(landmark.name.replace("_", " "))

		if missing_upper:
			return False, "Couldn't detect full body. Please make sure your full body is visible."

		nose = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
		left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
		right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

		shoulder_width = abs(left_shoulder.x - right_shoulder.x) * image_width
		head_to_shoulder = abs(left_shoulder.y - nose.y) * image_height

		if shoulder_width < head_to_shoulder * 1.2:
			return False, "Please step back to show more of your upper body, not just your face."

		return True, "Validation passed - proceeding with measurements"
	except Exception as exc:
		print(f"Error validating body image: {exc}")
		return False, "You arent providing images correctly. Please try again."


def _cors_headers():
	return {
		"Access-Control-Allow-Origin": "*",
		"Access-Control-Allow-Methods": "POST, OPTIONS",
		"Access-Control-Allow-Headers": "Content-Type",
	}


def _json_response(payload, status=200):
	return https_fn.Response(
		json.dumps(payload),
		status=status,
		mimetype="application/json",
		headers=_cors_headers(),
	)


@https_fn.on_request(timeout_sec=300)
def get_measurements(req: https_fn.Request) -> https_fn.Response:
	if req.method == "OPTIONS":
		return https_fn.Response("", status=204, headers=_cors_headers())

	print(
		json.dumps(
			{
				"event": "request_received",
				"method": req.method,
				"content_type": req.content_type,
				"content_length": req.content_length,
				"files": list(req.files.keys()),
				"form_fields": list(req.form.keys()),
			},
			default=str,
		)
	)

	try:
		if req.method != "POST":
			return _json_response({"error": "Use POST with multipart/form-data."}, status=405)

		if "front" not in req.files:
			return _json_response(
				{
					"error": "Missing front image for reference.",
					"debug": {
						"files": list(req.files.keys()),
						"form_fields": list(req.form.keys()),
						"content_type": req.content_type,
					},
				},
				status=400,
			)

		front_image_file = req.files["front"]
		front_image_np = np.frombuffer(front_image_file.read(), np.uint8)
		front_image_file.seek(0)

		front_frame = cv2.imdecode(front_image_np, cv2.IMREAD_COLOR)
		if front_frame is None:
			return _json_response(
				{
					"error": "Unable to decode front image.",
					"debug": {
						"front_bytes": len(front_image_np),
						"content_type": req.content_type,
					},
				},
				status=400,
			)

		is_valid, error_msg = validate_front_image(front_frame)
		if not is_valid:
			return _json_response(
				{
					"error": error_msg,
					"pose": "front",
					"code": "INVALID_POSE",
				},
				status=400,
			)

		user_height_cm = req.form.get("height_cm")
		use_depth_raw = (req.form.get("use_depth") or "1").strip().lower()
		depth_max_dim_raw = (req.form.get("depth_max_dim") or "").strip()
		depth_max_dim = None
		if depth_max_dim_raw:
			try:
				depth_max_dim = int(depth_max_dim_raw)
			except ValueError:
				depth_max_dim = None
		use_depth = use_depth_raw not in {"0", "false", "no", "off"}
		if depth_max_dim is not None and depth_max_dim <= 0:
			depth_max_dim = None
		if user_height_cm:
			try:
				user_height_cm = float(user_height_cm)
			except ValueError:
				user_height_cm = DEFAULT_HEIGHT_CM
		else:
			user_height_cm = DEFAULT_HEIGHT_CM

		received_images = {
			pose_name: req.files[pose_name]
			for pose_name in ["front", "left_side"]
			if pose_name in req.files
		}
		measurements = {}
		scale_factor = None
		focal_length = FOCAL_LENGTH
		results = {}
		frames = {}

		for pose_name, image_file in received_images.items():
			image_np = np.frombuffer(image_file.read(), np.uint8)
			frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
			if frame is None:
				return _json_response(
					{"error": f"Unable to decode {pose_name} image."},
					status=400,
				)
			frames[pose_name] = frame
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			results[pose_name] = get_holistic().process(rgb_frame)
			image_height, image_width, _ = frame.shape

			if pose_name == "front":
				if results[pose_name].pose_landmarks:
					_, scale_factor = calculate_distance_using_height(
						results[pose_name].pose_landmarks.landmark,
						image_height,
						user_height_cm,
					)
				else:
					scale_factor, focal_length = detect_reference_object(frame)

			depth_map = (
				estimate_depth(frame, depth_max_dim)
				if use_depth and pose_name in ["front", "left_side"]
				else None
			)

			if results[pose_name].pose_landmarks and pose_name == "front":
				measurements.update(
					calculate_measurements(
						results[pose_name],
						scale_factor,
						image_width,
						image_height,
						depth_map,
						frames[pose_name],
						user_height_cm,
					)
				)

		debug_info = {
			"scale_factor": float(scale_factor) if scale_factor else None,
			"focal_length": float(focal_length),
			"user_height_cm": float(user_height_cm),
		}

		return _json_response({"measurements": measurements, "debug_info": debug_info})
	except Exception as exc:
		print(f"Unhandled error: {exc}")
		return _json_response(
			{
				"error": "Internal server error while processing measurements.",
				"detail": str(exc),
			},
			status=500,
		)
