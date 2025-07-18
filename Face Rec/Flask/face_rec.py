import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from multiprocessing import Pool

def process_video(video_path, db_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # 1 frame per second
    frames_to_process = []
    frame_index = 0
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % frame_interval == 0:
            # Detect faces in the frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(rgb_frame, enforce_detection=False, detector_backend="opencv")
            if faces:  # Only process frames with faces
                frames_to_process.append((frame, frame_index, db_path, output_dir))
                frame_index += 1
        current_frame += 1
    cap.release()

    with Pool(processes=4) as pool:
        results = pool.map(process_frame_wrapper, frames_to_process)

    all_actors = set()
    for actors_in_frame in results:
        all_actors.update(actors_in_frame)
    return frame_index, list(all_actors)

def process_frame_wrapper(args):
    return process_frame(*args)

def process_frame(frame, frame_index, db_path, output_dir):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = DeepFace.extract_faces(rgb_frame, enforce_detection=False, detector_backend="opencv")

    detected_faces = []
    face_bounding_boxes = []

    for face in faces:
        face_array = (np.array(face["face"]) * 255).astype(np.uint8)
        detected_faces.append(face_array)
        face_bounding_boxes.append(face["facial_area"])

    results = []
    for i, face_img in enumerate(detected_faces):
        temp_path = f"temp_face_{frame_index}_{i}.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
        df = DeepFace.find(
            img_path=temp_path,
            db_path=db_path,
            enforce_detection=False,
            model_name="Facenet512",
            detector_backend="opencv",
            distance_metric="euclidean_l2",
            silent=True
        )
        if len(df) > 0:
            results.append(df[0])
        os.remove(temp_path)

    df_results = pd.concat(results) if results else pd.DataFrame()

    # Confidence threshold for more accurate matches
    CONFIDENCE_THRESHOLD = 0.8  # Lower means more confident

    if not df_results.empty and "distance" in df_results:
        best_matches = df_results[df_results["distance"] < CONFIDENCE_THRESHOLD]
    else:
        best_matches = pd.DataFrame()

    identity_map = {}
    for identity_path, distance in zip(best_matches.get("identity", []), best_matches.get("distance", [])):
        person_name = os.path.basename(os.path.dirname(identity_path))
        if person_name not in identity_map.values():
            identity_map[identity_path] = person_name

    # Draw boxes and names
    for box, _ in zip(face_bounding_boxes, detected_faces):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        matched_identity = next(iter(identity_map.values()), None)
        if matched_identity:
            cv2.putText(frame, matched_identity, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save frames with detected faces in a separate folder
    detected_faces_dir = os.path.join(output_dir, "detected_faces_frames")
    os.makedirs(detected_faces_dir, exist_ok=True)
    if detected_faces:
        output_path = os.path.join(detected_faces_dir, f"frame_{frame_index}.jpg")
        cv2.imwrite(output_path, frame)

    # Return unique actor names found in this frame
    return list(set(identity_map.values()))
