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
    frame_interval = int(fps * 5)  # Process every 5 seconds
    frames = []
    frame_index = 0
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame % frame_interval == 0:
            frames.append((frame, frame_index, db_path, output_dir))
            frame_index += 1
        current_frame += 1
    cap.release()

    with Pool(processes=4) as pool:
        results = pool.map(process_frame_wrapper, frames)

    all_actors = set()
    for actors_in_frame in results:
        all_actors.update(actors_in_frame)
    return frame_index, list(all_actors)

def process_frame_wrapper(args):
    return process_frame(*args)

# Prepare arguments for each frame
frame_args = [(frame, idx, db_path, output_dir) for idx, frame in enumerate(frames_to_process)]
with Pool(processes=4) as pool:  # Adjust number of processes to your CPU
    results = pool.map(process_frame_wrapper, frame_args)

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
    best_matches = df_results[df_results["distance"] < df_results["threshold"]]

    identity_map = {}
    for identity_path, distance in zip(best_matches["identity"], best_matches["distance"]):
        person_name = os.path.basename(os.path.dirname(identity_path))
        if person_name not in identity_map:
            identity_map[identity_path] = person_name

    # Draw boxes and names
    for (box, _) in zip(face_bounding_boxes, detected_faces):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        matched_identity = next(iter(identity_map.values()), None)
        if matched_identity:
            cv2.putText(frame, matched_identity, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
    cv2.imwrite(output_path, frame)

    # Return list of actor names found in this frame
    return list(identity_map.values())

    height, width, _ = frame.shape
    small_frame = cv2.resize(frame, (width // 2, height // 2))
