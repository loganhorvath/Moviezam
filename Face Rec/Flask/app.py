import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from face_recognition_video import process_video

UPLOAD_FOLDER = r'E:\Github\Moviezam\Face Rec\Flask\uploads'
RESULT_FOLDER = r'E:\Github\Moviezam\Face Rec\Flask\output_frames'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
DB_PATH = r'E:\Github\Moviezam\Face Rec\Flask\actors_DB'  # 👈 Replace with your real DB path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No video file provided."
        file = request.files['video']
        if file.filename == '':
            return "No selected file."
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_dir = os.path.join(app.config['RESULT_FOLDER'], filename.rsplit('.', 1)[0])

            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            file.save(video_path)

            total_frames = process_video(video_path, DB_PATH, output_dir)
            frame_paths = [f"{filename.rsplit('.', 1)[0]}/frame_{i}.jpg" for i in range(total_frames)]

            return render_template('results.html', frames=frame_paths)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
