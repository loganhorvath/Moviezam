import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from face_rec import process_video
import requests
from dotenv import load_dotenv
from config import UPLOAD_FOLDER, RESULT_FOLDER, DB_PATH

load_dotenv()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_person_id_tmdb(actor_name):
    api_key = os.getenv("API_KEY")
    url = "https://api.themoviedb.org/3/search/person"
    params = {"api_key": api_key, "query": actor_name}
    response = requests.get(url, params=params)
    data = response.json()
    if data.get("results"):
        return data["results"][0]["id"]
    return None

def get_actor_movies_tmdb(person_id):
    api_key = os.getenv("API_KEY")
    url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits"
    params = {"api_key": api_key}
    response = requests.get(url, params=params)
    data = response.json()
    movies = set()
    for movie in data.get("cast", []):
        if "title" in movie:
            movies.add(movie["title"])
    return movies

def find_shared_movies(movie_lists):
    if not movie_lists:
        return []
    return list(set.intersection(*movie_lists))

@app.route('/', methods=['GET', 'POST'])
def index():
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

            total_frames, recognized_actors = process_video(video_path, DB_PATH, output_dir)
            
            # Get TMDb person IDs and movie lists
            movie_lists = []
            for actor in recognized_actors:
                person_id = get_person_id_tmdb(actor)
                if person_id:
                    movies = get_actor_movies_tmdb(person_id)
                    movie_lists.append(movies)
            
            shared_movies = find_shared_movies(movie_lists)
            
            frame_paths = [f"{filename.rsplit('.', 1)[0]}/frame_{i}.jpg" for i in range(total_frames)]
            return render_template('results.html', frames=frame_paths, shared_movies=shared_movies, actors=recognized_actors)
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_dir = os.path.join(app.config['RESULT_FOLDER'], filename.rsplit('.', 1)[0])

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        file.save(video_path)

        total_frames, recognized_actors = process_video(video_path, DB_PATH, output_dir)

        # Get TMDb person IDs and movie lists
        movie_lists = []
        for actor in recognized_actors:
            person_id = get_person_id_tmdb(actor)
            if person_id:
                movies = get_actor_movies_tmdb(person_id)
                movie_lists.append(movies)

        shared_movies = find_shared_movies(movie_lists)

        # Prepare actor and movie data for frontend
        actors_data = [{'name': name} for name in recognized_actors]
        movies_data = [{'title': title} for title in shared_movies]

        return jsonify({'actors': actors_data, 'movies': movies_data})

    return jsonify({'error': 'Invalid file type.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
