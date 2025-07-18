import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from face_rec import process_video
import requests
from dotenv import load_dotenv
from config import UPLOAD_FOLDER, RESULT_FOLDER, DB_PATH
import yt_dlp

load_dotenv()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_person_id_tmdb(actor_name):
    bearer_token = os.getenv("TMDB_BEARER_TOKEN")
    url = f"https://api.themoviedb.org/3/search/person?query={actor_name.replace(' ', '%20')}&include_adult=false&language=en-US&page=1"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    if data.get("results"):
        return data["results"][0]["id"]
    return None

def get_actor_movies_tmdb(person_id):
    bearer_token = os.getenv("TMDB_BEARER_TOKEN")
    url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }
    response = requests.get(url, headers=headers)
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

def download_video_from_url(url, download_dir):
    ydl_opts = {
        'outtmpl': os.path.join(download_dir, 'input_video.%(ext)s'),
        'format': 'mp4/bestaudio/best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
        return video_path

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check for file upload
    if 'video' in request.files and request.files['video'].filename != '':
        file = request.files['video']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_dir = os.path.join(app.config['RESULT_FOLDER'], filename.rsplit('.', 1)[0])

            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            file.save(video_path)
        else:
            return jsonify({'error': 'Invalid file type.'}), 400

    # Check for URL upload
    elif request.form.get('videoUrl'):
        video_url = request.form.get('videoUrl')
        try:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            video_path = download_video_from_url(video_url, app.config['UPLOAD_FOLDER'])
            output_dir = os.path.join(app.config['RESULT_FOLDER'], 'url_upload')
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return jsonify({'error': f'Failed to download video: {str(e)}'}), 400

    else:
        return jsonify({'error': 'No video file provided.'}), 400

    # Process video (from either upload or URL)
    total_frames, recognized_actors = process_video(video_path, DB_PATH, output_dir)

    # Get TMDb person IDs and movie lists
    movie_lists = []
    actors_data = []
    for actor in recognized_actors:
        # Normalize actor name for TMDb
        actor_name = actor.replace("_", " ")
        person_id = get_person_id_tmdb(actor_name)
        if person_id:
            movies = get_actor_movies_tmdb(person_id)
            movie_lists.append(movies)
            actors_data.append({'name': actor_name})
        else:
            actors_data.append({'name': actor_name + " (not found on TMDb)"})

    shared_movies = find_shared_movies(movie_lists)

    # Prepare movie data for frontend
    movies_data = [{'title': title} for title in shared_movies]

    return jsonify({'actors': actors_data, 'movies': movies_data})

if __name__ == '__main__':
    app.run(debug=True)
