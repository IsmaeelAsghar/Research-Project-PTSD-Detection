from datetime import datetime
import os

from flask import Flask, jsonify, render_template, request, redirect, send_from_directory, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

import soundfile as sf
import moviepy.editor as mp
import resampy
from scipy.io import wavfile
from mtcnn import MTCNN

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure file upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp','mp4' }



#---------------------------------------------------------------------------------------
# Load models
audio_model_path = os.path.join(os.path.dirname(__file__), 'model', 'rest_net_full_4.h5')
audio_model = load_model(audio_model_path)

video_model_path = os.path.join(os.path.dirname(__file__), 'model', 'best_model_by_optuna_cap_2_2.h5')
video_model = tf.keras.models.load_model(video_model_path)




# MySQL Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@127.0.0.1/ptsd_project'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Define Doctor model
class Doctor(db.Model):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)  # Adjust if needed
    phone_number = db.Column(db.String(20))
    address = db.Column(db.Text)
    picture_filename = db.Column(db.String(255))

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Define Doctor registration form
class DoctorRegistrationForm(FlaskForm):
    fullname = StringField("Full Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo('password')])
    phone_number = StringField("Phone Number")
    address = TextAreaField("Address")
    picture = FileField("Upload Picture")
    submit = SubmitField("Register Doctor")
class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")

# -----------------
# logic for audio
# -----------------



# Constants
NUM_FRAMES = 2976
NUM_BANDS = 64
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01
EXAMPLE_WINDOW_SECONDS = 27.96
EXAMPLE_HOP_SECONDS = 15


def frame(data, window_length, hop_length):
    num_samples = data.shape[0]
    num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
    shape = (num_frames, window_length) + data.shape[1:]
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def periodic_hann(window_length):
    return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length)))

def stft_magnitude(signal, fft_length, hop_length=None, window_length=None):
    frames = frame(signal, window_length, hop_length)
    window = periodic_hann(window_length)
    windowed_frames = frames * window
    return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def hertz_to_mel(frequencies_hertz):
    return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

def spectrogram_to_mel_matrix(num_mel_bins=20, num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz < 0.0:
      raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
      raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
      raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)

    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))

    for i in range(num_mel_bins):
      lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
      lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
      upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
      mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
    mel_weights_matrix[0, :] = 0.0
    return mel_weights_matrix


def log_mel_spectrogram(data, audio_sample_rate=8000, log_offset=0.0,
                        window_length_secs=0.025, hop_length_secs=0.010,
                        **kwargs):
    window_length_samples = int(round(audio_sample_rate * window_length_secs))
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    spectrogram = stft_magnitude(
        data,
        fft_length=fft_length,
        hop_length=hop_length_samples,
        window_length=window_length_samples)

    mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(
        num_spectrogram_bins=spectrogram.shape[1],
        audio_sample_rate=audio_sample_rate, **kwargs))

    return np.log(mel_spectrogram + log_offset)


# Placeholder function for waveform to examples
def waveform_to_examples(data, sample_rate):
    # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  print("Mono Audio Data Shape:", data.shape)
  # Resample to the rate assumed by VGGish.
  if sample_rate != SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, SAMPLE_RATE)
  print("Resampled Audio Data Shape:", data.shape)

  # Compute log mel spectrogram features.

  log_mel = log_mel_spectrogram(
      data,
      audio_sample_rate=SAMPLE_RATE,
      log_offset=LOG_OFFSET,
      window_length_secs=STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=NUM_MEL_BINS,
      lower_edge_hertz=MEL_MIN_HZ,
      upper_edge_hertz=MEL_MAX_HZ)
  print("Log Mel Spectrogram Shape:", log_mel.shape)

  # Frame features into examples.
  features_sample_rate = 1.0 / STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  print("Log Mel Examples Shape:", log_mel_examples.shape)
  return log_mel_examples





# --------------------
# Data Loading Functions
# --------------------




def extract_audio(video_path, audio_path):
    try:
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        video_clip.close()
        return audio_path
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return None

def wav_read(wav_file):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    return wav_data, sr

# --------------------
# Audio Prediction Function
# --------------------

def predict_audio(video_path, model):
    try:
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_audio.wav")
        extracted_audio_path = extract_audio(video_path, audio_file_path)

        if extracted_audio_path is None:
            print(f"Error extracting audio from {video_path}")
            return None

        wav_data, sr = wav_read(extracted_audio_path)
        spectrograms = waveform_to_examples(wav_data, sr)
        max_timesteps = 2796

        if spectrograms.shape[0] > max_timesteps:
            spectrograms = spectrograms[:max_timesteps]
        elif spectrograms.shape[0] < max_timesteps:
            pad_width = [(0, max_timesteps - spectrograms.shape[0]), (0, 0), (0, 0)]
            spectrograms = np.pad(spectrograms, pad_width, 'constant')

        reshaped_examples = spectrograms.reshape(-1, 2796, 64, 3)
        normalized_examples = reshaped_examples / 255.0
        predictions = model.predict(normalized_examples)
        ptsd_probability = predictions[0][1]
        no_ptsd_probability = predictions[0][0]

        prediction_label = "PTSD" if ptsd_probability > no_ptsd_probability else "NO PTSD"
        prediction_score = ptsd_probability if ptsd_probability > no_ptsd_probability else no_ptsd_probability

        return prediction_score, prediction_label

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None, None


# --------------------
# Video Prediction Function
# --------------------

def predict_on_unseen_video(video_path, model, sequence_length=16, max_frames=100, time_interval=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frame_count = max(int(fps * time_interval), 1)
    detector = MTCNN()
    frame_sequence = []
    predictions_list = []

    for frame_index in range(0, min(frame_count, max_frames * interval_frame_count), interval_frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_faces(frame)
        for result in results:
            bbox = result['box']
            detected_face = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            resized_face = cv2.resize(detected_face, (224, 224))
            frame_sequence.append(resized_face)

            if len(frame_sequence) == sequence_length:
                sequence = np.array(frame_sequence)
                sequence = np.expand_dims(sequence, axis=0)
                predictions = model.predict(sequence)
                predictions_list.append(predictions)
                frame_sequence = []

    cap.release()

    if predictions_list:
        prediction_scores = np.mean(predictions_list, axis=0)
        predicted_class_index = np.argmax(prediction_scores)
        classes = ["NO PTSD", "PTSD"]
        final_prediction = classes[predicted_class_index]
        prediction_score = prediction_scores[0][predicted_class_index]
        return prediction_score, final_prediction

    return None, None
#


# ===================================================
#------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')


# Define the allowed picture extensions and the upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}
UPLOAD_FOLDER = os.path.join('static', 'doctors')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# ==================Admin panel




@app.route('/register_doctor', methods=['GET', 'POST'])
def register_doctor():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']
        phone_number = request.form['phone_number']
        address = request.form['address']
        picture = request.files['picture']
        
        # Check if picture file is uploaded and has allowed extension
        if picture and allowed_file(picture.filename):
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            picture_filename = secure_filename(f"{email}_{timestamp}.{picture.filename.rsplit('.', 1)[1].lower()}")
            picture_path = os.path.join(app.config['UPLOAD_FOLDER'], picture_filename)
            picture.save(picture_path)
        else:
            picture_filename = None
        
        try:
            new_doctor = Doctor(
                fullname=fullname,
                email=email,
                password_hash=generate_password_hash(password),
                phone_number=phone_number,
                address=address,
                picture_filename=picture_filename
            )
            db.session.add(new_doctor)
            db.session.commit()
            
            flash('Doctor registered successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        
        except Exception as e:
            db.session.rollback()
            print("Database Error:", str(e))
            flash(f'An error occurred: {str(e)}', 'error')
    
    return render_template('register_doctor.html')

# Ensure database is created
with app.app_context():
    db.create_all()





admin_email = 'ptsd_admin@gmail.com'
admin_password = 'ptsd'
hashed_admin_password = generate_password_hash(admin_password, method='pbkdf2:sha256', salt_length=16)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get form data
        email = request.form['email']
        password = request.form['password']

        # First check if the email belongs to an admin
        if email == admin_email:
            if check_password_hash(hashed_admin_password, password):
                session['user_id'] = 1  # You can set this to any constant or unique ID for the admin
                session['user_role'] = 'admin'
                flash('Admin login successful!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                flash('Invalid admin email or password.', 'danger')
                return redirect(url_for('login'))

        # If not admin, check if the email belongs to a doctor
        doctor = Doctor.query.filter_by(email=email).first()
        if doctor:
            if doctor.check_password(password):
                session['user_id'] = doctor.id
                session['user_role'] = 'doctor'
                flash('Doctor login successful!', 'success')
                return redirect(url_for('doctor_dashboard'))
            else:
                flash('Invalid doctor email or password.', 'danger')
                return redirect(url_for('login'))

        # If neither admin nor doctor, return an error
        flash('Invalid email or password.', 'danger')
        return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/doctor_dashboard')
def doctor_dashboard():
    if 'user_role' in session and session['user_role'] == 'doctor':
        return render_template('doctor_dashboard.html')
    else:
        flash('You must be a doctor to view this page.')
        return redirect(url_for('login'))


@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')






@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_role' in session and session['user_role'] == 'admin':
        doctors = Doctor.query.all()
        return render_template('admin_dashboard.html', doctors=doctors)
    else:
        flash('You must be an admin to view this page.')
        return redirect(url_for('login'))
    
    
    


# Predefined admin credentials
@app.route('/update_doctor/<int:doctor_id>', methods=['GET', 'POST'])
def update_doctor(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        phone_number = request.form['phone_number']
        address = request.form['address']
        
        # Handle picture update
        picture = request.files['picture']
        if picture and allowed_file(picture.filename):
            # Generate new picture filename based on email and timestamp
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            picture_filename = secure_filename(f"{email}_{timestamp}.{picture.filename.rsplit('.', 1)[1].lower()}")
            picture_path = os.path.join(app.config['UPLOAD_FOLDER'], picture_filename)
            picture.save(picture_path)
            doctor.picture_filename = picture_filename

        doctor.fullname = fullname
        doctor.email = email
        doctor.phone_number = phone_number
        doctor.address = address

        db.session.commit()
        return redirect(url_for('update_doctor', doctor_id=doctor_id))

    return render_template('update_doctor.html', doctor=doctor)



@app.route('/update_profile', methods=['POST'])
def update_profile():
    if request.method == 'POST':
        # Process the form submission here
        # You can access form data using request.form
        fullname = request.form['fullname']
        email = request.form['email']
        contact_number = request.form['contact_number']
        address = request.form['address']
        
        # Perform the update operation (e.g., update the database)
        
        # Redirect to a success page or back to the profile page
        return redirect(url_for('doctor_profile_view', doctor_id=Doctor.id))

    # If the request method is not POST, render an error page
    return render_template('error.html', message='Method Not Allowed')


@app.route('/remove_doctor/<int:doctor_id>', methods=['POST'])
def remove_doctor(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    db.session.delete(doctor)
    db.session.commit()
    return redirect(url_for('admin_dashboard'))




# =================================================
@app.route('/navbar')
def navbar():
     return render_template('navbar.html')


@app.route('/about')
def about():
     return render_template('about.html')


@app.route('/contactus')
def contactus():
     return render_template('contactus.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return render_template('result.html', video_path=file_path)

@app.route('/record', methods=['POST'])
def record_video():
    if 'video' in request.files:
        video = request.files['video']
        filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(video_path)
        return jsonify({"message": "Video uploaded successfully", "file_path": video_path}), 200
    return jsonify({"error": "No video part"}), 400







@app.route('/upload_recorded_video', methods=['POST'])
def handle_video_upload():
    video = request.files['video']
    if video:
        filename = secure_filename(video.filename)
        filepath = os.path.join('path_to_save_videos', filename)
        video.save(filepath)
        return jsonify({'message': 'Video successfully saved!', 'path': filepath})
    else:
        return jsonify({'error': 'No video uploaded'}), 400


@app.route('/videos')
def show_videos():
    # List all video files in the upload directory
    video_files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.lower().endswith(('.mp4', '.webm', '.ogg')):  # Check for video file extensions
            video_files.append(filename)
    return render_template('videos.html', videos=video_files)



#--------------------------------------------------


@app.route('/upload/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)







@app.route('/check_ptsd', methods=['GET', 'POST'])
def check_ptsd():
    # results = {'audio_prediction': 'Not processed', 'audio_score': 'Not processed', 'video_prediction': 'Not processed', 'video_score': 'Not processed'}  # Default results
    if request.method == 'POST':
        video_file = request.files['video']

        if video_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)

            try:
                audio_score, audio_prediction = predict_audio(video_path, audio_model)
            except Exception as e:
                print(f"Error processing audio: {e}")
                audio_score, audio_prediction = (None, "Error")

            try:
                video_score, video_prediction = predict_on_unseen_video(video_path, video_model, sequence_length=16, max_frames=200, time_interval=2)
            except Exception as e:
                print(f"Error processing video: {e}")
                video_score, video_prediction = (None, "Error")

            os.remove(video_path)
            results = {'audio_prediction': audio_prediction, 'audio_score': audio_score, 'video_prediction': video_prediction, 'video_score': video_score}
            print("Results being passed to template:", results)
            return render_template('result.html', results=results)
        
    return render_template('doctor_dashboard.html')


@app.route('/save_video', methods=['POST'])
def save_video():


    if 'video' not in request.files or 'name' not in request.form:
        return jsonify(success=False, message='No video file or name provided'), 400

    video = request.files['video']
    name = request.form['name']
    if video.filename == '' or name == '':
        return jsonify(success=False, message='No selected video file or name'), 400

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{name}_{timestamp}.mp4"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    return jsonify(success=True, video_path=video_path)


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_role', None)
    flash("You have been logged out successfully.", 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    
    app.run(debug=True)

