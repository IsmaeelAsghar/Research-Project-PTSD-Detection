from flask import Flask, jsonify, render_template, request, redirect, send_from_directory, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import resampy
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from wtforms.validators import DataRequired, Email, EqualTo
from werkzeug.security import generate_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, render_template, send_from_directory, url_for
import cv2
from mtcnn import MTCNN
import numpy as np
import tensorflow as tf
import soundfile as sf
import moviepy.editor as mp
from keras.models import load_model




app = Flask(__name__)

app.secret_key = 'your_secret_key_here'

# Configure file upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp','mp4' }



#---------------------------------------------------------------------------------------

# Load models 
base_dir = os.path.dirname(__file__)

audio_model_path = os.path.join(base_dir, 'model', 'yamnet_6.h5')
audio_model = load_model(audio_model_path)

video_model_path = os.path.join(base_dir, 'model', 'Refined_data_model_1.h5')
video_model = tf.keras.models.load_model(video_model_path)



class DoctorRegistrationForm(FlaskForm):
    fullname = StringField("Full Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired()])
    confirm_password = PasswordField("Confirm Password", validators=[DataRequired(), EqualTo('password')])
    phone_number = StringField("Phone Number")
    address = TextAreaField("Address")
    picture = FileField("Upload Picture")
    submit = SubmitField("Register Doctor")

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
import tensorflow as tf
import numpy as np
import moviepy.editor as mp
import soundfile as sf

# Define the Params class
class Params:
    def __init__(self):
        self.sample_rate = 16000.0
        self.stft_window_seconds = 0.025
        self.stft_hop_seconds = 0.010
        self.mel_bands = 64
        self.mel_min_hz = 125.0
        self.mel_max_hz = 7500.0
        self.log_offset = 0.001
        self.patch_window_seconds = 27.96
        self.patch_hop_seconds = 15
        self.num_classes = 521
        self.conv_padding = 'same'
        self.batchnorm_center = True
        self.batchnorm_scale = False
        self.batchnorm_epsilon = 1e-4
        self.classifier_activation = 'sigmoid'
        self.tflite_compatible = False

    @property
    def patch_frames(self):
        return int(round(self.patch_window_seconds / self.stft_hop_seconds))

    @property
    def patch_bands(self):
        return self.mel_bands

# Function to convert waveform to log-mel spectrogram patches
def waveform_to_log_mel_spectrogram_patches(waveform, params):
    window_length_samples = int(round(params.sample_rate * params.stft_window_seconds))
    hop_length_samples = int(round(params.sample_rate * params.stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    num_spectrogram_bins = fft_length // 2 + 1

    if params.tflite_compatible:
        magnitude_spectrogram = _tflite_stft_magnitude(
            signal=waveform,
            frame_length=window_length_samples,
            frame_step=hop_length_samples,
            fft_length=fft_length)
    else:
        magnitude_spectrogram = tf.abs(tf.signal.stft(
            signals=waveform,
            frame_length=window_length_samples,
            frame_step=hop_length_samples,
            fft_length=fft_length))

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.mel_bands,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.sample_rate,
        lower_edge_hertz=params.mel_min_hz,
        upper_edge_hertz=params.mel_max_hz)
    mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.log_offset)

    spectrogram_hop_length_samples = int(round(params.sample_rate * params.stft_hop_seconds))
    spectrogram_sample_rate = params.sample_rate / spectrogram_hop_length_samples
    patch_window_length_samples = int(round(spectrogram_sample_rate * params.patch_window_seconds))
    patch_hop_length_samples = int(round(spectrogram_sample_rate * params.patch_hop_seconds))
    features = tf.signal.frame(
        signal=log_mel_spectrogram,
        frame_length=patch_window_length_samples,
        frame_step=patch_hop_length_samples,
        axis=0)

    return log_mel_spectrogram, features

# Function to pad waveform
def pad_waveform(waveform, params):
    min_waveform_seconds = (
        params.patch_window_seconds +
        params.stft_window_seconds - params.stft_hop_seconds)
    min_num_samples = tf.cast(min_waveform_seconds * params.sample_rate, tf.int32)
    num_samples = tf.shape(waveform)[0]
    num_padding_samples = tf.maximum(0, min_num_samples - num_samples)

    num_samples = tf.maximum(num_samples, min_num_samples)
    num_samples_after_first_patch = num_samples - min_num_samples
    hop_samples = tf.cast(params.patch_hop_seconds * params.sample_rate, tf.int32)
    num_hops_after_first_patch = tf.cast(tf.math.ceil(
        tf.cast(num_samples_after_first_patch, tf.float32) /
        tf.cast(hop_samples, tf.float32)), tf.int32)
    num_padding_samples += (
        hop_samples * num_hops_after_first_patch - num_samples_after_first_patch)

    padded_waveform = tf.pad(waveform, [[0, num_padding_samples]],
                             mode='CONSTANT', constant_values=0.0)
    return padded_waveform

# Function to perform STFT for TFLite compatibility
def _tflite_stft_magnitude(signal, frame_length, frame_step, fft_length):
    def _hann_window():
        return tf.reshape(
            tf.constant(
                (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
                ).astype(np.float32),
                name='hann_window'), [1, frame_length])

    def _dft_matrix(dft_length):
        omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
        return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

    def _rdft(framed_signal, fft_length):
        complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(
            fft_length // 2 + 1), :].transpose()
        real_dft_matrix = tf.constant(
            np.real(complex_dft_matrix_kept_values).astype(np.float32),
            name='real_dft_matrix')
        imag_dft_matrix = tf.constant(
            np.imag(complex_dft_matrix_kept_values).astype(np.float32),
            name='imaginary_dft_matrix')
        signal_frame_length = tf.shape(framed_signal)[-1]
        half_pad = (fft_length - signal_frame_length) // 2
        padded_frames = tf.pad(
            framed_signal,
            [
                [0, 0],
                [half_pad, fft_length - signal_frame_length - half_pad]
            ],
            mode='CONSTANT',
            constant_values=0.0)
        real_stft = tf.matmul(padded_frames, real_dft_matrix)
        imag_stft = tf.matmul(padded_frames, imag_dft_matrix)
        return real_stft, imag_stft

    def _complex_abs(real, imag):
        return tf.sqrt(tf.add(real * real, imag * imag))

    framed_signal = tf.signal.frame(signal, frame_length, frame_step)
    windowed_signal = framed_signal * _hann_window()
    real_stft, imag_stft = _rdft(windowed_signal, fft_length)
    stft_magnitude = _complex_abs(real_stft, imag_stft)
    return stft_magnitude

def extract_audio(video_path, audio_path):
    try:
        video_clip = mp.VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

def load_audio_and_convert_to_spectrogram(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform, sample_rate = tf.audio.decode_wav(audio_binary)

    if waveform.shape[-1] == 2:
        waveform = tf.reduce_mean(waveform, axis=-1)

    if len(waveform.shape) == 2 and waveform.shape[-1] == 1:
        waveform = tf.squeeze(waveform, axis=-1)

    # Assuming Params class and related functions are defined somewhere in your code
    params = Params()
    padded_waveform = pad_waveform(waveform, params)
    log_mel_spectrogram, features = waveform_to_log_mel_spectrogram_patches(padded_waveform, params)

    return log_mel_spectrogram, features

def predict_ptsd(file_path, model):
    log_mel_spectrogram, features = load_audio_and_convert_to_spectrogram(file_path)
    features = tf.expand_dims(features, axis=-1)
    features = tf.concat([features, features, features], axis=-1)
    predictions = model.predict(features)
    
    # Get the predicted class per feature
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Determine the majority class from all feature predictions
    final_prediction = np.bincount(predicted_classes).argmax()
    
    # Calculate the prediction score (proportion of features predicted as the majority class)
    prediction_score = np.bincount(predicted_classes)[final_prediction] / len(predicted_classes)
    
    # Map numeric prediction to labels
    labels = {0: "No PTSD", 1: "PTSD"}
    return labels.get(final_prediction, "Unknown"), prediction_score


# logic for video

def predict_ptsd_from_video(video_path, model):
    audio_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_audio.wav")
    if extract_audio(video_path, audio_temp_path):
        return predict_ptsd(audio_temp_path, model)
    else:
        return "Audio extraction failed", 0.0



def predict_on_unseen_video(video_path, model, sequence_length=15, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return "Error", 0.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frame count of the video
    segment_interval = max(int(frame_count / max_frames), 1)  # Interval to pick frames from each segment
    detector = MTCNN()
    frame_sequence = []
    predictions_list = []

    for segment_index in range(max_frames):
        start_frame_index = segment_index * segment_interval
        end_frame_index = min((segment_index + 1) * segment_interval, frame_count)

        found_face = False
        for step in range(0, 15, 3):  # Check up to 5 frames within the segment, stepping by 3 frames each time
            frame_index = start_frame_index + step
            if frame_index >= end_frame_index:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.detect_faces(frame)
            if results:
                found_face = True
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
                        frame_sequence = []  # Reset for the next sequence
                if found_face:
                    break

    cap.release()

    # Handle predictions
    if predictions_list:
        prediction_scores = np.mean(predictions_list, axis=0)
        predicted_class_index = np.argmax(prediction_scores)
        classes = ["NO PTSD", "PTSD"]
        final_prediction = classes[predicted_class_index]
        return final_prediction, prediction_scores.flatten()[predicted_class_index]
    else:
        return "No faces detected or no frames processed.", 0.0



@app.route('/check_ptsd', methods=['GET', 'POST'])
def check_ptsd():
    if request.method == 'POST':
        video_file = request.files['video']

        if video_file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
            video_file.save(video_path)

            try:
                audio_prediction, audio_score = predict_ptsd_from_video(video_path, audio_model)
            except Exception as e:
                print(f"Error processing audio: {e}")
                audio_prediction, audio_score = "Error", 0.0

            try:
                video_prediction, video_score = predict_on_unseen_video(video_path, video_model, sequence_length=15, max_frames=150)
            except Exception as e:
                print(f"Error processing video: {e}")
                video_prediction, video_score = "Error", 0.0

            os.remove(video_path)
            results = {
                'audio_prediction': audio_prediction,
                'audio_score': audio_score,
                'video_prediction': video_prediction,
                'video_score': video_score
            }
            print("Results being passed to template:", results)
            return render_template('result.html', results=results)
        
    return render_template('doctor_dashboard.html')


# ===================================================

# Define the allowed picture extensions and the upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}
UPLOAD_FOLDER = os.path.join('static', 'doctors')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html')



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
                #flash('Admin login successful!', 'success')
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
                #flash('Doctor login successful!', 'success')
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




# -----------------------------
# 
# 
# =================================================
@app.route('/navbar')
def navbar():
     return render_template('navbar.html')

@app.route('/admin_navbar')
def admin_navbar():
     return render_template('admin_navbar.html')


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







# @app.route('/check_ptsd', methods=['GET', 'POST'])
# def check_ptsd():
#     # results = {'audio_prediction': 'Not processed', 'audio_score': 'Not processed', 'video_prediction': 'Not processed', 'video_score': 'Not processed'}  # Default results
#     if request.method == 'POST':
#         video_file = request.files['video']

#         if video_file:
#             video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
#             video_file.save(video_path)

#             try:
#                 audio_score, audio_prediction = predict_audio(video_path, audio_model)
#             except Exception as e:
#                 print(f"Error processing audio: {e}")
#                 audio_score, audio_prediction = (None, "Error")

#             try:
#                 video_score, video_prediction = predict_on_unseen_video(video_path, video_model, sequence_length=16, max_frames=200, time_interval=2)
#             except Exception as e:
#                 print(f"Error processing video: {e}")
#                 video_score, video_prediction = (None, "Error")

#             os.remove(video_path)
#             results = {'audio_prediction': audio_prediction, 'audio_score': audio_score, 'video_prediction': video_prediction, 'video_score': video_score}
#             print("Results being passed to template:", results)
#             return render_template('result.html', results=results)
        
#     return render_template('doctor_dashboard.html')


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
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    
    app.run(debug=True)

