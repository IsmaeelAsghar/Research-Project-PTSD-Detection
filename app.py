from flask import Flask, jsonify, render_template, request, redirect, send_from_directory, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FileField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure file upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

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


# Define the allowed picture extensions and the upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','webp'}
UPLOAD_FOLDER = os.path.join('static', 'doctors')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===================================================
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

@app.route('/result')
def result():
    return render_template('result.html')



@app.route('/videos')
def show_videos():
    # List all video files in the upload directory
    video_files = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.lower().endswith(('.mp4', '.webm', '.ogg')):  # Check for video file extensions
            video_files.append(filename)
    return render_template('videos.html', videos=video_files)


@app.route('/upload/<filename>')
def serve_video(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/detect_ptsd', methods=['POST'])
def detect_ptsd():
    if 'video' in request.files:
        video = request.files['video']
        # Process video to detect PTSD (replace with your logic)
        # Example: Simulate detection
        result = 'PTSD' if video.filename.startswith('recorded_video') else 'NO PTSD'
        return result
    return 'Failed to detect PTSD'

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_role', None)
    flash("You have been logged out successfully.", 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    
    app.run(debug=True)




