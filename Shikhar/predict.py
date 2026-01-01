from flask import Flask, request, jsonify, render_template, url_for, redirect
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length
from flask_bcrypt import Bcrypt
from jsonschema import ValidationError  # for validation
import logging
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup paths and config
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'backend', 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'mysecretkey'  # Used for securely signing session cookies

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
logging.basicConfig(level=logging.INFO)

# Setup login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

# User loader for login manager
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Registration form
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)],
                           render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)],
                             render_kw={"placeholder": "Password"})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user = User.query.filter_by(username=username.data).first()
        if existing_user:
            raise ValidationError('That username already exists. Please choose a different one.')

# Login form
class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)],
                           render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)],
                             render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', form=form, error="Invalid username or password.")
    return render_template('login.html', form=form)

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

# Protected dashboard route
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user.username)

# Logout
@app.route('/logout', methods=['GET'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Welcome message API
'''@app.route("/welcome", methods=['GET'])
def welcome():
    welcome_messages = [
        "Welcome to the chatbot! How can I assist you today?",
        "You can ask me about the weather.",
        "Feel free to say 'hello' or ask me anything else!",
        "I'm here to help you with your queries."
    ]
    return jsonify({"messages": welcome_messages})'''

# === OPTIONAL CHATBOT & WEATHER HANDLERS ===

'''
# Uncomment and implement these later if needed

# from chat import get_response, models
# from weather.weather import get_weather

@app.route("/weather", methods=['POST'])
def weather():
    logging.info("Received weather request: %s ", request.json)
    data = request.get_json()
    if not data or "city" not in data:
        return jsonify({"error": "City is required"}), 400
    city = data["city"]
    if len(city) == 0:
        return jsonify({"error": "City name cannot be empty"}), 400
    try:
        weather_response = get_weather(city)
    except Exception as e:
        logging.error("Error fetching weather data: %s", str(e))
        return jsonify({"error": "Failed to fetch weather data"}), 500
    return jsonify(weather_response)

@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received request: %s", request.json)
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400
    text = data["message"]
    if len(text) == 0:
        return jsonify({"error": "Message cannot be empty"}), 400
    if "urgent" in text.lower():
        logging.info("Received an urgent message: %s", text)
    try:
        response = get_response(text, models)
    except Exception as e:
        logging.error("Error in get_response: %s", str(e))
        return jsonify({"error": "An error occurred while processing your request."}), 500
    return jsonify({"answer": response})
'''

# Initialize database
with app.app_context():
    db.create_all()

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
