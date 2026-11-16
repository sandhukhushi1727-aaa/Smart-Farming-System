import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
from PIL import Image, UnidentifiedImageError
import os
import math
import uuid
import traceback
from pathlib import Path

import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory
)

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from db import get_db_connection
from ml.predict_disease import predict_disease

DISEASE_CONFIDENCE_THRESHOLD = 60.0


# ==========================================
# FLASK APPLICATION CONFIGURATION
# ==========================================

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")

if not app.secret_key:
    raise RuntimeError("SECRET_KEY is missing from environment variables.")

app.config.update(

    SESSION_COOKIE_HTTPONLY=True,

    SESSION_COOKIE_SAMESITE="Lax",

    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production"

)


# ==========================================
# PROJECT DIRECTORY
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent


# ==========================================
# IMAGE UPLOAD CONFIGURATION
# ==========================================

UPLOAD_FOLDER = PROJECT_DIR / "uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# IMAGE VALIDATION FUNCTION
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ==========================================
# LOAD CROP RECOMMENDATION ML MODEL
# ==========================================

MODEL_PATH = PROJECT_DIR / "ml" / "models" / "crop_model.pkl"

saved_crop_model = joblib.load(MODEL_PATH)

crop_model = saved_crop_model["model"]

crop_label_encoder = saved_crop_model["label_encoder"]

crop_feature_names = saved_crop_model["feature_names"]

print("Crop recommendation model loaded successfully!")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# ABOUT PAGE
# ==========================================

@app.route("/about")
def about():

    return render_template("about.html")


# ==========================================
# CONTACT PAGE
# ==========================================

@app.route("/contact", methods=["GET", "POST"])
def contact():

    if request.method == "POST":

        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        form_data = {
            "name": name,
            "email": email,
            "subject": subject,
            "message": message
        }

        if not all([name, email, subject, message]):

            return render_template(
                "contact.html",
                error="Please fill in all the fields.",
                form_data=form_data
            )

        if len(name) > 100 or len(email) > 254:

            return render_template(
                "contact.html",
                error="Please check the length of your name or email.",
                form_data=form_data
            )

        if len(subject) > 150 or len(message) > 3000:

            return render_template(
                "contact.html",
                error="Your subject or message is too long.",
                form_data=form_data
            )

        connection = get_db_connection()

        if connection is None:

            return render_template(
                "contact.html",
                error="Database connection failed. Please try again later.",
                form_data=form_data
            )

        cursor = None

        try:

            cursor = connection.cursor()

            query = """
                INSERT INTO contact_messages
                (name, email, subject, message)
                VALUES (%s, %s, %s, %s)
            """

            cursor.execute(
                query,
                (name, email, subject, message)
            )

            connection.commit()

            return render_template(
                "contact.html",
                success="Thank you! Your message was received.",
                form_data={}
            )

        except Exception:

            connection.rollback()

            app.logger.exception("Contact form submission failed")

            return render_template(
                "contact.html",
                error="Unable to save your message. Please try again.",
                form_data=form_data
            )

        finally:

            if cursor is not None:
                cursor.close()

            connection.close()

    return render_template("contact.html", form_data={})


# ==========================================
# CROP RECOMMENDATION PAGE
# ML MODEL + MYSQL DATABASE
# ==========================================

@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():

    if request.method == "POST":

        values, error = validate_crop_inputs(request.form)

        if error:
            flash(error, "error")
            return redirect(url_for("crop_recommendation"))

        nitrogen = values["nitrogen"]
        phosphorus = values["phosphorus"]
        potassium = values["potassium"]
        temperature = values["temperature"]
        humidity = values["humidity"]
        ph = values["ph"]
        rainfall = values["rainfall"]

        try:

            # Check whether all fields are filled

            if not all(form_data.values()):

                return render_template(
                    "crop.html",
                    error="Please fill in all seven fields.",
                    form_data=form_data
                )

            # Convert input values into numbers

            N = float(form_data["nitrogen"])
            P = float(form_data["phosphorus"])
            K = float(form_data["potassium"])

            temperature = float(form_data["temperature"])
            humidity = float(form_data["humidity"])
            ph = float(form_data["ph"])
            rainfall = float(form_data["rainfall"])

            values_to_check = [
                N, P, K, temperature,
                humidity, ph, rainfall
            ]

            # Reject invalid numerical values

            if not all(math.isfinite(value) for value in values_to_check):

                raise ValueError("Invalid numerical value.")

            # Prepare input for ML model

            input_data = pd.DataFrame(
                [[N, P, K, temperature, humidity, ph, rainfall]],
                columns=crop_feature_names
            )

            # Predict crop

            prediction = crop_model.predict(input_data)

            crop_name = crop_label_encoder.inverse_transform(
                prediction
            )[0]

            # Calculate model confidence

            probabilities = crop_model.predict_proba(input_data)[0]

            confidence = round(
                float(max(probabilities)) * 100,
                2
            )

            # ==================================
            # SAVE PREDICTION IN MYSQL
            # ==================================

            connection = None
            cursor = None
            save_error = False

            try:

                connection = get_db_connection()

                if connection is None:

                    raise RuntimeError("Database connection failed.")

                cursor = connection.cursor()

                user_id = session.get("user_id")

                query = """
                    INSERT INTO crop_predictions (
                        user_id,
                        nitrogen,
                        phosphorus,
                        potassium,
                        temperature,
                        humidity,
                        ph,
                        rainfall,
                        recommended_crop,
                        confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                database_values = (
                    user_id,
                    N,
                    P,
                    K,
                    temperature,
                    humidity,
                    ph,
                    rainfall,
                    crop_name,
                    confidence
                )

                cursor.execute(query, database_values)

                connection.commit()

                print("Crop prediction saved successfully!")

            except Exception:

                save_error = True

                if connection is not None:

                    try:
                        connection.rollback()

                    except Exception:
                        pass

                app.logger.exception(
                    "Unable to save crop prediction"
                )

            finally:

                if cursor is not None:
                    cursor.close()

                if connection is not None:
                    connection.close()

            # ==================================
            # DISPLAY PREDICTION RESULT
            # ==================================

            return render_template(
                "crop_result.html",

                crop_name=crop_name,
                confidence=confidence,
                saved=not save_error,

                N=N,
                P=P,
                K=K,

                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall=rainfall
            )

        except (ValueError, TypeError):

            return render_template(
                "crop.html",
                error="Please enter valid numerical values for all fields.",
                form_data=form_data
            )

        except Exception:

            app.logger.exception("Crop prediction failed")

            return render_template(
                "crop.html",
                error="Something went wrong while predicting the crop.",
                form_data=form_data
            )

    return render_template("crop.html", form_data={})


# ==========================================
# PLANT DISEASE DETECTION PAGE
# CNN MODEL + MYSQL DATABASE
# ==========================================

@app.route("/disease-detection", methods=["GET", "POST"])
def disease_detection():

    if request.method == "POST":

        # Check whether an image was selected

        if "leaf_image" not in request.files:

            flash("No image selected!", "error")
            return redirect(request.url)

        file = request.files["leaf_image"]

        # Check whether the filename is empty

        if file.filename == "":

            flash("Please select an image!", "error")
            return redirect(request.url)

        # Validate image format

        if not allowed_file(file.filename):

            flash("Only JPG, JPEG and PNG images are allowed!", "error")
            return redirect(request.url)

        connection = None
        cursor = None

        try:

            # Generate a unique filename

            original_filename = secure_filename(file.filename)

            extension = original_filename.rsplit(".", 1)[1].lower()

            new_filename = f"{uuid.uuid4().hex}.{extension}"

            # Save uploaded image

            save_path = UPLOAD_FOLDER / new_filename

            file.save(save_path)

            # Validate actual image content

            try:

                image = Image.open(uploaded_file)

                # Check actual image format
                if image.format not in ["JPEG", "PNG"]:

                    flash("Only JPG, JPEG and PNG images are allowed.", "error")
                    return redirect(url_for("disease_detection"))

                # Verify image is not corrupted
                image.verify()

                # Reset file pointer before saving
                uploaded_file.seek(0)

            except (UnidentifiedImageError, OSError, ValueError):

                flash("Invalid or corrupted image. Please upload a valid image.", "error")
                return redirect(url_for("disease_detection"))

            # ==================================
            # PREDICT DISEASE USING CNN MODEL
            # ==================================

            disease, confidence = predict_disease(str(save_path))

            confidence = round(float(confidence), 2)

            print("Predicted Disease:", disease)
            print("Prediction Confidence:", confidence)

            # DAY 59 - Model Confidence Handling

            confidence = round(float(confidence), 2)

            DISEASE_CONFIDENCE_THRESHOLD = 60.0

            if confidence < 60.0:

                return render_template(
                "disease_result.html",
                prediction_uncertain=True,
                confidence=confidence
            )

            # ==================================
            # CONNECT TO MYSQL DATABASE
            # ==================================

            connection = get_db_connection()

            if connection is None:

                raise RuntimeError("Database connection failed.")

            cursor = connection.cursor(dictionary=True)

            # ==================================
            # FETCH DISEASE INFORMATION
            # FROM DISEASES TABLE
            # ==================================

            cursor.execute(
                """
                SELECT
                    disease_id,
                    crop_name,
                    disease_name,
                    symptoms,
                    causes,
                    prevention,
                    management
                FROM diseases
                WHERE disease_name = %s
                """,
                (disease,)
            )

            disease_details = cursor.fetchone()

            print("Database Result:", disease_details)

            # Check whether disease information exists

            if disease_details is None:

                flash(
                    f"Disease information for '{disease}' was not found in the database.",
                    "error"
                )

                return redirect(request.url)

            # ==================================
            # SAVE DISEASE PREDICTION IN MYSQL
            # ==================================

            user_id = session.get("user_id")

            # User must be logged in because user_id is NOT NULL
            if user_id is None:
                flash("Please login before detecting a disease.", "error")
                return redirect(url_for("login"))

            image_path = f"uploads/{new_filename}"

            cursor.execute(
                """
                INSERT INTO disease_predictions
                (
                    user_id,
                    disease_id,
                    image_path,
                    predicted_disease,
                    confidence
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    user_id,
                    disease_details["disease_id"],
                    image_path,
                    disease_details["disease_name"],
                    confidence
                )
            )

            connection.commit()

            print("Disease prediction saved successfully!")

            # ==================================
            # DISPLAY DISEASE RESULT
            # ==================================

            return render_template(
                "disease_result.html",

                disease=disease_details["disease_name"],

                crop=disease_details["crop_name"],

                confidence=confidence,

                image_filename=new_filename,

                symptoms=disease_details["symptoms"],

                causes=disease_details["causes"],

                prevention=disease_details["prevention"],

                management=disease_details["management"]
            )

        except Exception as e:

            # Rollback database transaction if an error occurs

            if connection is not None:

                try:
                    connection.rollback()

                except Exception:
                    pass

            app.logger.exception("Disease prediction failed")

            print("Disease Prediction Error:", e)

            traceback.print_exc()

            flash(
                "Unable to predict disease. Please check the server terminal for details.",
                "error"
            )

            return redirect(request.url)

        finally:

            # Close cursor only after SELECT and INSERT are completed

            if cursor is not None:

                try:
                    cursor.close()

                except Exception:
                    pass

            # Close database connection

            if connection is not None:

                try:
                    connection.close()

                except Exception:
                    pass

    return render_template("disease.html")


# ==========================================
# LOGIN PAGE
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()

        password = request.form.get("password", "")

        if not email or not password:

            return render_template(
                "login.html",
                error="Please enter your email and password."
            )

        connection = get_db_connection()

        if connection is None:

            return render_template(
                "login.html",
                error="Database connection failed. Please try again."
            )

        cursor = None

        try:

            cursor = connection.cursor(dictionary=True)

            cursor.execute(
                """
                SELECT user_id, name, email, password_hash, role
                FROM users
                WHERE email = %s
                """,
                (email,)
            )

            user = cursor.fetchone()

            if user and check_password_hash(
                user["password_hash"], password
            ):

                session.clear()

                session["user_id"] = user["user_id"]

                session["user_name"] = user["name"]

                session["user_email"] = user["email"]

                session["role"] = user["role"]

                if session.get("role") == "admin":
                    return redirect(url_for("admin"))
                else:
                    return redirect(url_for("dashboard"))

            return render_template(
                "login.html",
                error="Invalid email or password."
            )

        except Exception:

            app.logger.exception("Login failed")

            return render_template(
                "login.html",
                error="Unable to log in right now. Please try again."
            )

        finally:

            if cursor is not None:
                cursor.close()

            connection.close()

    return render_template("login.html")



# ==========================================
# DASHBOARD PAGE
# ADVANCED DASHBOARD STATISTICS
# ==========================================

# ==========================================
# DASHBOARD PAGE
# STATISTICS + CHARTS
# ==========================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    connection = None
    cursor = None

    # Default statistics
    total_crop_predictions = 0
    total_disease_predictions = 0
    healthy_plants = 0
    diseased_plants = 0

    # Default chart data
    crop_distribution = []
    disease_distribution = []

    try:

        connection = get_db_connection()

        if connection is None:
            raise RuntimeError("Database connection failed.")

        cursor = connection.cursor(dictionary=True)

        # ==================================
        # TOTAL CROP PREDICTIONS
        # ==================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM crop_predictions
            WHERE user_id = %s
            """,
            (user_id,)
        )

        total_crop_predictions = cursor.fetchone()["total"]

        # ==================================
        # DISEASE STATISTICS
        # ==================================

        cursor.execute(
            """
            SELECT
                COUNT(*) AS total_disease,
                COALESCE(
                    SUM(
                        CASE
                            WHEN LOWER(TRIM(predicted_disease))
                                 LIKE %s
                            THEN 1
                            ELSE 0
                        END
                    ),
                    0
                ) AS healthy
            FROM disease_predictions
            WHERE user_id = %s
            """,
            ("%healthy%", user_id)
        )

        disease_result = cursor.fetchone()

        total_disease_predictions = disease_result["total_disease"]

        healthy_plants = disease_result["healthy"]

        diseased_plants = (
            total_disease_predictions - healthy_plants
        )

        # ==================================
        # CROP RECOMMENDATION DISTRIBUTION
        # ==================================

        cursor.execute(
            """
            SELECT
                recommended_crop,
                COUNT(*) AS total
            FROM crop_predictions
            WHERE user_id = %s
            GROUP BY recommended_crop
            ORDER BY total DESC
            """,
            (user_id,)
        )

        crop_distribution = cursor.fetchall()

        # ==================================
        # DISEASE DETECTION DISTRIBUTION
        # ==================================

        cursor.execute(
            """
            SELECT
                predicted_disease,
                COUNT(*) AS total
            FROM disease_predictions
            WHERE user_id = %s
            GROUP BY predicted_disease
            ORDER BY total DESC
            """,
            (user_id,)
        )

        disease_distribution = cursor.fetchall()

    except Exception:

        app.logger.exception(
            "Unable to load dashboard data"
        )

        flash(
            "Unable to load dashboard statistics and charts.",
            "error"
        )

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return render_template(
        "dashboard.html",

        name=session.get("user_name", "Farmer"),

        total_crop_predictions=total_crop_predictions,

        total_disease_predictions=total_disease_predictions,

        healthy_plants=healthy_plants,

        diseased_plants=diseased_plants,

        crop_distribution=crop_distribution,

        disease_distribution=disease_distribution
    )


# ==========================================
# HISTORY PAGE
# CROP + DISEASE PREDICTION HISTORY
# ==========================================

@app.route("/history")
def history():

    # Check login
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_id = session["user_id"]

    connection = None
    cursor = None

    crop_history = []
    disease_history = []

    try:

        # Connect to MySQL
        connection = get_db_connection()

        if connection is None:
            raise RuntimeError("Database connection failed.")

        cursor = connection.cursor(dictionary=True)

        # ==================================
        # FETCH CROP PREDICTION HISTORY
        # ==================================

        cursor.execute(
            """
            SELECT
                recommended_crop,
                confidence,
                created_at
            FROM crop_predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )

        crop_history = cursor.fetchall()

        # ==================================
        # FETCH DISEASE PREDICTION HISTORY
        # ==================================

        cursor.execute(
            """
            SELECT
                predicted_disease,
                confidence,
                created_at
            FROM disease_predictions
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )

        disease_history = cursor.fetchall()

    except Exception:

        app.logger.exception("Unable to fetch prediction history")

        flash(
            "Unable to load prediction history. Please try again.",
            "error"
        )

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    # ==================================
    # DISPLAY HISTORY PAGE
    # ==================================

    return render_template(
        "history.html",
        crop_history=crop_history,
        disease_history=disease_history
    )

@app.route("/admin")
def admin():

    # Check whether user is logged in
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    # Check whether logged-in user is admin
    if session.get("role") != "admin":
        flash("Access denied! Admin privileges required.", "error")
        return redirect(url_for("dashboard"))

    connection = None
    cursor = None

    total_users = 0
    crop_predictions = 0
    disease_predictions = 0
    total_predictions = 0

    try:
        connection = get_db_connection()

        if connection is None:
            raise RuntimeError("Database connection failed.")

        cursor = connection.cursor(dictionary=True)

        # Total registered users
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM users
        """)
        total_users = cursor.fetchone()["total"]

        # Total crop predictions
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM crop_predictions
        """)
        crop_predictions = cursor.fetchone()["total"]

        # Total disease predictions
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM disease_predictions
        """)
        disease_predictions = cursor.fetchone()["total"]

        # Total predictions
        total_predictions = crop_predictions + disease_predictions

    except Exception:
        app.logger.exception("Admin dashboard statistics error")
        flash("Unable to load admin dashboard statistics.", "error")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return render_template(
        "admin.html",
        total_users=total_users,
        total_predictions=total_predictions,
        crop_predictions=crop_predictions,
        disease_predictions=disease_predictions
    )

# ==================================
# ADMIN USER MANAGEMENT
# ==================================

@app.route("/admin/users")
def admin_users():

    # Check admin authentication
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        flash("Access denied! Admin privileges required.", "error")
        return redirect(url_for("dashboard"))

    connection = None
    cursor = None
    users = []

    # Get search keyword
    search_query = request.args.get("q", "").strip()

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        if search_query:

            cursor.execute("""
                SELECT user_id, name, email, phone, role, created_at
                FROM users
                WHERE name LIKE %s OR email LIKE %s
                ORDER BY user_id DESC
            """, (
                f"%{search_query}%",
                f"%{search_query}%"
            ))

        else:

            cursor.execute("""
                SELECT user_id, name, email, phone, role, created_at
                FROM users
                ORDER BY user_id DESC
            """)

        users = cursor.fetchall()

    except Exception:
        app.logger.exception("Unable to load users")
        flash("Unable to load users.", "error")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return render_template(
        "admin_users.html",
        users=users,
        search_query=search_query
    )

# ==================================
# DELETE USER
# ==================================

@app.route("/admin/users/delete/<int:user_id>", methods=["POST"])
def delete_user(user_id):

    # Check admin authentication
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        flash("Access denied!", "error")
        return redirect(url_for("dashboard"))

    # Prevent admin from deleting their own account
    if user_id == session["user_id"]:
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("admin_users"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        connection.start_transaction()

        # Delete the user's crop prediction history
        cursor.execute("""
            DELETE FROM crop_predictions
            WHERE user_id = %s
        """, (user_id,))

        # Delete the user's disease prediction history
        cursor.execute("""
            DELETE FROM disease_predictions
            WHERE user_id = %s
        """, (user_id,))

        # Delete the user account
        cursor.execute("""
            DELETE FROM users
            WHERE user_id = %s
        """, (user_id,))

        if cursor.rowcount == 0:
            connection.rollback()
            flash("User not found.", "error")
        else:
            connection.commit()
            flash("User deleted successfully.", "success")

    except Exception:
        if connection is not None:
            connection.rollback()

        app.logger.exception("Unable to delete user")
        flash("Unable to delete user. Please check database constraints.", "error")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return redirect(url_for("admin_users"))

# ==================================
# ADMIN PREDICTION MANAGEMENT
# ==================================

@app.route("/admin/predictions")
def admin_predictions():

    # Check admin authentication
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        flash("Access denied! Admin privileges required.", "error")
        return redirect(url_for("dashboard"))

    connection = None
    cursor = None
    predictions = []

    try:
        connection = get_db_connection()

        if connection is None:
            raise RuntimeError("Database connection failed.")

        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                u.name AS user_name,
                'Crop Recommendation' AS prediction_type,
                cp.recommended_crop AS prediction,
                cp.confidence AS confidence,
                cp.created_at AS prediction_date
            FROM crop_predictions cp
            LEFT JOIN users u
                ON cp.user_id = u.user_id

            UNION ALL

            SELECT
                u.name AS user_name,
                'Disease Detection' AS prediction_type,
                dp.predicted_disease AS prediction,
                dp.confidence AS confidence,
                dp.created_at AS prediction_date
            FROM disease_predictions dp
            LEFT JOIN users u
                ON dp.user_id = u.user_id

            ORDER BY prediction_date DESC
        """)

        predictions = cursor.fetchall()

    except Exception:
        app.logger.exception("Unable to load prediction records")
        flash("Unable to load prediction records.", "error")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return render_template(
        "admin_predictions.html",
        predictions=predictions
    )

    # ==================================
# ADMIN DISEASE MANAGEMENT
# ==================================

@app.route("/admin/diseases")
def admin_diseases():

    # Admin authentication
    if "user_id" not in session:
        flash("Please login first.", "error")
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        flash("Access denied! Admin privileges required.", "error")
        return redirect(url_for("dashboard"))

    connection = None
    cursor = None
    diseases = []

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT disease_id, disease_name, crop_name,
                   symptoms, causes, prevention, management
            FROM diseases
            ORDER BY disease_id DESC
        """)

        diseases = cursor.fetchall()

    except Exception:
        app.logger.exception("Unable to load diseases")
        flash("Unable to load disease records.", "error")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return render_template(
        "admin_diseases.html",
        diseases=diseases
    )


# ==================================
# ADD NEW DISEASE
# ==================================

@app.route("/admin/diseases/add", methods=["POST"])
def add_disease():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        flash("Access denied!", "error")
        return redirect(url_for("dashboard"))

    disease_name = request.form.get("disease_name", "").strip()
    crop_name = request.form.get("crop_name", "").strip()
    symptoms = request.form.get("symptoms", "").strip()
    causes = request.form.get("causes", "").strip()
    prevention = request.form.get("prevention", "").strip()
    management = request.form.get("management", "").strip()

    if not all([
        disease_name, crop_name, symptoms,
        causes, prevention, management
    ]):
        flash("Please fill in all disease fields.", "error")
        return redirect(url_for("admin_diseases"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO diseases
            (
                disease_name,
                plant_name,
                crop_name,
                symptoms,
                causes,
                prevention,
                management,
                suggested_treatment
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            disease_name,
            crop_name,
            crop_name,
            symptoms,
            causes,
            prevention,
            management,
            management
        ))

        connection.commit()

        flash("Disease added successfully!", "success")

    except Exception:
        if connection is not None:
            connection.rollback()

        app.logger.exception("Unable to add disease")
        flash("Unable to add disease. Check database columns and constraints.", "error")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return redirect(url_for("admin_diseases"))


# ==================================
# EDIT EXISTING DISEASE
# ==================================

@app.route("/admin/diseases/edit/<int:disease_id>", methods=["GET", "POST"])
def edit_disease(disease_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        flash("Access denied!", "error")
        return redirect(url_for("dashboard"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        if request.method == "POST":

            disease_name = request.form.get("disease_name", "").strip()
            crop_name = request.form.get("crop_name", "").strip()
            symptoms = request.form.get("symptoms", "").strip()
            causes = request.form.get("causes", "").strip()
            prevention = request.form.get("prevention", "").strip()
            management = request.form.get("management", "").strip()

            if not all([
                disease_name, crop_name, symptoms,
                causes, prevention, management
            ]):
                flash("Please fill in all fields.", "error")
            else:
                cursor.execute("""
                    UPDATE diseases
                    SET disease_name = %s,
                        plant_name = %s,
                        crop_name = %s,
                        symptoms = %s,
                        causes = %s,
                        prevention = %s,
                        management = %s,
                        suggested_treatment = %s
                    WHERE disease_id = %s
                """, (
                    disease_name,
                    crop_name,
                    crop_name,
                    symptoms,
                    causes,
                    prevention,
                    management,
                    management,
                    disease_id
                ))

                connection.commit()

                flash("Disease updated successfully!", "success")
                return redirect(url_for("admin_diseases"))

        cursor.execute("""
            SELECT *
            FROM diseases
            WHERE disease_id = %s
        """, (disease_id,))

        disease = cursor.fetchone()

        if disease is None:
            flash("Disease not found.", "error")
            return redirect(url_for("admin_diseases"))

        return render_template(
            "edit_disease.html",
            disease=disease
        )

    except Exception:
        if connection is not None:
            connection.rollback()

        app.logger.exception("Unable to edit disease")
        flash("Unable to process disease information.", "error")
        return redirect(url_for("admin_diseases"))

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# ==================================
# DELETE DISEASE
# ==================================

@app.route("/admin/diseases/delete/<int:disease_id>", methods=["POST"])
def delete_disease(disease_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("role") != "admin":
        flash("Access denied!", "error")
        return redirect(url_for("dashboard"))

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("""
            DELETE FROM diseases
            WHERE disease_id = %s
        """, (disease_id,))

        if cursor.rowcount == 0:
            flash("Disease not found.", "error")
        else:
            connection.commit()
            flash("Disease deleted successfully!", "success")

    except Exception:
        if connection is not None:
            connection.rollback()

        app.logger.exception("Unable to delete disease")
        flash("Unable to delete disease. Check database constraints.", "error")

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()

    return redirect(url_for("admin_diseases"))


# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# ==========================================
# REGISTRATION PAGE
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name", "").strip()

        email = request.form.get("email", "").strip().lower()

        phone = request.form.get("phone", "").strip()

        password = request.form.get("password", "")

        confirm_password = request.form.get("confirm_password", "")

        # Check empty fields

        if not all([
            name,
            email,
            phone,
            password,
            confirm_password
        ]):

            return render_template(
                "register.html",
                error="Please fill in all fields."
            )

        # Check field lengths

        if len(name) > 100 or len(email) > 254 or len(phone) > 20:

            return render_template(
                "register.html",
                error="One or more fields exceed the allowed length."
            )

        # Check password length

        if len(password) < 8:

            return render_template(
                "register.html",
                error="Password must contain at least 8 characters."
            )

        # Check password confirmation

        if password != confirm_password:

            return render_template(
                "register.html",
                error="Passwords do not match."
            )

        # Hash password

        password_hash = generate_password_hash(password)

        # Connect to database

        connection = get_db_connection()

        if connection is None:

            return render_template(
                "register.html",
                error="Database connection failed. Please try again."
            )

        cursor = None

        try:

            cursor = connection.cursor()

            query = """
                INSERT INTO users
                (name, email, phone, password_hash)
                VALUES (%s, %s, %s, %s)
            """

            cursor.execute(
                query,
                (name, email, phone, password_hash)
            )

            connection.commit()

            return render_template(
                "register.html",
                success="Registration successful! You can now log in."
            )

        except Exception as error:

            connection.rollback()

            if getattr(error, "errno", None) == 1062:

                return render_template(
                    "register.html",
                    error="This email is already registered."
                )

            app.logger.exception("Registration failed")

            return render_template(
                "register.html",
                error="Registration failed. Please try again."
            )

        finally:

            if cursor is not None:
                cursor.close()

            connection.close()

    return render_template("register.html")


# ==========================================
# DISPLAY UPLOADED IMAGES
# ==========================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    def validate_crop_inputs(data):

        required_fields = [
            "nitrogen",
            "phosphorus",
            "potassium",
            "temperature",
            "humidity",
            "ph",
            "rainfall"
        ]

    values = {}

    # Check empty values
    for field in required_fields:

        if field not in data or data[field].strip() == "":
            return None, f"{field.capitalize()} cannot be empty."

        # Check numeric values
        try:
            values[field] = float(data[field])

        except ValueError:
            return None, f"{field.capitalize()} must be a valid number."

    # Check negative values
    for field, value in values.items():

        if value < 0:
            return None, f"{field.capitalize()} cannot be negative."

    # Check temperature (Celsius)
    if not 0 <= values["temperature"] <= 50:
        return None, "Temperature must be between 0°C and 50°C."

    # Check humidity
    if not 0 <= values["humidity"] <= 100:
        return None, "Humidity must be between 0 and 100."

    # Check pH
    if not 0 <= values["ph"] <= 14:
        return None, "pH must be between 0 and 14."

    # Check rainfall
    if values["rainfall"] <= 0:
        return None, "Rainfall must be greater than 0."

    return values, None

@app.errorhandler(413)
def file_too_large(error):

    flash("Image size is too large. Please upload an image smaller than 5 MB.", "error")

    return redirect(url_for("disease_detection"))


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)