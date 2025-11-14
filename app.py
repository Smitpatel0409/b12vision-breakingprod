"""
Flask Application for Vitamin B12 Hand Analysis with MinIO Storage

This application processes hand images to analyze Vitamin B12 levels by:
1. Receiving uploaded hand images
2. Processing them using MediaPipe and color analysis
3. Storing original and processed images in MinIO
4. Returning B12 status and color analysis results
"""

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from uuid import uuid4
from datetime import datetime
from datetime import timedelta
import traceback
from handbissuefix import process_image
import json

# Load environment variables from .env file
load_dotenv()

# Flask application setup
app = Flask(__name__)

# Configuration constants
UPLOAD_FOLDER = "uploads"              # Temporary folder for uploaded images
PROCESSED_FOLDER = "processed"          # Temporary folder for processed images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}  # Supported image formats
MAX_FILE_SIZE = 10 * 1024 * 1024       # 10MB maximum file size

# Flask app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create required directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# MinIO configuration from environment variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "10.11.7.142:9000")  # External MinIO server
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")    # MinIO access key
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123") # MinIO secret key
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "b12-analysis")          # Bucket name for storage
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"  # Use HTTPS?
# Public URL should be accessible from client browsers
MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL", f"http://10.11.7.142:9000")
# Use public read policy instead of presigned URLs
USE_PUBLIC_POLICY = os.getenv("USE_PUBLIC_POLICY", "True").lower() == "true"

# Initialize MinIO client and ensure bucket exists
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )
    
    # Test connection by checking if bucket exists
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        print(f"✓ Created MinIO bucket: {MINIO_BUCKET}")
    else:
        print(f"✓ MinIO bucket exists: {MINIO_BUCKET}")
    
    # Set bucket policy to allow public read access if enabled
    if USE_PUBLIC_POLICY:
        try:
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{MINIO_BUCKET}/*"]
                    }
                ]
            }
            minio_client.set_bucket_policy(MINIO_BUCKET, json.dumps(policy))
            print(f"✓ Set public read policy for bucket: {MINIO_BUCKET}")
        except Exception as e:
            print(f"⚠ Warning: Could not set bucket policy: {e}")
            print("  Falling back to presigned URLs")
            USE_PUBLIC_POLICY = False
    
    print(f"✓ Successfully connected to MinIO at {MINIO_ENDPOINT}")
        
except Exception as e:
    print(f"✗ MinIO initialization error: {e}")
    print(f"  - Endpoint: {MINIO_ENDPOINT}")
    print(f"  - Access Key: {MINIO_ACCESS_KEY}")
    print(f"  - Secure: {MINIO_SECURE}")
    print("Application will continue but file uploads will fail")
    traceback.print_exc()
    minio_client = None


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_to_minio(file_path, object_name, expiry_seconds=86400):
    """
    Upload a file to MinIO and return a URL.

    Args:
        file_path (str): Local path of the file
        object_name (str): Object name in MinIO bucket
        expiry_seconds (int): Expiry time for presigned URL (default 24 hours)

    Returns:
        str: URL for accessing the file (public or presigned)
    """
    if not minio_client:
        print("✗ MinIO client not initialized")
        return None

    try:
        # Get file size
        file_size = os.path.getsize(file_path)

        # Detect correct MIME type
        extension = file_path.lower().split('.')[-1]
        content_type_map = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'webp': 'image/webp',
            'bmp': 'image/bmp'
        }
        content_type = content_type_map.get(extension, 'image/jpeg')

        # Upload file to MinIO with the right content type
        minio_client.fput_object(
            MINIO_BUCKET,
            object_name,
            file_path,
            content_type=content_type
        )

        print(f"✓ Uploaded to MinIO: {object_name} ({file_size} bytes)")

        # Return public URL if policy is enabled, otherwise presigned URL
        if USE_PUBLIC_POLICY:
            # Direct public URL
            url = f"{MINIO_PUBLIC_URL}/{MINIO_BUCKET}/{object_name}"
            print(f"✓ Public URL: {url}")
            return url
        else:
            # Generate pre-signed URL
            presigned_url = minio_client.presigned_get_object(
                MINIO_BUCKET,
                object_name,
                expires=timedelta(seconds=expiry_seconds)
            )
            print(f"✓ Presigned URL valid for {expiry_seconds} seconds")
            return presigned_url

    except S3Error as e:
        print(f"✗ MinIO S3 error during upload: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"✗ Unexpected error during MinIO upload: {e}")
        traceback.print_exc()
        return None


def cleanup_local_file(file_path):
    """
    Delete a local file after it has been uploaded to MinIO.
    This prevents disk space from filling up with temporary files.
    
    Args:
        file_path (str): Path to the file to delete
        
    Returns:
        bool: True if file was deleted successfully, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Cleaned up local file: {file_path}")
            return True
        else:
            print(f"⚠ File not found for cleanup: {file_path}")
            return False
    except PermissionError as e:
        print(f"✗ Permission denied deleting {file_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error deleting {file_path}: {e}")
        return False


def validate_image_file(file):
    """
    Validate the uploaded file before processing.
    Checks for file existence, proper filename, and allowed file type.
    
    Args:
        file: FileStorage object from Flask request
        
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    # Check if file exists and has a filename
    if not file or file.filename == "":
        return False, "No file selected"
    
    # Check if file extension is allowed
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # File size is automatically validated by Flask's MAX_CONTENT_LENGTH
    return True, None


@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Handle requests where the uploaded file exceeds MAX_CONTENT_LENGTH.
    
    Returns:
        JSON response with error message and 413 status code
    """
    return jsonify({
        "error": f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB"
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    """
    Handle internal server errors gracefully.
    
    Returns:
        JSON response with error message and 500 status code
    """
    return jsonify({
        "error": "Internal server error. Please try again later."
    }), 500


@app.route('/')
def index():
    """
    Render the main application page.
    
    Returns:
        Rendered HTML template
    """
    return render_template("index.html")


@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring application status.
    Useful for load balancers and monitoring systems.
    
    Returns:
        JSON response with application health status
    """
    minio_status = "disconnected"
    minio_details = {}
    
    if minio_client:
        try:
            # Try to list buckets to verify connection
            minio_client.bucket_exists(MINIO_BUCKET)
            minio_status = "connected"
            minio_details = {
                "endpoint": MINIO_ENDPOINT,
                "bucket": MINIO_BUCKET,
                "secure": MINIO_SECURE,
                "public_policy": USE_PUBLIC_POLICY
            }
        except Exception as e:
            minio_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy" if minio_status == "connected" else "degraded",
        "minio": minio_status,
        "minio_details": minio_details,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Main endpoint for uploading and processing hand images.
    
    Process flow:
    1. Validate uploaded file
    2. Save file locally with unique filename
    3. Process image for B12 analysis (detect hand, analyze color)
    4. Upload both original and processed images to MinIO
    5. Clean up local temporary files
    6. Return analysis results with image URLs
    
    Returns:
        JSON response with:
        - base_image_url: URL of original uploaded image in MinIO
        - processed_image_url: URL of annotated processed image in MinIO
        - vitamin_b12_status: "Sufficient" or "Deficient"
        - color_score_diff: Numerical difference in color scores
    """
    try:
        # Check if MinIO is available
        if not minio_client:
            return jsonify({
                "error": "Storage service unavailable. Please contact administrator."
            }), 503
        
        # Validate that a file was uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        
        # Validate file properties
        is_valid, error_message = validate_image_file(file)
        if not is_valid:
            return jsonify({"error": error_message}), 400

        # Generate unique filename to prevent collisions
        filename = secure_filename(file.filename)  # Sanitize filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid4())
        unique_filename = f"{timestamp}_{unique_id}_{filename}"
        local_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save uploaded file to local temporary storage
        try:
            file.save(local_upload_path)
            print(f"✓ Saved uploaded file: {local_upload_path}")
        except Exception as e:
            print(f"✗ Failed to save uploaded file: {e}")
            return jsonify({"error": "Failed to save uploaded file"}), 500

        # Process the image for B12 analysis
        print(f"⚙ Processing image: {local_upload_path}")
        processed_path, vitamin_b12_status, color_score_diff = process_image(
            local_upload_path, 
            app.config['PROCESSED_FOLDER']
        )

        # Check if processing was successful
        if processed_path is None:
            # Clean up uploaded file if processing failed
            cleanup_local_file(local_upload_path)
            return jsonify({
                "error": vitamin_b12_status  # Error message is returned in status field
            }), 400

        # Get filename of processed image
        processed_filename = os.path.basename(processed_path)

        # Define object paths in MinIO bucket (organized by unique_id)
        base_s3_key = f"b12_uploads/{unique_id}/{unique_filename}"
        processed_s3_key = f"b12_uploads/{unique_id}/processed_{processed_filename}"

        # Upload original image to MinIO
        print(f"⬆ Uploading original image to MinIO...")
        base_image_url = upload_to_minio(local_upload_path, base_s3_key)
        
        # Upload processed image to MinIO
        print(f"⬆ Uploading processed image to MinIO...")
        processed_image_url = upload_to_minio(processed_path, processed_s3_key)

        # Check if both uploads were successful
        if not base_image_url or not processed_image_url:
            # Clean up local files even if upload failed
            cleanup_local_file(local_upload_path)
            cleanup_local_file(processed_path)
            return jsonify({
                "error": "Failed to upload images to storage"
            }), 500

        # Clean up local temporary files after successful upload
        cleanup_local_file(local_upload_path)
        cleanup_local_file(processed_path)

        # Return successful response with all analysis data
        print(f"✓ Processing complete: {vitamin_b12_status} (diff: {color_score_diff})")
        return jsonify({
            "base_image_url": base_image_url,
            "processed_image_url": processed_image_url,
            "vitamin_b12_status": vitamin_b12_status,
            "color_score_diff": color_score_diff,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        # Catch any unexpected errors
        print(f"✗ Unexpected error in upload endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "An unexpected error occurred during processing"
        }), 500


if __name__ == "__main__":
    """
    Run the Flask development server.
    In production, use a WSGI server like Gunicorn or uWSGI.
    """
    print("=" * 60)
    print("Starting Vitamin B12 Analysis Application")
    print(f"MinIO Endpoint: {MINIO_ENDPOINT}")
    print(f"MinIO Public URL: {MINIO_PUBLIC_URL}")
    print(f"MinIO Bucket: {MINIO_BUCKET}")
    print(f"Use Public Policy: {USE_PUBLIC_POLICY}")
    print(f"Max File Size: {MAX_FILE_SIZE // (1024 * 1024)}MB")
    print("=" * 60)
    
    app.run(
        debug=True,           # Enable debug mode for development
        host='0.0.0.0',      # Listen on all network interfaces
        port=5002            # Run on port 5002
    )