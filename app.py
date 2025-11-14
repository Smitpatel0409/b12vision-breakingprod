"""
Flask Application for Vitamin B12 Hand Analysis with Local Storage

This application processes hand images to analyze Vitamin B12 levels by:
1. Receiving uploaded hand images
2. Processing them using MediaPipe and color analysis
3. Storing original and processed images locally
4. Returning B12 status and color analysis results
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from uuid import uuid4
from datetime import datetime
import traceback
from handbissuefix import process_image

# Load environment variables from .env file
load_dotenv()

# Flask application setup
app = Flask(__name__)

# Configuration constants
UPLOAD_FOLDER = "uploads"              # Temporary folder for uploaded images
PROCESSED_FOLDER = "processed"          # Temporary folder for processed images
STORED_FOLDER = "stored_images"         # Permanent storage folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}  # Supported image formats
MAX_FILE_SIZE = 10 * 1024 * 1024       # 10MB maximum file size

# Flask app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['STORED_FOLDER'] = STORED_FOLDER
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create required directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STORED_FOLDER, exist_ok=True)

# Create subdirectories in stored_images for organization
os.makedirs(os.path.join(STORED_FOLDER, 'originals'), exist_ok=True)
os.makedirs(os.path.join(STORED_FOLDER, 'processed'), exist_ok=True)

print("=" * 60)
print("Storage Directories Initialized:")
print(f"  - Upload folder: {UPLOAD_FOLDER}")
print(f"  - Processed folder: {PROCESSED_FOLDER}")
print(f"  - Storage folder: {STORED_FOLDER}")
print("=" * 60)


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the uploaded file
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_to_storage(source_path, folder_type, unique_id, original_filename):
    """
    Copy a file to permanent local storage.

    Args:
        source_path (str): Path of the source file
        folder_type (str): 'originals' or 'processed'
        unique_id (str): Unique identifier for the upload session
        original_filename (str): Original name of the file

    Returns:
        tuple: (storage_path: str, relative_url: str) or (None, None) on error
    """
    try:
        # Create destination path
        destination_dir = os.path.join(app.config['STORED_FOLDER'], folder_type, unique_id)
        os.makedirs(destination_dir, exist_ok=True)
        
        destination_path = os.path.join(destination_dir, original_filename)
        
        # Copy file to storage
        import shutil
        shutil.copy2(source_path, destination_path)
        
        # Generate relative URL for accessing the file
        relative_url = f"/files/{folder_type}/{unique_id}/{original_filename}"
        
        print(f"✓ Saved to storage: {destination_path}")
        print(f"✓ Access URL: {relative_url}")
        
        return destination_path, relative_url
        
    except Exception as e:
        print(f"✗ Error saving to storage: {e}")
        traceback.print_exc()
        return None, None


def cleanup_local_file(file_path):
    """
    Delete a local file after it has been stored permanently.
    This prevents disk space from filling up with temporary files.
    
    Args:
        file_path (str): Path to the file to delete
        
    Returns:
        bool: True if file was deleted successfully, False otherwise
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Cleaned up temporary file: {file_path}")
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


@app.route('/files/<folder_type>/<unique_id>/<filename>')
def serve_file(folder_type, unique_id, filename):
    """
    Serve stored files to clients.
    
    Args:
        folder_type (str): 'originals' or 'processed'
        unique_id (str): Unique identifier for the upload session
        filename (str): Name of the file to serve
        
    Returns:
        File response or 404 error
    """
    try:
        directory = os.path.join(app.config['STORED_FOLDER'], folder_type, unique_id)
        return send_from_directory(directory, filename)
    except Exception as e:
        print(f"✗ Error serving file: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring application status.
    Useful for load balancers and monitoring systems.
    
    Returns:
        JSON response with application health status
    """
    storage_status = "healthy"
    storage_details = {}
    
    try:
        # Check if storage directories are accessible
        for folder in ['originals', 'processed']:
            test_dir = os.path.join(app.config['STORED_FOLDER'], folder)
            if not os.path.exists(test_dir) or not os.access(test_dir, os.W_OK):
                storage_status = "degraded"
                break
        
        storage_details = {
            "stored_folder": app.config['STORED_FOLDER'],
            "writable": os.access(app.config['STORED_FOLDER'], os.W_OK),
            "exists": os.path.exists(app.config['STORED_FOLDER'])
        }
        
    except Exception as e:
        storage_status = f"error: {str(e)}"
    
    return jsonify({
        "status": storage_status,
        "storage": storage_status,
        "storage_details": storage_details,
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
    4. Store both original and processed images permanently
    5. Clean up temporary files
    6. Return analysis results with image URLs
    
    Returns:
        JSON response with:
        - base_image_url: URL of original uploaded image
        - processed_image_url: URL of annotated processed image
        - vitamin_b12_status: "Sufficient" or "Deficient"
        - color_score_diff: Numerical difference in color scores
    """
    try:
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
        unique_filename = f"{timestamp}_{filename}"
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

        # Store original image permanently
        print(f"💾 Storing original image...")
        _, base_image_url = save_to_storage(
            local_upload_path, 
            'originals', 
            unique_id, 
            unique_filename
        )
        
        # Store processed image permanently
        print(f"💾 Storing processed image...")
        _, processed_image_url = save_to_storage(
            processed_path, 
            'processed', 
            unique_id, 
            processed_filename
        )

        # Check if both storage operations were successful
        if not base_image_url or not processed_image_url:
            # Clean up temporary files even if storage failed
            cleanup_local_file(local_upload_path)
            cleanup_local_file(processed_path)
            return jsonify({
                "error": "Failed to store images"
            }), 500

        # Clean up temporary files after successful storage
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
    print(f"Storage Folder: {STORED_FOLDER}")
    print(f"Max File Size: {MAX_FILE_SIZE // (1024 * 1024)}MB")
    print("=" * 60)
    
    app.run(
        debug=True,           # Enable debug mode for development
        host='0.0.0.0',      # Listen on all network interfaces
        port=5002            # Run on port 5002
    )