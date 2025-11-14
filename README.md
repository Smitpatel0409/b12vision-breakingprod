# Vitamin B12 Hand Analysis Application (https://b12vision-breakingprod.onrender.com/)

A Flask-based application that analyzes hand images to determine Vitamin B12 status using computer vision and color analysis techniques.

## 🎯 Features

- **Hand Detection & Orientation**: Automatically detects hands using MediaPipe and corrects orientation
- **Color Analysis**: Compares nail bed and finger joint colors against reference shades
- **B12 Status Detection**: Determines if vitamin B12 levels are sufficient or deficient
- **MinIO Storage**: Stores original and processed images in MinIO object storage
- **Lighting Validation**: Ensures image quality is sufficient for accurate analysis
- **Left Hand Verification**: Requires left hand for consistent analysis

## 📋 Prerequisites

- Docker and Docker Compose (recommended)
- OR Python 3.10+ with pip
- MinIO server (included in Docker Compose setup)

## 🚀 Quick Start with Docker Compose

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Create HSV Values File

Create a file named `hsv_values.txt` with reference shade values (one per line):

```
(360°, 100%, 100%)
(180°, 50%, 75%)
(120°, 60%, 80%)
...
```

### 3. Start the Services

```bash
# Start MinIO and Flask app
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Access the Application

- **Flask App**: http://localhost:5002
- **MinIO Console**: http://localhost:9001
  - Username: `minioadmin`
  - Password: `minioadmin123`

### 5. Test the Health Check

```bash
curl http://localhost:5002/health
```

## 🔧 Manual Setup (Without Docker)

### 1. Install MinIO

**On Linux/Mac:**
```bash
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server /data --console-address ":9001"
```

**On Windows:**
Download from https://min.io/download and run:
```cmd
minio.exe server C:\data --console-address ":9001"
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Edit `.env` with your MinIO credentials.

### 4. Create HSV Values File

Create `hsv_values.txt` with your reference shade values.

### 5. Run the Application

```bash
python app.py
```

## 📁 Project Structure

```
.
├── app.py                  # Main Flask application
├── handbissuefix.py        # Image processing and analysis
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Multi-container orchestration
├── .env.example           # Environment variables template
├── hsv_values.txt         # Reference shade values
├── templates/             # HTML templates
│   └── index.html
├── uploads/               # Temporary upload folder
└── processed/             # Temporary processed images folder
```

## 🔌 API Endpoints

### `POST /upload`

Upload and analyze a hand image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "base_image_url": "http://localhost:9000/b12-analysis/grim_uploads/uuid/original.jpg",
  "processed_image_url": "http://localhost:9000/b12-analysis/grim_uploads/uuid/processed.png",
  "vitamin_b12_status": "Sufficient",
  "color_score_diff": 1.2,
  "timestamp": "2024-01-01T12:00:00"
}
```

**Error Response:**
```json
{
  "error": "No hand detected. Please provide an image with a visible left hand."
}
```

### `GET /health`

Check application health status.

**Response:**
```json
{
  "status": "healthy",
  "minio": "connected",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🎨 Image Requirements

For best results, ensure your hand image meets these criteria:

- ✅ **Left hand only** (palm facing camera, fingers pointing up)
- ✅ **Good lighting** (bright, even, natural light)
- ✅ **Clear visibility** (all fingers visible, no obstructions)
- ✅ **Proper focus** (hand is in focus, not blurry)
- ✅ **Supported formats**: PNG, JPG, JPEG, BMP, WEBP
- ✅ **File size**: Under 10MB

### Common Rejection Reasons

- ❌ Right hand detected (use left hand)
- ❌ Image too dark (increase lighting)
- ❌ Image too harsh (use softer, natural light)
- ❌ No hand detected (ensure hand is clearly visible)
- ❌ File too large (compress image or use lower resolution)

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MINIO_ENDPOINT` | MinIO server address | `localhost:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key | - |
| `MINIO_SECRET_KEY` | MinIO secret key | - |
| `MINIO_BUCKET` | Bucket name | `b12-analysis` |
| `MINIO_SECURE` | Use HTTPS | `false` |
| `MINIO_PUBLIC_URL` | Public URL for file access | `http://localhost:9000` |

### Docker Compose Configuration

To change MinIO credentials, edit `docker-compose.yml`:

```yaml
services:
  minio:
    environment:
      MINIO_ROOT_USER: your_username
      MINIO_ROOT_PASSWORD: your_secure_password
  
  app:
    environment:
      MINIO_ACCESS_KEY: your_username
      MINIO_SECRET_KEY: your_secure_password
```

## 🐛 Troubleshooting

### MinIO Connection Failed

```bash
# Check if MinIO is running
docker-compose ps

# View MinIO logs
docker-compose logs minio

# Restart MinIO
docker-compose restart minio
```

### Application Won't Start

```bash
# Check application logs
docker-compose logs app

# Rebuild container
docker-compose up -d --build

# Check if port 5002 is available
netstat -an | grep 5002
```

### HSV File Not Found

Ensure `hsv_values.txt` exists in the project root:

```bash
ls -la hsv_values.txt
```

### Images Not Uploading

1. Check MinIO credentials in `.env`
2. Verify bucket exists in MinIO Console
3. Check application logs for specific errors

## 📊 How It Works

1. **Image Upload**: Client uploads hand image via `/upload` endpoint
2. **Validation**: System checks file format, size, and lighting quality
3. **Hand Detection**: MediaPipe detects hand landmarks
4. **Orientation Correction**: Image is rotated to standard orientation (fingers up)
5. **Handedness Check**: Verifies it's a left hand
6. **ROI Extraction**: Extracts two regions:
   - Area A: Nail bed region (between DIP and PIP joints)
   - Area B: Finger joint region (at PIP joint)
7. **Color Analysis**: Compares ROI colors against reference shades
8. **Status Determination**: Calculates color difference; >1.49 = Deficient
9. **Storage**: Uploads original and annotated images to MinIO
10. **Response**: Returns URLs and analysis results to client

## 🔒 Security Considerations

### For Production Deployment:

1. **Change default MinIO credentials**
2. **Enable HTTPS** (set `MINIO_SECURE=true`)
3. **Use environment-specific `.env` files**
4. **Set up bucket policies** for public/private access
5. **Implement authentication** for Flask endpoints
6. **Add rate limiting** to prevent abuse
7. **Use reverse proxy** (nginx) for SSL termination
8. **Regular security updates** for dependencies

## 📝 License

[Your License Here]

## 👥 Contributors

[Your Name/Team]

## 📧 Support

For issues or questions, please open an issue on GitHub or contact [your-email@example.com]# b12vision-breakingprod
