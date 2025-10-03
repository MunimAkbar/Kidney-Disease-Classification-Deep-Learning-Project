from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import os
import base64
from io import BytesIO
from PIL import Image
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        # Import your prediction pipeline here
        try:
            from cnnClassifier.pipeline.prediction import PredictionPipeline
            self.classifier = PredictionPipeline(self.filename)
            logger.info("✅ Prediction pipeline loaded successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Prediction pipeline not available: {e}")
            self.classifier = None

# Initialize the client app
clApp = ClientApp()

def validate_image(base64_string):
    """Validate if the provided string is a valid base64 image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Add padding if necessary
        padding = 4 - len(base64_string) % 4
        if padding != 4:
            base64_string += '=' * padding
        
        # Try to decode and validate as image
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        image.verify()  # Verify it's a valid image
        
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image data: {str(e)}"

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """Serve the HTML interface"""
    try:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>DeepVision AI - Object Classification</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; text-align: center; }
                .status { background: #27ae60; color: white; padding: 10px; border-radius: 5px; text-align: center; }
                .endpoints { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 DeepVision AI</h1>
                <p class="status">✅ API Server is Running</p>
                <div class="endpoints">
                    <h3>Available Endpoints:</h3>
                    <ul>
                        <li><strong>GET /</strong> - This page</li>
                        <li><strong>GET /health</strong> - Health check</li>
                        <li><strong>POST /predict</strong> - Image classification</li>
                        <li><strong>POST /batch-predict</strong> - Batch image classification</li>
                        <li><strong>POST /train</strong> - Train model</li>
                    </ul>
                </div>
                <p>Use the frontend interface or send POST requests to /predict with base64 image data.</p>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        return jsonify({
            "message": "DeepVision AI Object Classification API",
            "version": "2.0.0",
            "status": "operational",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "batch_predict": "/batch-predict",
                "train": "/train"
            }
        })

@app.route("/health", methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "DeepVision AI",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_loaded": clApp.classifier is not None
    })

@app.route("/train", methods=['GET','POST'])
@cross_origin()
def trainRoute():
    try:
        os.system("python main.py")
        return jsonify({
            "success": True,
            "message": "Training completed successfully!",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        # Check if model is loaded
        if clApp.classifier is None:
            return jsonify({
                "success": False,
                "error": "Prediction model not available. Please check if the model is properly loaded."
            }), 503

        # Check if request contains JSON data
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Request must be JSON"
            }), 400
        
        data = request.get_json()
        
        # Check if image data is present
        if 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data provided. Please include 'image' field in request."
            }), 400
        
        image_base64 = data['image']
        
        if not image_base64 or len(image_base64) < 100:
            return jsonify({
                "success": False,
                "error": "Image data too short or empty"
            }), 400
        
        # Validate image
        is_valid, validation_message = validate_image(image_base64)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": validation_message
            }), 400
        
        # Decode and process image
        from cnnClassifier.utils.common import decodeImage
        decodeImage(image_base64, clApp.filename)
        
        # Get prediction
        logger.info("Processing image for classification...")
        result = clApp.classifier.predict()
        
        # Enhanced response format
        response = {
            "success": True,
            "message": "Image analyzed successfully",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_version": "1.0.0",
            "data": result[0] if isinstance(result, list) and len(result) > 0 else result
        }
        
        # Add processed image if available
        if isinstance(result, list) and len(result) > 1 and result[1]:
            response["processed_image"] = result[1]
        
        logger.info("Prediction completed successfully")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

# Demo endpoint for testing without actual model
@app.route("/demo-predict", methods=['POST'])
@cross_origin()
def demo_predict():
    """Demo endpoint that returns sample predictions without requiring the actual model"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image data provided"
            }), 400
        
        # Return sample predictions
        sample_predictions = {
            "predictions": [
                {"class": "Healthy", "confidence": 0.89},
                {"class": "Diseased", "confidence": 0.11},
                {"class": "Unknown", "confidence": 0.05}
            ],
            "top_prediction": "Healthy",
            "confidence": 0.89
        }
        
        return jsonify({
            "success": True,
            "message": "Demo prediction completed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": sample_predictions,
            "note": "This is a demo response. Real model is not loaded."
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint not found. Available endpoints: /, /health, /predict, /demo-predict, /train"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == "__main__":
    logger.info("🚀 Starting DeepVision AI Flask Server...")
    logger.info("📱 Access the application at: http://127.0.0.1:8080")
    logger.info("🔧 Or: http://localhost:8080")
    app.run(host='127.0.0.1', port=8080, debug=True)