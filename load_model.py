# load_model.py - Script to download the LFM2 model during build
import os
import subprocess
import sys

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "./model"
MODEL_NAME = "LFM2-1.2B-RAG"
QUANTIZATION = "Q4_0"

def main():
    """Download and prepare the LFM2 model"""
    
    print("=" * 60)
    print("🚀 LFM2 Model Download Script")
    print("=" * 60)
    
    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        print(f"📁 Creating model directory: {MODEL_PATH}")
        os.makedirs(MODEL_PATH, exist_ok=True)
    
    print(f"📦 Model: {MODEL_NAME}")
    print(f"⚙️  Quantization: {QUANTIZATION}")
    print(f"💾 Save path: {MODEL_PATH}")
    print()
    
    try:
        # Install leap-bundle if not already installed
        print("🔧 Installing leap-bundle...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "leap-bundle"],
            check=True,
            capture_output=True
        )
        print("✅ leap-bundle installed")
        print()
        
        # Download the model using leap-bundle
        print(f"⬇️  Downloading {MODEL_NAME} with {QUANTIZATION} quantization...")
        print("   This may take a few minutes...")
        
        # Run leap-bundle download command
        result = subprocess.run(
            [
                "leap-bundle",
                "download",
                MODEL_NAME,
                f"--quantization={QUANTIZATION}",
                f"--output-dir={MODEL_PATH}"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("✅ Model downloaded successfully!")
        print()
        
        # List downloaded files
        print("📋 Downloaded files:")
        for file in os.listdir(MODEL_PATH):
            file_path = os.path.join(MODEL_PATH, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   - {file} ({size_mb:.2f} MB)")
        
        print()
        print("=" * 60)
        print("✅ Model setup complete!")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during model download: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
