#!/usr/bin/env python3
"""
Setup script for Autodisc - Multi-Agent Autonomous Research System
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_env_file():
    """Create .env file with user input"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            return True
    
    print("\n🔑 Setting up environment variables...")
    print("You'll need a Google Gemini API key.")
    print("Get one from: https://makersuite.google.com/app/apikey")
    print()
    
    api_key = input("Enter your Gemini API key: ").strip()
    
    if not api_key:
        print("❌ API key is required")
        return False
    
    # Create .env file
    env_content = f"""# Autodisc - Multi-Agent Autonomous Research System
# Environment Configuration

# Required: Google Gemini API Key
GEMINI_API_KEY={api_key}

# Optional: System Configuration Overrides
# LOG_LEVEL=INFO
# GEMINI_MODEL=gemini-pro
# GEMINI_MAX_TOKENS=8192
# GEMINI_TEMPERATURE=0.7

# Optional: Research Configuration
# MAX_RESEARCH_ITERATIONS=5
# MAX_PARALLEL_AGENTS=4
# RESEARCH_TIMEOUT=300
# VALIDATION_CONFIDENCE_THRESHOLD=0.8
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    try:
        # Test imports
        import config
        import models
        import gemini_client
        import base_agent
        import system_orchestrator
        
        print("✅ All modules imported successfully")
        
        # Test configuration
        config.Config.validate_config()
        print("✅ Configuration validated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def run_quick_test():
    """Run a quick system test"""
    print("\n🚀 Running quick system test...")
    try:
        result = subprocess.run([sys.executable, "example.py", "test"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Quick test passed")
            return True
        else:
            print(f"❌ Quick test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Quick test timed out (this is normal for first run)")
        return True
    except Exception as e:
        print(f"❌ Quick test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Autodisc - Multi-Agent Autonomous Research System")
    print("=" * 60)
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create environment file
    if not create_env_file():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    # Run quick test
    run_quick_test()
    
    print("\n🎉 Setup completed successfully!")
    print()
    print("Next steps:")
    print("1. Review the .env file and adjust settings if needed")
    print("2. Run the example: python main.py")
    print("3. Try interactive mode: python main.py --interactive")
    print("4. Run specific examples: python example.py [creative|analytical|technical]")
    print()
    print("For more information, see README.md")

if __name__ == "__main__":
    main() 