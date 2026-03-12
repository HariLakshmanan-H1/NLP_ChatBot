import subprocess
import sys
import time
import os
import signal
import atexit
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to control restart loop
shutting_down = False
processes = []

def cleanup(signum=None, frame=None):
    """Cleanup function to terminate all subprocesses"""
    global shutting_down
    if shutting_down:
        return
    
    shutting_down = True
    logger.info("Shutting down all processes...")
    
    for proc in processes:
        if proc and proc.poll() is None:  # If process is still running
            try:
                proc.terminate()
                # Give it a moment to terminate gracefully
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except:
                pass
    
    # Force kill any remaining processes
    if sys.platform != 'win32':
        os.system('pkill -f "uvicorn api:app" 2>/dev/null')
        os.system('pkill -f "streamlit run app.py" 2>/dev/null')
    else:
        os.system('taskkill /F /IM python.exe /FI "WINDOWTITLE eq *uvicorn*" 2>nul')
        os.system('taskkill /F /IM python.exe /FI "WINDOWTITLE eq *streamlit*" 2>nul')
    
    logger.info("All processes terminated")
    # Force exit
    os._exit(0)

# Register cleanup on exit
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)
atexit.register(cleanup)

def run_fastapi(app_dir: str = ".", host: str = "0.0.0.0", port: int = 8000):
    """Launches FastAPI using Uvicorn"""
    logger.info(f"Starting FastAPI backend on {host}:{port}...")
    
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--host", host,
            "--port", str(port),
            "--reload",
            "--log-level", "info"
        ],
        cwd=app_dir
    )
    processes.append(proc)
    return proc

def run_streamlit():
    """Launches Streamlit UI"""
    logger.info("Starting Streamlit UI...")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",  # Add this to prevent browser auto-open
            "--browser.gatherUsageStats", "false"
        ]
    )
    processes.append(proc)
    return proc

def check_ollama() -> bool:
    """Check if Ollama is running"""
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def print_startup_message(ollama_running: bool):
    """Print colorful startup message"""
    print("\n" + "="*70)
    print("🚀 NCO 2015 AI Career Advisor".center(70))
    print("="*70)
    
    status = "✅ Running" if ollama_running else "❌ Not detected"
    print(f"Ollama: {status}")
    
    print("\n📡 Services:")
    print(f"   • Backend API:  http://localhost:8000")
    print(f"   • API Docs:     http://localhost:8000/docs")
    print(f"   • Streamlit UI: http://localhost:8501")
    
    print("\n📁 Data:")
    print("   • Using: ./data/nco_2015.pkl")
    
    print("\n⌨️  Commands:")
    print("   • Press Ctrl+C ONCE to stop all services")
    print("   • If services don't stop, run: pkill -f 'uvicorn|streamlit'")
    print("="*70 + "\n")

def main():
    """Main function"""
    global shutting_down
    
    # Check if Ollama is running
    ollama_running = check_ollama()
    
    # Print startup message
    print_startup_message(ollama_running)
    
    if not ollama_running:
        logger.warning("Ollama is not running. LLM features will not work.")
        logger.info("Start Ollama with: ollama serve")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            logger.info("Exiting...")
            return
    
    # Start FastAPI
    api_process = run_fastapi()
    time.sleep(3)
    
    # Start Streamlit
    ui_process = run_streamlit()
    
    try:
        # Keep the script running
        while not shutting_down:
            time.sleep(1)
            
            # Check if processes are still running, but DON'T auto-restart
            if api_process.poll() is not None and not shutting_down:
                logger.error("FastAPI process died unexpectedly")
                logger.info("Please restart the application")
                cleanup()
                break
            
            if ui_process.poll() is not None and not shutting_down:
                logger.error("Streamlit process died unexpectedly")
                logger.info("Please restart the application")
                cleanup()
                break
                
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        cleanup()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        cleanup()

if __name__ == "__main__":
    main()