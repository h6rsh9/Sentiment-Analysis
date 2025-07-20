from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import threading
import time

app = Flask(__name__)
CORS(app)

def start_gradio():
    subprocess.run(['python', 'app.py'])

@app.route('/run-script', methods=['POST'])
def run_script():
    try:
        print("Starting Gradio app...")
        threading.Thread(target=start_gradio).start()
        time.sleep(5)  # Increase the delay to ensure Gradio has started
        print("Gradio app should be running now.")
        return jsonify({'status': 'success', 'url': 'http://127.0.0.1:7860'}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

