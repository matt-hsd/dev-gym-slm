start_server() {
    echo "Starting the server..."
    python -m uvicorn inference:app --reload
}

stop_server() {
    echo "Stopping the server..."
    pkill -f "uvicorn inference:app"
}

train_feedback_classifier() {
    echo "Training feedback classifier..."
    python training/feedback-classifier.py --speed
}

activate_env() {
    echo "Activating virtual environment..."
    deactivate
    python3 -m venv .venv
    source .venv/bin/activate
}

install_dependencies() {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

test_feedback_classifier_0() {
    echo "Testing feedback classifier..."
    curl -X POST "http://localhost:8000/classify/feedback" \
      -H "Content-Type: application/json" \
      -d '{"text": "Rider needed help with seatbelt, driver assisted."}'
}

test_feedback_classifier_3() {
    echo "Testing feedback classifier..."
    curl -X POST "http://localhost:8000/classify/feedback" \
      -H "Content-Type: application/json" \
      -d '{"text": "Driver eating hamburger while driving."}'
}

test_feedback_classifier_5() {
    echo "Testing feedback classifier..."
    curl -X POST "http://localhost:8000/classify/feedback" \
      -H "Content-Type: application/json" \
      -d '{"text": "Rider having medical emergency, driver called 911."}'
}
