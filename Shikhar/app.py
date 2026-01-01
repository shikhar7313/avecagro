from flask import Flask, request, jsonify
from flask_cors import CORS
from chat import get_response, models
import logging
 
app = Flask(__name__)
CORS(app)
 
@app.route("/welcome", methods=["GET"])
def welcome():
    welcome_messages = [
        "Welcome to the chatbot! How can I assist you today?",
        "You can ask me about the weather.",
        "Feel free to say 'hello' or ask me anything else!",
        "I'm here to help you with your queries."
    ]
    return jsonify({"messages": welcome_messages})
 
 
@app.route("/predict", methods=["POST"])
def predict():
    logging.info("Received request: %s", request.json)
    data = request.get_json()
 
    # Validate input
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
 
    text = data['message']
 
    if not text or not text.strip():
        return jsonify({'error': 'Message cannot be empty'}), 400
 
    # Call your model
    try:
        responses = get_response(text, models)
        if isinstance(responses, list):
            response_text = "\n".join(responses)
        else:
            response_text = str(responses)
 
        logging.info('Model response: %s', response_text)
 
    except Exception as e:
        logging.error('Error in get_response: %s', e)
        return jsonify({'error': 'Model error'}), 500
 
    # Send back plain response
    return jsonify({'answer': response_text})
 
 
if __name__ == "__main__":
    app.run(debug=True, port=5001)