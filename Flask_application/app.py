from flask import Flask,request,jsonify,render_template
import os
import torch

from utils_bert import get_response,predict_class, get_model_type


app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/handle_message', methods=['POST'])
def handle_message():
    message = request.json['message']
    intents_list = predict_class(message)
    response = get_response(intents_list)
    model_type = get_model_type()
    
    return jsonify({
        'response': response,
        'model_type': model_type
    })


if __name__ == '__main__':
 port = int(os.environ.get("PORT", 5000))  # use Render's assigned port
 app.run(host='0.0.0.0', port=port, debug=True)