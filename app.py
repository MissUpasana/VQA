from flask import Flask, request, jsonify, render_template
import os
import json
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image



app = Flask(__name__)    

# Load the VILT model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        # Get user-provided text
        text = request.form['text']
        
        print('User question is:',text)

        # Get the uploaded image file
        image_file = request.files['image']
        
        # uncomment below two lines of code if you want to save uploaded image on server
        #file_path = os.path.join(os.getcwd(), image_file.filename)
        #image_file.save(file_path)
        
        image = Image.open(image_file)

        # Prepare inputs
        encoding = processor(image, text, return_tensors="pt")

        # Forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        predicted_answer = model.config.id2label[idx]

        # Return response as JSON
        response = {'predicted_answer': predicted_answer}
        return jsonify(response)

    except Exception as e:
        error_response = {'error': str(e)}
        return jsonify(error_response), 400
        
         # Forward pass
         
        # uncomment below code if you want to return top 3 predicted answers with their probability_percentage
        #outputs = model(**encoding)
        #logits = outputs.logits
        #predicted_indices = logits.argsort(-1, descending=True).squeeze()

        #top_predicted_answers = []
        #for idx in predicted_indices[:3]:
            #predicted_answer = model.config.id2label[idx.item()]
            #probability = logits[0][idx].item()
            #probability_percentage = probability * 100
            #top_predicted_answers.append({'answer': predicted_answer, 'probability': probability_percentage})

        # Return response as JSON
        #response = {'top_predicted_answers': top_predicted_answers}
        #return jsonify(response)

    #except Exception as e:
        #error_response = {'error': str(e)}
        #return jsonify(error_response), 400
    
    

#  for local
if __name__ == "__main__":
    app.run(debug=True)

#  for cloud
# if __name__ == "__main__":
#     app.run(host = '0.0.0.0',port=8080)