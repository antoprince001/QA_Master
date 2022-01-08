from flask import Flask,render_template,request,jsonify

import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

app = Flask(__name__)

def question_answer(question, text):
    
    #tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)    #number of tokens in segment A (question)
    num_seg_a = sep_idx+1    #number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s for segment embeddings
    segment_ids = [0]*num_seg_a + [1]*num_seg_b    
    
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)    
    
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
                
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
    return "Predicted answer : " +answer.capitalize()


@app.route('/')
def home():
        return render_template('index.html',text="")

@app.route('/qafinder',methods=['post'])
def answer_master():
    question = request.form['question']
    text = request.form['text']
    answer = question_answer(question, text)
    return render_template('index.html',answer=answer,text=text)


if __name__ == "__main__":
    
    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    app.run(threaded=True, port=5000,debug=True)



