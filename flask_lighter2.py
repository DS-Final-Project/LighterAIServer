import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split
#import back_imgTextChange as bit

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/percentage', methods=['POST'])
def get_percent():
    param = request.get_json()
    
    searchWord=param['chatWords']

    print(param['chatWords'])
    
    
    #받은 데이터 정제하기_이미지
    #extracted_data = bit.extract_data_from_string(searchWord)
    #formatted_text_list = bit.format_text_by_y_value(extracted_data)
    #print(formatted_text_list)
    searchWord_Img = searchWord.split("\n")
    #받은 데이터 정제하기_카카오톡 데이터
    
    
    
  
    # BERT 토크나이저 로드
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    model.load_state_dict(torch.load('lighter_multibi_finetuned4.pth', map_location=torch.device('cpu')), strict=False)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    def evaluate_and_extract(texts):
        # 전체 대화에 대한 가스라이팅 확률 계산
        whole_text = ' '.join(texts)
        whole_encoded = tokenizer.encode_plus(
            whole_text,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=512,
            return_tensors='pt'
        )

        input_id = whole_encoded['input_ids'].to(device)
        attention_mask = whole_encoded['attention_mask'].to(device)

        model.eval()
        with torch.no_grad():
            whole_output = model(input_ids=input_id, attention_mask=attention_mask)

        whole_logits = whole_output[0]
        whole_prob = torch.nn.functional.softmax(whole_logits, dim=-1)[0][1].item()

        # 각 문장에 대한 가스라이팅 확률 계산
        gaslighting_scores = []
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                return_attention_mask=True,
                pad_to_max_length=True,
                max_length=512,
                return_tensors='pt'
            )

            input_id = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                output = model(input_ids=input_id, attention_mask=attention_mask)

            logits = output[0]
            prob = torch.nn.functional.softmax(logits, dim=-1)[0][1].item()
            gaslighting_scores.append((text, prob))

        # 상위 2개 가스라이팅 확률을 가진 문장 선택
        sorted_scores = sorted(gaslighting_scores, key=lambda x: x[1], reverse=True)[:2]
        top2_texts = [item[0] for item in sorted_scores]

        return whole_prob, top2_texts

    #example_conversation =["어? 프사 바꿨네","이쁘다 ㅎㅎ","고마워","근데 코가 쫌 아쉽다ㅋㅋ 코 때문에 뭔가 부족한 느낌?","코만 성형하면 진짜 예쁠텐데","내가 다 아깝다 나였으면 코하고 존예 소리 들을 듯"]


    #whole_gaslighting_prob, top2_gaslighting_texts = evaluate_and_extract(example_conversation)
    #whole_gaslighting_prob, top2_gaslighting_texts = evaluate_and_extract(formatted_text_list)
    whole_gaslighting_prob, top2_gaslighting_texts = evaluate_and_extract(searchWord_Img)
    
    print(f"전체 대화의 가스라이팅 확률: {whole_gaslighting_prob:.2f}")
    print(f"가스라이팅 정도가 가장 심한 2개의 문장: {top2_gaslighting_texts}")



    # 특정 문장 입력
    #input_sentence = searchWord
    
    
    resultjson = {'resultNum': int((whole_gaslighting_prob)*100),'doubtText1':top2_gaslighting_texts[0],'doubtText2':top2_gaslighting_texts[1]}
    return jsonify(resultjson)
    #return jsonify(param)
    #return 'Hello, World!'
 
if __name__ == "__main__":
    #app.run(debug = False)
    app.run(debug = False, host="0.0.0.0", port=5000)