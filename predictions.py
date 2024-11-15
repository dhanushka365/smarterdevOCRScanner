import numpy as np
import pandas as pd
import cv2
import pytesseract
import spacy
import re
import string
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load NER model
model_ner = spacy.load('./output/model-best/')

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('', '', whitespace)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt)
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
    return str(removepunctuation)

# Class for grouping labels
class GroupGen:
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id

# Parsing function
def parser(text, label):
    text = text.strip()  # General clean-up for all labels
    
    if label in ['PAYMENTBYERPHONE', 'SHIPPINGPHONE']:
        text = re.sub(r'\D', '', text)
    elif label == 'PAYMENTBYEREMAIL':
        text = re.sub(r'[^A-Za-z0-9@_.\-]', '', text.lower())
    elif label == 'WEB':
        text = re.sub(r'[^A-Za-z0-9:/.%#\-]', '', text.lower())
    elif label in ['PAYMENTBYERNAME', 'SHIPPINGNAME', 'PAYMENTCOMPANYNAME']:
        text = re.sub(r'[^a-zA-Z ]', '', text).title()
    elif label in ['FAULTDETAIL', 'INSTRUCTIONNOTES']:
        text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text).strip() 
    elif label in ['PAYMENTCOMPANYNAME', 'PAYMENTBYERADDRESS', 'SHIPPINGSTREET']:
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text).title()
    elif label in ['REQUIREDDATE', 'ORDERDATE']:
        try:
            text = datetime.strptime(text, '%d/%m/%Y').strftime('%Y-%m-%d')
        except ValueError:
            pass
    return text

grp_gen = GroupGen()

# Main function for predictions and tagging
def getPredictions(image):
    tessData = pytesseract.image_to_data(image)
    tessList = list(map(lambda x: x.split('\t'), tessData.split('\n')))
    df = pd.DataFrame(tessList[1:], columns=tessList[0])
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(cleanText)

    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])
    print("OCR Content:", content)

    doc = model_ner(content)
    docjson = doc.to_json()
    doc_text = docjson['text']
    datafram_tokens = pd.DataFrame(docjson['tokens'])
    datafram_tokens['token'] = datafram_tokens[['start', 'end']].apply(lambda x: doc_text[x[0]:x[1]], axis=1)

    right_table = pd.DataFrame(docjson['ents'])[['start', 'label']]
    datafram_tokens = pd.merge(datafram_tokens, right_table, how='left', on='start')
    datafram_tokens.fillna('O', inplace=True)

    df_clean['end'] = df_clean['text'].apply(lambda x: len(x) + 1).cumsum() - 1
    df_clean['start'] = df_clean[['text', 'end']].apply(lambda x: x[1] - len(x[0]), axis=1)
    dataframe_info = pd.merge(df_clean, datafram_tokens[['start', 'token', 'label']], how='inner', on='start')

    bb_df = dataframe_info.query("label != 'O' ")
    img = image.copy()
    
    for x, y, w, h, label in bb_df[['left', 'top', 'width', 'height', 'label']].values:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Ensure label is a string
        label_text = ' '.join(map(str, label)) if isinstance(label, np.ndarray) else str(label)
        cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])

    class groupgen():
        def __init__(self):
            self.id = 0
            self.text = ''
            
        def getgroup(self, text):
            if self.text == text:
                return self.id
            else:
                self.id += 1
                self.text = text
                return self.id
            
    grp_gen = groupgen()
    
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)
    
    # right and bottom of bounding box
    bb_df[['left', 'top', 'width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']
    
    # tagging: groupby group
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    
    img_tagging = group_tag_img.agg({
        'left': min,
        'right': max,
        'top': min,
        'bottom': max,
        'label': np.unique,
        'token': lambda x: " ".join(x)
    })

    img_bb = image.copy()
    for l, r, t, b, label, token in img_tagging.values:
        cv2.rectangle(img_bb, (l, t), (r, b), (0, 255, 0), 2)
        
        # Ensure label is a string
        label_text = ' '.join(map(str, label)) if isinstance(label, np.ndarray) else str(label)
        cv2.putText(img_bb, label_text, (l, t), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    info_array = dataframe_info[['token', 'label']].values
    entities = {
        'PAYMENTBYERPHONE': [], 'PAYMENTBYERFAX': [], 'PAYMENTBYEREMAIL': [], 'PAYMENTCOMPANYNAME': [],
        'PAYMENTBYERADDRESS': [], 'PAYMENTBYERNAME': [], 'ORDERDATE': [], 'ACCESSKEY': [], 
        'FAULTDETAIL': [], 'INSTRUCTIONNOTES': [], 'SHIPPINGNAME': [], 'SHIPPINGPHONE': [],
        'SHIPPINGSTREET': [], 'PAYMENTBILLINGNAME': [], 'PAYMENTPONUMBER': []
    }

    previous_label_tag = 'O'

    for token, label in info_array:
        if label == 'O':
            continue
        bio_tag, label_tag = label.split('-')
        parsed_token = parser(token, label_tag)

        if bio_tag == 'B' or previous_label_tag != label_tag:
            entities[label_tag].append(parsed_token)
        elif bio_tag == 'I' and previous_label_tag == label_tag:
            if label_tag in ['PAYMENTCOMPANYNAME', 'SHIPPINGNAME', 'PAYMENTBILLINGNAME']:
                entities[label_tag][-1] += f" {parsed_token}"
            else:
                entities[label_tag][-1] += parsed_token

        previous_label_tag = label_tag

    return img_bb, entities

# Assuming `image` is the input image to process
# img_bb, entities = getPredictions(image)
