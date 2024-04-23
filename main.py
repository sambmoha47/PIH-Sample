import fitz
import numpy as np
from PIL import Image
import streamlit as st
import io
import torch
# from transformers import AutoTokenizer, AutoModelForDocumentQuestionAnswering, LayoutLMv2ImageProcessor
# tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-invoices")
# model = AutoModelForDocumentQuestionAnswering.from_pretrained("impira/layoutlm-invoices")
# image_processor = LayoutLMv2ImageProcessor()
from transformers import pipeline

pipe = pipeline("document-question-answering", model="impira/layoutlm-invoices")

zoom_x = 2.0  
zoom_y = 2.0 
mat = fitz.Matrix(zoom_x, zoom_y)

def process_vqa(img, question):
    # encoding = tokenizer(
    #             return_token_type_ids=True,
    #             truncation="only_second",
    #             return_overflowing_tokens=True,
    #             text = question,
    #             text_pair = words,
    #             is_split_into_words = True
    #         )
    # encoding.pop("overflow_to_sample_mapping", None)
    # num_spans = len(encoding["input_ids"])

    # p_mask = [[tok != 1 for tok in encoding.sequence_ids(span_id)] for span_id in range(num_spans)]
    # for span_idx in range(num_spans):
    #     span_encoding = {k: torch.tensor(v[span_idx : span_idx + 1]) for (k, v) in encoding.items()}
    #     input_ids_span_idx = encoding["input_ids"][span_idx]

    #     if tokenizer.cls_token_id is not None:
    #         cls_indices = np.nonzero(np.array(input_ids_span_idx) == tokenizer.cls_token_id)[0]
    #         for cls_index in cls_indices:
    #             p_mask[span_idx][cls_index] = 0

    #     bbox = []
    #     for input_id, sequence_id, word_id in zip(
    #         encoding.input_ids[span_idx],
    #         encoding.sequence_ids(span_idx),
    #         encoding.word_ids(span_idx),
    #     ):
    #         if sequence_id == 1:
    #             bbox.append(boxes[word_id])
    #         elif input_id == tokenizer.sep_token_id:
    #             bbox.append([1000] * 4)
    #         else:
    #             bbox.append([0] * 4)

    #     span_encoding["bbox"] = torch.tensor(bbox).unsqueeze(0)

    # model_outputs = model(**span_encoding)
    # model_outputs = dict(model_outputs.items())
    # model_outputs["attention_mask"] = span_encoding.get("attention_mask", None)

    # start_logits = model_outputs['start_logits']
    # end_logits = model_outputs['end_logits']
    # predicted_start_idx = start_logits.argmax(-1).item()
    # predicted_end_idx = end_logits.argmax(-1).item()

    # answer = tokenizer.decode(span_encoding["input_ids"].squeeze()[predicted_start_idx : predicted_end_idx + 1])
    pp = pipe(image=img, question=question)
    return pp[0]['answer']

st.title('Invoice Extractor')

uploaded_file = st.sidebar.file_uploader("Upload Invoice", type=['jpg', 'jpeg', 'png', 'pdf'])

image_placeholder = st.empty()

if uploaded_file is not None:
    print(uploaded_file.type)
    if uploaded_file.type == 'application/pdf':
        document = fitz.open(stream=io.BytesIO(uploaded_file.getvalue()), filetype="pdf")
        page = document[0]
        map = page.get_pixmap(matrix=mat)
        map.save('bb.png')
        image = Image.open('bb.png')
    else:
        image_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_data))

    image_placeholder.image(image, caption='Uploaded Invoice', use_column_width=True)
else:
    image_placeholder.image('upload.png', caption='Please upload an Invoice.')

question = st.text_input("Enter your question about the invoice")

if st.button('Submit Question'):
    if uploaded_file is not None and question:
        answer = process_vqa(image, question)
        st.write("Answer:", answer)
    else:
        st.warning("Please upload an image and enter a question.")
