from toolbox import CatchException, update_ui
from .crazy_utils import request_gpt_model_in_new_thread_with_ui_alive, input_clipping
import requests
from bs4 import BeautifulSoup
from request_llms.bridge_all import model_info
import re
from tavily import TavilyClient
import ast



import os
import logging

import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer

logging.basicConfig(
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO)

gpu_list = None


class Bert(nn.Module):
    def __init__(self, BERT_PATH='./BERT_CCPoem_v1'):
        super(Bert, self).__init__()

        self.bert = BertModel.from_pretrained(BERT_PATH)

    def init_multi_gpu(self, device):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, cls=False):
        result = []
        # print(data)
        x = data['input_ids']
        y = self.bert(x, attention_mask=data['attention_mask'],
                         token_type_ids=data['token_type_ids'])[0]
        
        if(cls):
            result = y[:, 0, :].view(y.size(0), -1)
            result = result.cpu().tolist()
        else:
            result = []
            y = y.cpu()
            # y = torch.mean(y, 1)
            # result = y.cpu().tolist()
            for i in range(y.shape[0]):
                tmp = y[i][1:torch.sum(data['attention_mask'][i]) - 1, :]
                result.append(tmp.mean(0).tolist())

        return result


class BertFormatter():
    def __init__(self, BERT_PATH='./BERT_CCPoem_v1'):
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    def process(self, data):
        res_dict = self.tokenizer.batch_encode_plus(
            data, pad_to_max_length=True)

        input_list = {'input_ids': torch.LongTensor(res_dict['input_ids']),
                      'attention_mask': torch.LongTensor(res_dict['attention_mask']),
                      "token_type_ids": torch.LongTensor(res_dict['token_type_ids'])}
        return input_list


def init(BERT_PATH="./BERT_CCPoem_v1"):
    global gpu_list
    gpu_list = []

    device_list = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if(device_list[0] == ""):
        device_list = []
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))

    cuda = torch.cuda.is_available()
    logging.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logging.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    model = Bert(BERT_PATH)
    formatter = BertFormatter(BERT_PATH)
    if len(gpu_list) > 0:
        model = model.cuda()
    if(len(gpu_list) > 1):
        try:
            model.init_multi_gpu(gpu_list)
        except Exception as e:
            logging.warning(
                "No init_multi_gpu implemented in the model, use single gpu instead. {}".format(str(e)))
    return model, formatter


def predict_vec_rep(data, model, formatter):
    data = formatter.process(data)
    model.eval()

    for i in data:
        if(isinstance(data[i], torch.Tensor)):
            if len(gpu_list) > 0:
                data[i] = data[i].cuda()

    result = model(data)

    return result


def cos_sim(vector_a, vector_b, sim=True):

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    if(not sim):
        return cos
    sim = 0.5 + 0.5 * cos
    return sim






@CatchException
def 相似度计算(main_topic, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
    """
    txt             输入栏用户输入的文本，例如需要翻译的一段话，再例如一个包含了待处理文件的路径
    llm_kwargs      gpt模型参数，如温度和top_p等，一般原样传递下去就行
    plugin_kwargs   插件模型的参数，暂时没有用武之地
    chatbot         聊天显示框的句柄，用于显示给用户
    history         聊天历史，前情提要
    system_prompt   给gpt的静默提醒
    user_request    当前用户的请求信息（IP地址等）
    """
    history = []    # 清空历史，以免输入溢出
    chatbot.append((f"正在计算这两个诗句的相似度 ： {main_topic}",
                    None))
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面 # 由于请求gpt需要一段时间，我们先及时地做一次界面更新
    
    txt1 , txt2 = main_topic.split(',')

    model, formatter = init()

    result_0 = predict_vec_rep([txt1], model, formatter)[0]
    result_1 = predict_vec_rep([txt2], model, formatter)[0]

    percentage = cos_sim(result_0, result_1) * 100
    gpt_say = f"{txt1} , {txt2}  之间的相似度为:  {percentage:.4f}%"


    chatbot[-1] = (f"正在计算这两个诗句的相似度 ： {main_topic}", gpt_say)
    history.append(main_topic);history.append(gpt_say)
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面 # 界面更新

