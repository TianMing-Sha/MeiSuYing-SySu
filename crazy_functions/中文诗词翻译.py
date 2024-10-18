from toolbox import CatchException, update_ui
from .crazy_utils import request_gpt_model_in_new_thread_with_ui_alive, input_clipping
import requests
from bs4 import BeautifulSoup
from request_llms.bridge_all import model_info
import re
from tavily import TavilyClient
import ast
import udkundoku

@CatchException
def 中文诗词翻译(main_topic, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, user_request):
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
    chatbot.append((f"正在将这个句子翻译为日语： {main_topic}",
                    None))
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面 # 由于请求gpt需要一段时间，我们先及时地做一次界面更新

    lzh=udkundoku.load()
    txt1 , txt2 = main_topic.split(',')
    s=lzh(txt1)
    t=udkundoku.translate(s)
    gpt_say = t.sentence()
    chatbot[-1] = (f"正在将这个句子翻译为日语： {main_topic}", gpt_say)
    history.append(main_topic);history.append(gpt_say)
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面 # 界面更新

