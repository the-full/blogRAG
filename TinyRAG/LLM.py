#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   LLM.py
@Time    :   2024/02/12 13:50:47
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''
import os
from typing import Dict, List

from openai import OpenAI
from zhipuai import ZhipuAI

PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPALTE="""
        结合上下文来回答用户的问题。如果你不知道答案，就说你不知道。你总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请声明数据库中没有这个内容，你将按照自己的所知的回答，提醒用户注意分辨回答内容。
        有用的回答:""",
    InternLM_PROMPT_TEMPALTE="""
        先对上下文进行内容总结,再使用上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)


class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        return ""

    def load_model(self):
        pass

class DeepSeekChat(BaseModel):
    def __init__(self, path: str = '', model: str = "deepseek-chat") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI(
            api_key = os.getenv("DEEPSEEK_API_KEY", ""),
            base_url = os.getenv("DEEPSEEK_BASE_URL", ""),
        )
        history.append({
            'role': 'user',
            'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        })
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message.content

class ZhipuChat(BaseModel):
    def __init__(self, path: str = '', model: str = "glm-4-flash") -> None:
        super().__init__(path)
        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        self.model = model

    def chat(self, prompt: str, history: List[Dict], content: str) -> str:
        history.append({
            'role': 'user', 
            'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPALTE'].format(question=prompt, context=content)
        })
        response = self.client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=150,
            temperature=0.1
        )
        return response.choices[0].message
