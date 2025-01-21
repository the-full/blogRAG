#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   Embeddings.py
@Time    :   2024/02/10 21:55:39
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

import os
from typing import List

import numpy as np
from tqdm import tqdm
from zhipuai import ZhipuAI

os.environ['CURL_CA_BUNDLE'] = ''
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


class BaseEmbeddings:
    """
    Base class for embeddings
    """
    def __init__(self, path: str, is_api: bool, dimensions: int) -> None:
        self.path = path
        self.is_api = is_api
        self.dimensions = dimensions
    
    def get_embedding(self, text: str, model: str = "") -> List[float]:
        raise NotImplementedError
    
    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def __call__(self, documents: List[str]) -> List[float]:
        vectors = []
        for doc in tqdm(documents, desc="Calculating embeddings"):
            vectors.append(self.get_embedding(doc))
        return vectors

class ZhipuEmbedding(BaseEmbeddings):
    """
    class for Zhipu embeddings
    """
    def __init__(self, path: str = '', is_api: bool = True, dimensions: int = 2048) -> None:
        super().__init__(path, is_api, dimensions)
        if self.is_api:
            self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY")) 
    
    def get_embedding(self, text: str, model="embedding-2") -> List[float]:
        response = self.client.embeddings.create(
            model=model,
            input=text,
            dimensions=1024,
        )
        return response.data[0].embedding
