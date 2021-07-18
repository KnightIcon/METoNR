# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random
import re


def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def newsample(news, ratio):
    """ Sample ratio samples from news list. 
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number
    
    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)



