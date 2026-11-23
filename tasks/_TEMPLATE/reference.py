"""参考实现。judgment 类题目不需要本文件。

约定：暴露 arun(*tensors) -> outputs，纯 torch，不依赖被测库。
"""
import torch


def arun(A, B):
    return A @ B
