"""参考实现。暴露 arun(*tensors) -> outputs，纯 torch。

这是**正确性**参考，不是性能基线：torch 的 a @ b 在本机走 hipBLASLt，
但作为「题目参考」它只定义语义，不作为达标分母。
"""
import torch


def arun(A, B):
    return A @ B
