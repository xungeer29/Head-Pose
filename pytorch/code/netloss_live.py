# -*- coding:utf-8 -*-
import torch
import numpy as np
import pdb


class MyLossFunction(Function):
    """Layer of Efficient Siamese loss function."""
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        self.margin = 10
        self.Num = 0
        batch = 2
        level = 68
        SepSize = batch*level
        self.dis = []
        for i in range(0, SepSize-batch):
            for j in range(i+batch, SepSize):
                self.dis.append(input[i] - input[j])
                self.Num += 1
        self.dis = np.asarray(self.dis)
        self.loss = np.maximum(0, self.margin-self.dis)
        output = np.sum(self.loss)/input[0].size
        # 将numpy转化到torch张量

        return output

    def backward(ctx, grad_output):
        """The parameters here have the same meaning as data_layer"""
        batch=2
        index = 0
        level = 68
        SepSize = batch*level
        self.ref= np.zeros(bottom[0].num,dtype=np.float32)
        for i in range(SepSize*k,SepSize*(k+1)-batch):
            for j in range(SepSize*k + int((i-SepSize*k)/batch+1)*batch,SepSize*(k+1)):
                if self.loss[index]>0:
                    self.ref[i] += -1
                    self.ref[j] += +1
                index +=1

        # Efficient Siamese backward pass        
        bottom[0].diff[...]= np.reshape(self.ref,(bottom[0].num,1))/bottom[0].num
        
                
    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""

        top[0].reshape(1)
 
