from functools import reduce
from typing import Union
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .util import *

class FS:
	def __init__(self) -> None:
		super().__init__()
		pass

	def getModuleByName(self, module, access_string):
		names = access_string.split(sep='.')
		return reduce(getattr, names, module)
	
	def getModuleNameList(self, module):
		moduleNames = []
		for name, l in module.named_modules():
			if not isinstance(l, nn.Sequential) and not isinstance(l, type(module)) and (name != ''):
          # print(name)
					moduleNames.append(name)
		return moduleNames
	
	def generateTargetIndexList(slef, shape, n):
		result = []
		for i in range(n):
			tmp = []
			for i in shape:
				tmp.append(random.randint(0, i-1))
			result.append(tmp)
		return result

	def setLayerPerturbation(self, model: nn.Module):
		weights = model.features[0].weight.cpu().numpy()
		weights.fill(0)
		model.features[0].weight = torch.nn.Parameter(torch.FloatTensor(weights).cuda())
	
	# def onlineNeuronInjection(model: nn.Module, targetLayer: str, NofTargetLayer: Union[list, int], targetNeuron: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):
	# 	if(not((type(errorRate) == type(str)) ^ (type(NofError) == type(str)))):
	# 		raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
	# 	if(errorRate == "unset"):
	# 		_numError = NofError
	# 	if(NofError == "unset"):
	# 		_numError = 

	# 	if(targetLayer == "random"): # NofTargetLayer must be int
	# 		if(type(NofTargetLayer) != type(int)):
	# 			raise TypeError('Parameter "NofTargetLayer" must be int, when the value of parameter "targetLayer" is "random".')


	# 	return model
	
	def onlineSingleLayerOutputInjection(self, model: nn.Module, targetLayer: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer = self.getModuleByName(model, _moduleNames[random.randint(0, len(_moduleNames)-1)])
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)

		# print(_targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		def hook(module, input, output):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal targetBit
			_neurons = output.cpu().numpy()
			_originalNeuronShape = _neurons.shape
			_singleDimensionalNeurons = _neurons.reshape(-1)


			if(errorRate == "unset"):
				_numError = NofError
			if(NofError == "unset"):
				_numError = int(_neurons.size * errorRate)

			# print(_neurons.shape)
			# print(_neurons.size)
			# print(_numError)

			_targetIndexes = self.generateTargetIndexList(_singleDimensionalNeurons.shape, _numError)
			# print(_targetIndexes)

			# print(targetBit)
			if(targetBit == "random"):
				_targetBitIdx = random.randint(0, 31)
			elif(type(targetBit) == int):
				_targetBitIdx = targetBit


			for _targetNeuronIdx in _targetIndexes:
				bits = list(binary(_singleDimensionalNeurons[_targetNeuronIdx]))
				bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
				_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat("".join(bits))

				_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)

			return torch.FloatTensor(_neurons).cuda()
		
		hookHandler = _targetLayer.register_forward_hook(hook)

		return hookHandler
	
	def onlineSingleLayerInputInjection(self, model: nn.Module, targetLayer: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer = self.getModuleByName(model, _moduleNames[random.randint(0, len(_moduleNames)-1)])
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)

		# print(_targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		def hook(module, input):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal targetBit
			# print(input)
			_neurons = input[0].cpu().numpy()
			_originalNeuronShape = _neurons.shape
			_singleDimensionalNeurons = _neurons.reshape(-1)


			if(errorRate == "unset"):
				_numError = NofError
			if(NofError == "unset"):
				_numError = int(_neurons.size * errorRate)

			# print(_neurons.shape)
			# print(_neurons.size)
			# print(_numError)

			_targetIndexes = self.generateTargetIndexList(_singleDimensionalNeurons.shape, _numError)
			# print(_targetIndexes)

			# print(targetBit)
			if(targetBit == "random"):
				_targetBitIdx = random.randint(0, 31)
			elif(type(targetBit) == int):
				_targetBitIdx = targetBit


			for _targetNeuronIdx in _targetIndexes:
				bits = list(binary(_singleDimensionalNeurons[_targetNeuronIdx]))
				bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
				_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat("".join(bits))

				_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)

			return torch.FloatTensor(_neurons).cuda()
		
		hookHandler = _targetLayer.register_forward_pre_hook(hook)

		return hookHandler
	
	# def onlineMultiLayerOutputInjection(self, model: nn.Module, targetLayer: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):


	# def offlineSinglayerWeightInjection(self, model: nn.Module, targetLayer: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):
