from functools import reduce
import inspect
from typing import Union
import torch.nn as nn
import torch
import random
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .util import *

class FS:
	def __init__(self) -> None:
		super().__init__()
		pass

	_recentPerturbation = None  # For Initialize accumalated faults when perform weight injection. targetLayer, singleDimensionalIdx, original value
	_neurons = []
	_NofNeurons = 0

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

	def gatherAllNeuronValues(self, model: nn.Module):
		_moduleNames = self.getModuleNameList(model)
		def hook(module, input, output):
			_neurons = output.cpu().numpy()
			_singleDimensionalNeurons = _neurons.reshape(-1)
			self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			self._NofNeurons += len(_singleDimensionalNeurons)

		for layer in _moduleNames:
			self.getModuleByName(model, layer).register_forward_hook(hook)


	
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
			# plt.hist(_singleDimensionalNeurons, bins=100, range=[-2, 2])
			# plt.xlabel("Weight Value")
			# plt.ylabel("Count")
			# plt.show()


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


	def offlineSinglayerWeightInjection(self, model: nn.Module, targetLayer: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random", accumulate: bool=True):
		_moduleNames = self.getModuleNameList(model)
		# _moduleNames = [i for i in _moduleNames if "MaxPool2d" not in i or "ReLU" not in i]

		if(accumulate == False and self._recentPerturbation != None):  # Target of this method is SingleLayer, don't care of _recentPerturbation.targetLayerIdx = list
			# print(self._recentPerturbation)
			_recentTargetLayer = self.getModuleByName(model, _moduleNames[self._recentPerturbation["targetLayerIdx"]])
			_recentTargetWeights = _recentTargetLayer.weight.cpu().numpy()
			_originalShape = _recentTargetWeights.shape
			_SDrecentTargetWeights = _recentTargetWeights.reshape(-1)
			# print("Recovery")
			for i in range(len(self._recentPerturbation["targetWeightIdxes"])):
				# print("("+str(i+1)+") " + str(_SDrecentTargetWeights[self._recentPerturbation["targetWeightIdxes"][i]]) + " -> " + str(self._recentPerturbation["originalValues"][i]))
				_SDrecentTargetWeights[self._recentPerturbation["targetWeightIdxes"][i]] = np.float64(self._recentPerturbation["originalValues"][i])
			
			_recentTargetLayer.weight = torch.nn.Parameter(torch.DoubleTensor(_SDrecentTargetWeights.reshape(_originalShape)).cuda())
			
			self._recentPerturbation = None

		_exceptLayers = [nn.modules.pooling, nn.modules.dropout, nn.modules.activation]

		_layerFilter = tuple(x[1] for i in _exceptLayers for x in inspect.getmembers(i, inspect.isclass))
		# print(_layerFilter)
		if(targetLayer == "random"):
			_verifiedLayer = False
			while(not _verifiedLayer):
				_targetLayerIdx = random.randint(0, len(_moduleNames)-1)
				_targetLayer = self.getModuleByName(model, _moduleNames[_targetLayerIdx])
				# print(_targetLayer, (type(_targetLayer) not in _layerFilter))
				if(type(_targetLayer) not in _layerFilter):
					_verifiedLayer = True
					# print("Escaping loop")
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)

		# print(type(_targetLayer))

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		_weights = _targetLayer.weight.cpu().numpy()
		_originalWeightShape = _weights.shape
		_singleDimensionalWeights = _weights.reshape(-1)

		if(errorRate == "unset"):
				_numError = NofError
		if(NofError == "unset"):
			_numError = int(_weights.size * errorRate)

		_targetIndexes = self.generateTargetIndexList(_singleDimensionalWeights.shape, _numError)
		
		if(targetBit == "random"):
			_targetBitIdx = random.randint(0, 31)
		elif(type(targetBit) == int):
			_targetBitIdx = targetBit

		_originalValues = []
		for _targetWeightIdx in _targetIndexes:
			_originalValues.append(_singleDimensionalWeights[_targetWeightIdx])
			bits = list(binary(_singleDimensionalWeights[_targetWeightIdx]))
			bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
			_singleDimensionalWeights[_targetWeightIdx] = np.float64(binToFloat("".join(bits)))
		
		self._recentPerturbation = {
				"targetLayerIdx": _targetLayerIdx,
				"targetWeightIdxes": _targetIndexes,
				"originalValues": _originalValues
			}

		_weights = _singleDimensionalWeights.reshape(_originalWeightShape)
		# print(_targetLayer.weight.cpu().numpy() == _weights)
		# _targetLayer.weight = torch.nn.Parameter(torch.FloatTensor(_weights).cuda())
		torch.set_default_tensor_type(torch.cuda.DoubleTensor)
		print(type(torch.cuda.DoubleTensor(_weights)))
		_targetLayer.weight = torch.nn.Parameter(torch.DoubleTensor(_weights).cuda())

		# print(_singleDimensionalWeights)
		# print(len(_singleDimensionalWeights))