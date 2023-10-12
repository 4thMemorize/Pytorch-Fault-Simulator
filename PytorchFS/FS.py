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

	_log = []                   # For record information of all injection
	_recentPerturbation = None  # For Initialize accumalated faults when perform weight injection. targetLayer, singleDimensionalIdx, original value
	_neurons = []
	_NofNeurons = 0
	_layerInfo = None           # For optimize layer selection process. store kind of layer and it's indexes

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
	
	def selectRandomTargetLayer(self, model, moduleNames, layerTypes=None):
		if(layerTypes == None):
			_targetIdx = random.randint(0, len(_targetLayerIdxCand)-1)
			_targetLayer = self.getModuleByName(model, moduleNames[_targetIdx])
		else:
			_targetLayerIdxCand = []
			for x in layerTypes:
				if str(x) not in self._layerInfo:
					msg = f"This model has no attribute: {x}. You must set targetLayerTypes which belongs to {self._layerInfo.keys()}"
					raise KeyError(msg)
				else:
					_targetLayerIdxCand += self._layerInfo["{}".format(x)]

			# print(_targetLayerIdxCand)
			_targetIdx = random.randint(0, len(_targetLayerIdxCand)-1)
			# print(_targetLayerIdx, _targetLayerIdxCand[_targetLayerIdx])
			_targetLayer = self.getModuleByName(model, moduleNames[_targetLayerIdxCand[_targetIdx]])
			# print(_targetLayer, (type(_targetLayer) not in _layerFilter))
		return _targetLayer, _targetIdx
		

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

	def getLog(self):
		return self._log
	
	def initLog(self):
		self._log = []


	def gatherAllNeuronValues(self, model: nn.Module, targetNeuron: str, targetlayerTypes: list=None):
		_moduleNames = self.getModuleNameList(model)
		_targetLayerIdxCand = []

		def inputHook(module, input):
			# print(module)
			_neurons = input[0].cpu().numpy()
			_singleDimensionalNeurons = _neurons.reshape(-1)
			self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			self._NofNeurons += len(_singleDimensionalNeurons)

		def outputHook(module, input, output):
			# print(module)
			_neurons = output.cpu().numpy()
			_singleDimensionalNeurons = _neurons.reshape(-1)
			self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			self._NofNeurons += len(_singleDimensionalNeurons)

		for x in targetlayerTypes:
			if str(x) not in self._layerInfo:
				msg = f"This model has no attribute: {x}. You must set targetLayerTypes which belongs to {self._layerInfo.keys()}"
				raise KeyError(msg)
			else:
				_targetLayerIdxCand += self._layerInfo["{}".format(x)]

		# print(_targetLayerIdxCand)

		if(targetNeuron=="output"):
			for idx in _targetLayerIdxCand:
				self.getModuleByName(model, _moduleNames[idx]).register_forward_hook(outputHook)
		elif(targetNeuron=="input"):
			for idx in _targetLayerIdxCand:
				self.getModuleByName(model, _moduleNames[idx]).register_forward_pre_hook(inputHook)
		else:
			raise ValueError("You must set 'targetNeuron' \"input\" or \"output\".")


	def setLayerInfo(self, model: nn.Module):
		self._layerInfo = dict()
		_moduleNames = self.getModuleNameList(model)
		for i in range(len(_moduleNames)):
			layer = self.getModuleByName(model, _moduleNames[i])
			if(str(type(layer)) not in self._layerInfo):
				self._layerInfo["{}".format((type(layer)))] = [i]
			else:
				self._layerInfo["{}".format((type(layer)))].append(i)
		
		# for i in self._layerInfo["{}".format((str(nn.modules.Conv2d)))]:
		# 	print(i)
		# 	print(self.getModuleByName(model, _moduleNames[i]))

	
	def onlineSingleLayerOutputInjection(self, model: nn.Module, targetLayer: str, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)
			_targetLayerIdx = _moduleNames.index(targetLayer)

		# print(_targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		def hook(module, input, output):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal targetBit
			nonlocal _targetLayerIdx
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

			# print(_targetBitIdx)

			tmpLog = []
			for _targetNeuronIdx in _targetIndexes:
				beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
				beforeBinaryRep = binary(beforeDecRep)
				bits = list(beforeBinaryRep)
				bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
				afterBinaryRep = "".join(bits)
				_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat(afterBinaryRep)
				tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx, _targetBitIdx, beforeBinaryRep, beforeDecRep, afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx]))

			_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)
		
			self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			self._NofNeurons += len(_singleDimensionalNeurons)

			if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
			else:
				self._log.append(tmpLog)

			return torch.FloatTensor(_neurons).cuda()
		
		hookHandler = _targetLayer.register_forward_hook(hook)

		return hookHandler
	
	def onlineSingleLayerInputInjection(self, model: nn.Module, targetLayer: str, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):
		_moduleNames = self.getModuleNameList(model)
		if(targetLayer == "random"):
			_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
		elif(type(targetLayer) == str):
			_targetLayer = self.getModuleByName(model, targetLayer)
			_targetLayerIdx = _moduleNames.index(targetLayer)

		# print(_targetLayer)

		if(not((type(errorRate) == str) ^ (type(NofError) == str))):
			raise ValueError('Only one parameter between "errorRate" and "NofError" must be defined.')
		if( type(errorRate) == int and errorRate > 1): raise ValueError('The value of parameter "errorRate" must be smaller than 1.')

		def hook(module, input):
			nonlocal _moduleNames  # Enclosing(바깥함수)에서 가공한 변수(총 에러 개수 등)를 nonlocal 키워드로 끌어와 그때그때 조건에 따른 hook function을 generate하는 게 가능함.
			nonlocal errorRate		 # 에러 개수를 errorRate로 받았을 때 neuron개수와 곱해주는 등, 안/바깥 함수 간 연산이 필요할 때 위와 같이 사용
			nonlocal NofError
			nonlocal targetBit
			nonlocal _targetLayerIdx
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

			tmpLog = []
			for _targetNeuronIdx in _targetIndexes:
				beforeDecRep = _singleDimensionalNeurons[_targetNeuronIdx]
				beforeBinaryRep = binary(beforeDecRep)
				bits = list(beforeBinaryRep)
				bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
				afterBinaryRep = "".join(bits)
				_singleDimensionalNeurons[_targetNeuronIdx] = binToFloat(afterBinaryRep)

				tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetNeuronIdx, _targetBitIdx, beforeBinaryRep, beforeDecRep, afterBinaryRep, _singleDimensionalNeurons[_targetNeuronIdx]))

			_neurons = _singleDimensionalNeurons.reshape(_originalNeuronShape)
			self._neurons = np.concatenate((self._neurons, _singleDimensionalNeurons))
			self._NofNeurons += len(_singleDimensionalNeurons)
			
			if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
			else:
				self._log.append(tmpLog)

			return torch.FloatTensor(_neurons).cuda()
		
		hookHandler = _targetLayer.register_forward_pre_hook(hook)

		return hookHandler
	
	# def onlineMultiLayerOutputInjection(self, model: nn.Module, targetLayer: str, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random"):


	def offlineSinglayerWeightInjection(self, model: nn.Module, targetLayer: str, targetLayerTypes: list=None, errorRate: float="unset", NofError: int="unset", targetBit: Union[int, str]="random", accumulate: bool=True):
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
			
			_recentTargetLayer.weight = torch.nn.Parameter(torch.FloatTensor(_SDrecentTargetWeights.reshape(_originalShape)).cuda())
			
			self._recentPerturbation = None

		_exceptLayers = [nn.modules.pooling, nn.modules.dropout, nn.modules.activation]

		_layerFilter = tuple(x[1] for i in _exceptLayers for x in inspect.getmembers(i, inspect.isclass))
		# print(_layerFilter)
		if(targetLayer == "random"):
			_verifiedLayer = False
			while(not _verifiedLayer):
				_targetLayer, _targetLayerIdx = self.selectRandomTargetLayer(model, _moduleNames, targetLayerTypes)
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
		tmpLog = []
		for _targetWeightIdx in _targetIndexes:
			_originalValues.append(_singleDimensionalWeights[_targetWeightIdx])
			beforeDecRep = _singleDimensionalWeights[_targetWeightIdx]
			beforeBinaryRep = binary(beforeDecRep)
			bits = list(beforeBinaryRep)
			bits[_targetBitIdx] = str(int(not bool(int(bits[_targetBitIdx]))))
			afterBinaryRep = "".join(bits)
			_singleDimensionalWeights[_targetWeightIdx] = np.float64(binToFloat(afterBinaryRep))
			tmpLog.append("{}:{}:{}:{}:{}:{}:{}:{}".format(_targetLayerIdx, _targetLayer, _targetWeightIdx, _targetBitIdx, beforeBinaryRep, beforeDecRep, afterBinaryRep, _singleDimensionalWeights[_targetWeightIdx]))
			
		self._recentPerturbation = {
				"targetLayerIdx": _targetLayerIdx,
				"targetWeightIdxes": _targetIndexes,
				"originalValues": _originalValues
			}
		if(len(tmpLog) == 1):
				self._log.append(tmpLog[0])
		else:
			self._log.append(tmpLog)

		_weights = _singleDimensionalWeights.reshape(_originalWeightShape)
		# print(_targetLayer.weight.cpu().numpy() == _weights)
		_targetLayer.weight = torch.nn.Parameter(torch.FloatTensor(_weights).cuda())
		# torch.set_default_tensor_type(torch.cuda.DoubleTensor)
		# print(type(torch.cuda.DoubleTensor(_weights)))
		# _targetLayer.weight = torch.nn.Parameter(torch.DoubleTensor(_weights).cuda())

		# print(_singleDimensionalWeights)
		# print(len(_singleDimensionalWeights))