Pytorch Fault Simulator
=

## Requirements
torch (1.13.0)   
numpy (1.21.4)   

## Methods
> setLayerInfo   
* Description   
  > 본 module을 사용하기 위해서, model을 불러오고 inference를 시작하기 전에 꼭 수행해야 하는 작업   
  (injection overhead 최소화 등의 목적)   
  이후에 기술되는 모든 메서드들을 사용하기 전, 다음 작업이 선행되어야 한다:   
  \* 이후 기술되는 메서드들을 사용할 때도, FS().method()처럼 새 인스턴스를 생성하는 것이 아닌,   
  아래 예시와 같이 기존에 변수에 할당해 둔 인스턴스를 사용해야 함
  ```python
  import vgg16
  import FS
  
  model = vgg16()
  fault_sim = FS()
  
  fault_sim.setLayerInfo(model)
  
  ...
  fault_sim.onlineSingleLayerOutputInjection(...)
  ```
* parameters
  * model: torch.nn.Module (required)   
    bit flip을 주입할 모델   

* return value   
  * None   
   

> onlineSingleLayerOutputInjection   
* parameters
  * model: torch.nn.Module (required)   
    bit flip을 주입할 모델   

  * errorRate: float (required *)   
    Layer output중 bit flip의 대상이 될 neuron의 비율   
    ex) layer output=100,000개, errorRate이 0.1이라면 총 10,000번의 bit flip 발생   
    \* NofError와 같이 사용될 수 없음
       
  * NofError: int (required *)   
    Layer output중 bit flip의 대상이 될 neuron의 개수   
    ex) layer output=100,000개, errorRate이 100이라면 총 100번의 bit flip 발생   
    \* errorRate와 같이 사용될 수 없음
       
  * targetBit: int or "random" (required)   
    bit flip이 발생하는 bit position을 특정할 수 있음   
    32bit float 자료형을 사용하는 모델의 경우 0~31로 지정 가능   
    랜덤한 bit position에 에러를 주입하고 싶은 경우 "random" 문자열 전달
       
  * targetLayer: str or "random" (required)   
    target이 되는 layer를 지정하고 싶은 경우 지정 가능   
    본 module 내 getModuleNameList 메서드의 반환값 중 하나를 선택해 전달   
    target이 되는 layer를 랜덤하게 지정하고 싶은 경우에는 "random" 문자열을 꼭 전달해야 함
       
  * targetLayerTypes: list\<Layer class\> (optional)   
    target이 되는 layer의 종류를 지정하고 싶은 경우 사용   
    ex) conv2d layer들만 target으로 제한하고 싶은 경우, 빈 리스트 안에 torch.nn.Conv2d 클래스를 넣어 전달,   
    여러 타입을 지정하는 것 또한 가능
    
* return value   
  * torch.utils.hooks.RemovableHandle   
    layer에 설정된 hook을 제거할 수 있는 handle   
    remove() 메서드를 사용해 hook을 제거할 수 있으며,   
    매 inference마다 injection 메서드를 호출하는 경우, inference가 끝난 뒤 꼭 remove()를 호출해줘야 함   
    (그렇지 않은 경우 hook이 중첩되어 inference 횟수만큼 injection이 추가로 실행됨)

> onLineSingleLayerInputInjection   
* parameters
  * model: torch.nn.Module (required)   
    bit flip을 주입할 모델   

  * errorRate: float (required *)   
    Layer output중 bit flip의 대상이 될 neuron의 비율   
    ex) layer output=100,000개, errorRate이 0.1이라면 총 10,000번의 bit flip 발생   
    \* NofError와 같이 사용될 수 없음
       
  * NofError: int (required *)   
    Layer output중 bit flip의 대상이 될 neuron의 개수   
    ex) layer output=100,000개, errorRate이 100이라면 총 100번의 bit flip 발생   
    \* errorRate와 같이 사용될 수 없음
       
  * targetBit: int or "random" (required)   
    bit flip이 발생하는 bit position을 특정할 수 있음   
    32bit float 자료형을 사용하는 모델의 경우 0~31로 지정 가능   
    랜덤한 bit position에 에러를 주입하고 싶은 경우 "random" 문자열 전달
       
  * targetLayer: str or "random" (required)   
    target이 되는 layer를 지정하고 싶은 경우 지정 가능   
    본 module 내 getModuleNameList 메서드의 반환값 중 하나를 선택해 전달   
    target이 되는 layer를 랜덤하게 지정하고 싶은 경우에는 "random" 문자열을 꼭 전달해야 함
       
  * targetLayerTypes: list\<Layer class\> (optional)   
    target이 되는 layer의 종류를 지정하고 싶은 경우 사용   
    ex) conv2d layer들만 target으로 제한하고 싶은 경우, 빈 리스트 안에 torch.nn.Conv2d 클래스를 넣어 전달,   
    여러 타입을 지정하는 것 또한 가능
    
* return value   
  * torch.utils.hooks.RemovableHandle   
    layer에 설정된 hook을 제거할 수 있는 handle   
    remove() 메서드를 사용해 hook을 제거할 수 있으며,   
    매 inference마다 injection 메서드를 호출하는 경우, inference가 끝난 뒤 꼭 remove()를 호출해줘야 함   
    (그렇지 않은 경우 hook이 중첩되어 inference 횟수만큼 injection이 추가로 실행됨)

> offlineSinglayerWeightInjection   
* parameters
  * model: torch.nn.Module (required)   
    bit flip을 주입할 모델   

  * errorRate: float (required *)   
    Layer output중 bit flip의 대상이 될 neuron의 비율   
    ex) layer output=100,000개, errorRate이 0.1이라면 총 10,000번의 bit flip 발생   
    \* NofError와 같이 사용될 수 없음
       
  * NofError: int (required *)   
    Layer output중 bit flip의 대상이 될 neuron의 개수   
    ex) layer output=100,000개, errorRate이 100이라면 총 100번의 bit flip 발생   
    \* errorRate와 같이 사용될 수 없음
       
  * targetBit: int or "random" (required)   
    bit flip이 발생하는 bit position을 특정할 수 있음   
    32bit float 자료형을 사용하는 모델의 경우 0~31로 지정 가능   
    랜덤한 bit position에 에러를 주입하고 싶은 경우 "random" 문자열 전달
       
  * targetLayer: str or "random" (required)   
    target이 되는 layer를 지정하고 싶은 경우 지정 가능   
    본 module 내 getModuleNameList 메서드의 반환값 중 하나를 선택해 전달   
    target이 되는 layer를 랜덤하게 지정하고 싶은 경우에는 "random" 문자열을 꼭 전달해야 함
       
  * accumulate: bool (required)   
    weight에 주입된 에러는 layer의 input, output과 다르게 inference가 끝난 뒤에도 model에 잔류함   
    이를 유지(누적)할지, 복원할지 선택할 수 있는 parameter   
    accumulate=True를 사용함으로써 hard error(stuck-at)를 구현할 수도 있음   

  * targetLayerTypes: list\<Layer class\> (optional)   
    target이 되는 layer의 종류를 지정하고 싶은 경우 사용   
    ex) conv2d layer들만 target으로 제한하고 싶은 경우, 빈 리스트 안에 torch.nn.Conv2d 클래스를 넣어 전달,   
    여러 타입을 지정하는 것 또한 가능   

* return value   
  * None   


> gatherAllNeuronValues   


> getLog   
* parameter   
  * None   

* return value
  * list   
    각 injection에 대한 전체 log가 포함된 배열을 반환   
    각 필드는 콜론(:)으로 구분되며, 구성은 다음과 같음:   
    {0}:{1}:{2}:{3}:{4}:{5}:{6}:{7}   
       
    0: target layer의 index (몇 번째 layer인지)   
    1: target layer의 정보   
    2: target neuron/weight의 "1차원" index   
    3: target bit의 index   
    4: target neuron/weight의 원본 값 (2진 표현)   
    5: target neuron/weight의 원본 값 (10진 표현)   
    6: bit flip 주입 이후 target neuron/weight의 값 (2진 표현)   
    7: bit flip 주입 이후 target neuron/weight의 값 (10진 표현)

## Example
```python
from Models.VGGModel import VGG, vgg16_bn

testset = torchvision.datasets.CIFAR10(root=os.path.dirname(__file__)+'/../Datasets/CIFAR10/data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

model = vgg16_bn().cuda()
model.load_state_dict(torch.load(os.path.dirname(__file__)+"/../Models/trained/vgg16_bn.pt"))

model.eval()
fs = FS()
fs.setLayerInfo(model)

with torch.no_grad():
  for data in testloader:
    inputs, labels = data[0].cuda(), data[1].cuda()
    handle = fs.onlineSingleLayerOutputInjection(model=model, targetLayerTypes=[torch.nn.ReLU], NofError=1, targetBit=1)
    outputs = model(inputs)
    handle.remove()
```
