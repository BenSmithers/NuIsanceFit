from NuIsanceFit import steering
from NuIsanceFit.data import Data
from NuIsanceFit.param import ParamPoint
from NuIsanceFit.torchmin import GolemModel

from torch import optim

dataobj = Data(steering)
goWeighter = GolemModel(dataobj)

params = ParamPoint().to_tensor()


n = 1000
print(goWeighter.parameters())
optimizer = optim.Adam(goWeighter.parameters(), lr=0.1)

for i in range(n):
    optimizer.zero_grad()

    loss = goWeighter.get_llh()
    loss.backward()
    optimizer.step()

print(goWeighter.weights)
