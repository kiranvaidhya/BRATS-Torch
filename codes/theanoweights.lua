require 'cudnn'
require 'cunn'
py = require('fb.python')

py.exec([=[
import cPickle
f = open('new_n4pre_training.pkl','rb')
gen = cPickle.load(f)
w = []
b = []
for i in xrange(3):
	w.append(cPickle.load(f))
	b.append(cPickle.load(f))
	]=])

weights = py.eval('w')
biases = py.eval('b')

model = nn.Sequential()
model:add(nn.Linear(1764,2500))
model:add(cudnn.Sigmoid())
model:add(nn.Linear(2500,1000))
model:add(cudnn.Sigmoid())
model:add(nn.Linear(1000,500))
model:add(cudnn.Sigmoid())
model:add(nn.Linear(500,5))

model:get(7).weight:zero()
model:get(7).bias:zero()

model:add(cudnn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

-- local method = 'xavier'
-- model = require('weight-init')(model, method)

k = 1
for i = 1,5,2 do
	model:get(i).weight = weights[k]:t()
	model:get(i).bias = biases[k]
	k = k+1
end

model = model:cuda()
criterion = criterion:cuda()