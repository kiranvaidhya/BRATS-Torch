require 'cudnn'
require 'cunn'
require 'optim'
require 'randomkit'
require 'rmsprop_anneal'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-batchSize',96,'batchSize: 96')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.8, 'momentum (SGD only)')
cmd:option('-learningRateDecay',1e-07,'LR decay')
cmd:option('-nHidden',1000,'Number of hidden neurons')
cmd:option('-noise',0.2,'Masking noise')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | adagrad | rmsprop | nag | armsprop')
cmd:option('-save', 'tmp', 'subdirectory to save/log experiments in')
cmd:option('-trdata','data/s_pretrain.t7','Train Data: Pretraining data')
cmd:option('-tedata','data/sv_pretrain.t7','Test Data')
cmd:option('-coefL1',0,'coefL1')
cmd:option('-coefL2',0,'coefL2')
cmd:option('-size','normal', 'size: normal | small (for faster experiments)')
cmd:option('-layerNo',1,'nth layer')
cmd:option('-epochs',300,'epochs: number of epochs')
cmd:option('-validFreq',5,'Validation Frequency: 5')
cmd:text()

opt = cmd:parse(arg)


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

print '==> Loading Data..'
data = torch.load(opt.trdata)

if opt.loader == 'data/preProcessed.t7' then
	pSize = data.data:size(3)
	data.data = data.data:reshape(data.data:size(1),pSize*pSize)
	pSize = pSize*pSize
else
	data = {
	data = data;
	}
	pSize = data.data:size(2)
end

if opt.size == 'small' then
	data.data = data.data[{{1,50000}}]
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

print '==> Noising Data..'

noisedData = torch.Tensor(data.data:size(1),pSize):zero():float()
noisedData = randomkit.binomial(noisedData,1,1-opt.noise)
noisedData = data.data:cmul(noisedData)

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-- print '==> Loading Validation Data'
-- testData = torch.load(opt.tedata)

-- if type(testData) == "userdata" then
-- 	testData = {
-- 	data = testData;
-- 	}
	
-- else
	
-- 	-- testData.data = testData.data[{{1,50000}}]
-- 	-- testData.labels = testData.labels[{{1,50000}}]

-- 	if opt.size == 'small' then
-- 		testData.data = testData.data[{{1,10000}}]
-- 		testData.labels = testData.labels[{{1,10000}}]
-- 	end
-- end


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

print '==> Defining Model'

py = require('fb.python')

py.exec([=[
import numpy
rng = numpy.random.RandomState(123)
W = []
W.append(numpy.asarray(rng.uniform(low=-4*numpy.sqrt(6./(1764+2500)),high=4*numpy.sqrt(6./(1764+2500)),size=(1764,2500))))
W.append(numpy.asarray(rng.uniform(low=-4*numpy.sqrt(6./(2500+1000)),high=4*numpy.sqrt(6./(2500+1000)),size=(2500,1000))))
W.append(numpy.asarray(rng.uniform(low=-4*numpy.sqrt(6./(1000+500)),high=4*numpy.sqrt(6./(1000+500)),size=(1000,500))))
	]=])

weights = py.eval('W')

model = nn.Sequential()
model:add(nn.Linear(pSize,opt.nHidden))
model:add(cudnn.Sigmoid())
model:add(nn.Linear(opt.nHidden,pSize))

-- model:get(1).weight = weights[opt.layerNo]:t()

local method = 'xavier'
model = require('weight-init')(model, method)

model:get(3).weight = model:get(1).weight:t()
model:get(3).gradWeight = model:get(1).gradWeight:t()

model:get(1).bias:zero()
model:get(3).bias:zero()

model = model:cuda()



print '==> Defining Loss Function'

criterion = nn.MSECriterion():cuda()

print '==> Here is the model'
print(model)

-------------------------------------------------------------------------------
dofile('pretrain.lua')
-------------------------------------------------------------------------------


print '==> Starting Training..'

bestScore = 1000000
epoch = 1
for i = 1,opt.epochs do
	train(model, data, noisedData)
	-- if( i%opt.validFreq == 0 ) then
	-- 	test()
	-- end
	epoch = epoch + 1
end



