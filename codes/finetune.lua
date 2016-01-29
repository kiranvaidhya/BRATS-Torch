require 'cudnn'
require 'cunn'
require 'optim'
require 'randomkit'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-batchSize',96,'batchSize: 96')
cmd:option('-loader','data/f_train.t7','Data: preProcessed.t7')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-learningRateDecay',1e-07,'LR decay')
cmd:option('-nHidden',1000,'Number of hidden neurons')
cmd:option('-noise',0.2,'Masking noise')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | adagrad')
cmd:option('-mode','pretrained','mode: pretrained | random')
cmd:option('-coefL1',0,'coefL1')
cmd:option('-coefL2',0,'coefL2')
cmd:option('-epochs',800,'Number of epochs')
cmd:option('-save', 'finetuningResults', 'subdirectory to save/log experiments in')
cmd:text()

opt = cmd:parse(arg)

print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      learningRateDecay = opt.learningRateDecay
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd
elseif opt.optimization == 'adagrad' then
    optimState = {
      learningRate = opt.learningRate,
    }
    optimMethod = optim.adagrad

else
   error('unknown optimization method')
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

print '==> Loading Data..'
data = torch.load(opt.loader)

if opt.loader == 'preProcessed.t7' then
  pSize = data.data:size(3)
  data.data = data.data:reshape(data.data:size(1),pSize*pSize)
  pSize = pSize*pSize
else
  data.labels = data.labels + 1
  pSize = data.data:size(2)
end

if opt.mode == 'pretrained' then
	-- model = torch.load('results/ae1/model.net')
	-- model:remove(3)

	-- da2 = torch.load('results/ae2/model.net')
	-- da2:remove(3)

	-- model:add(da2:get(1))
	-- model:add(da2:get(2))

	-- da3 = torch.load('results/ae3/model.net')
	-- da3:remove(3)

 --  -- model:add(nn.Dropout(0.4))

	-- model:add(da3:get(1))
	-- model:add(da3:get(2))

 --  model:add(nn.Dropout(0.2))

	-- model:add(nn.Linear(500,5))

	-- model:get(8).weight:zero()
	-- model:get(8).bias:zero()

	-- model:add(cudnn.LogSoftMax())
  dofile('theanoweights.lua')

else
	model = nn.Sequential()
	model:add(nn.Linear(1024,opt.nHidden))
	model:add(nn.Sigmoid())
end

model = model:cuda()

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()



classes = {'1','2','3','4','5'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector

parameters,gradParameters = model:getParameters()

testData = torch.load('data/f_validate.t7')
testData.labels = testData.labels + 1
dofile('validate.lua')


function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,data.data:size(1),opt.batchSize do
      -- disp progress
      xlua.progress(t, data.data:size(1))

      local inputs = data.data[{{t,math.min(t+opt.batchSize-1,data.data:size(1))}}]:cuda()
      local targets = data.labels[{{t,math.min(t+opt.batchSize-1,data.data:size(1))}}]:cuda()

      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       local outputs = model:forward(inputs):cuda()
                       local f = criterion:forward(outputs, targets)

                        -- estimate df/dW
                       local df_do = criterion:backward(outputs, targets)
                       model:backward(inputs, df_do)

                       for i = 1,inputs:size(1) do
                          confusion:add(outputs[i], targets[i])
                       end

                       if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
                          -- locals:
                          local norm,sign= torch.norm,torch.sign

                          -- Loss:

                          f = f + opt.coefL1 * norm(parameters,1)
                          f = f + opt.coefL2 * norm(parameters,2)^2/2

                          -- Gradients:
                          gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
                       end

                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / data.data:size(1)
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   -- local filename = paths.concat(opt.save, 'model'..tostring(epoch)..'.net')
   -- os.execute('mkdir -p ' .. sys.dirname(filename))
   -- print('==> saving model to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

print '==> Starting Training..'

bestScore = 0
epoch = 1
for i = 1,opt.epochs do
	train()
  test()
end

