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
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-learningRateDecay',1e-07,'LR decay')
cmd:option('-nHidden',1000,'Number of hidden neurons')
cmd:option('-noise',0.2,'Masking noise')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | adagrad | rmsprop')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-loader','preProcessed.t7','Data: preProcessed.t7')
cmd:option('-size','normal', 'size: normal | small (for faster experiments)')
cmd:text()

opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


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
elseif opt.optimization == 'rmsprop' then
	optimState = {
		learningRate = opt.learningRate,
	}
	optimMethod = optim.rmsprop
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

print '==> Defining Model'

model = nn.Sequential()
model:add(nn.Linear(pSize,opt.nHidden))
model:add(nn.Sigmoid())
model:add(nn.Linear(opt.nHidden,pSize))

model:get(3).weight = model:get(1).weight:t()
model:get(3).gradWeight = model:get(1).gradWeight:t()

model = model:cuda()

print '==> Defining Loss Function'

criterion = nn.MSECriterion():cuda()

print '==> Here is the model'
print(model)

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

print '==> Loading parameters'
parameters, gradParameters = model:getParameters()

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

function train(model, data, noisedData)

	err = 0

   -- epoch tracker
	epoch = epoch or 1

	-- local vars
	local time = sys.clock()

	model:training()

	-- do one epoch
	-- print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

	for t = 1,data.data:size(1),opt.batchSize do
	-- disp progress
		xlua.progress(t, data.data:size(1))

		local inputs = noisedData[{{t,math.min(t+opt.batchSize-1,noisedData:size(1))}}]:cuda()
		local targets = data.data[{{t,math.min(t+opt.batchSize-1,data.data:size(1))}}]:cuda()

		local feval = function(x)
		           -- get new parameters
			if x ~= parameters then
		  		parameters:copy(x)
			end

			inputs = inputs:cuda()
			targets = targets:cuda()

			-- reset gradients
			gradParameters:zero()

			local outputs = model:forward(inputs)
			local f = criterion:forward(outputs,targets)

			local df_do = criterion:backward(outputs, targets)
			model:backward(inputs, df_do)

			-- if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
			--    -- locals:
			--    local norm,sign= torch.norm,torch.sign

			--    -- Loss:

			--    f = f + opt.coefL1 * norm(parameters,1)
			--    f = f + opt.coefL2 * norm(parameters,2)^2/2

			--    -- Gradients:
			--    gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
			-- end


			return f,gradParameters
			end

		-- optimize on current mini-batch
		if optimMethod == optim.asgd then
			_,_,average = optimMethod(feval, parameters, optimState)
		else
			_,fs = optimMethod(feval, parameters, optimState)
		end

		err = err + fs[1]

   end

   -- time taken
   time = sys.clock() - time
   time = time / data.data:size(1)
   -- print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   err = err / data.data:size(1)

   print("\n==> Reconstruction Error: "..(err))

	trainLogger:add{['% mean class accuracy (train set)'] = err}
	if opt.plot then
		trainLogger:style{['% mean class accuracy (train set)'] = '-'}
		trainLogger:plot()
	end

	-- save/log current net
	local filename = paths.concat(opt.save, 'model.net')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	-- print('==> saving model to '..filename)
	torch.save(filename, model)

end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------


print '==> Starting Training..'

epoch = 1
for i = 1,100 do
	train(model, data, noisedData)
	epoch = epoch + 1
end



