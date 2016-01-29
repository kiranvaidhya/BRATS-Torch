require 'cudnn'
require 'cunn'
require 'optim'
require 'randomkit'

if not opt then
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
	cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | adagrad | rmsprop | nag')
	cmd:option('-save', 'tmp', 'subdirectory to save/log experiments in')
	cmd:option('-trdata','data/s_pretrain.t7','Train Data: Pretraining data')
	cmd:option('-tedata','data/sv_pretrain.t7','Test Data')
	cmd:option('-coefL1',0,'coefL1')
	cmd:option('-coefL2',0,'coefL2')
	cmd:option('-size','normal', 'size: normal | small (for faster experiments)')
	cmd:option('-epochs',200,'epochs: number of epochs')
	cmd:option('-validFreq',5,'Validation Frequency: 5')
	cmd:text()
end


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
elseif opt.optimization == 'nag' then
	optimState = {
		learningRate = opt.learningRate,
		learningRateDecay = opt.learningRateDecay,
		momentum = opt.momentum,
		learningRateDecay = opt.learningRateDecay
	}
	optimMethod = optim.nag
else
   error('unknown optimization method')
end



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
	print("\n\n==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

	score = 0

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

			-- reset gradients
			gradParameters:zero()

			local outputs = model:forward(inputs)
			local f = criterion:forward(outputs,targets)

			local df_do = criterion:backward(outputs, targets)
			model:backward(inputs, df_do)

			if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
			   -- locals:
			   local norm,sign= torch.norm,torch.sign

			   -- Loss:

			   f = f + opt.coefL1 * norm(parameters,1)
			   f = f + opt.coefL2 * norm(parameters,2)^2/2

			   -- Gradients:
			   gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
			end

			for k = 1,inputs:size(1) do
				score = score + (outputs[k] - inputs[k])*(outputs[k] - inputs[k])
			end


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
   score = score / data.data:size(1)

   print("\n==> Reconstruction Error: "..(score))

	trainLogger:add{['% mean class accuracy (train set)'] = err}
	if opt.plot then
		trainLogger:style{['% mean class accuracy (train set)'] = '-'}
		trainLogger:plot()
	end
end



print '==> defining test procedure'

-- test function
function test()
   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end



   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print("\n==> testing on test set:")
   score = 0

   for t = 1,testData.data:size(1),opt.batchSize do

      xlua.progress(t, testData.data:size(1))

      local testInputs = testData.data[{{t,math.min(t+opt.batchSize-1,testData.data:size(1))}}]:cuda()

      reconstructions = model:forward(testInputs)

      for k = 1,testInputs:size(1) do
         z = reconstructions[k] - testInputs[k]
         score = score + z*z
      end

   end

   score = score/testData.data:size(1)

   if score < bestScore then
      local filename = paths.concat('results',opt.save, 'model.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('\n==> saving model to '..filename)
      torch.save(filename, model)
      bestScore = score      
   end


   print("\nValidation reconstruction score: "..(score))

   -- timing
   time = sys.clock() - time
   time = time / testData.data:size(1)


   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = score}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
  
end