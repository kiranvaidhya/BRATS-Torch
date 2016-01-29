 ----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'cudnn'

----------------------------------------------------------------------
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
   print('==> testing on test set:')
   for t = 1,testData.data:size(1),opt.batchSize do

      xlua.progress(t, testData.data:size(1))

      local testInputs = testData.data[{{t,math.min(t+opt.batchSize-1,testData.data:size(1))}}]:cuda()
      local testTargets = testData.labels[{{t,math.min(t+opt.batchSize-1,testData.data:size(1))}}]:cuda()

      predictions = model:forward(testInputs)

      for k = 1,testInputs:size(1) do
         confusion:add(predictions[k],testTargets[k])   
      end

   end

   -- timing
   time = sys.clock() - time
   time = time / testData.data:size(1)
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   if (confusion.totalValid * 100) > bestScore then
      local filename = paths.concat('results',opt.save, 'model.net')
      os.execute('mkdir -p ' .. sys.dirname(filename))
      print('==> saving model to '..filename)
      torch.save(filename, model)
      bestScore = confusion.totalValid * 100
   end

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
  
   -- next iteration:
   confusion:zero()
end
