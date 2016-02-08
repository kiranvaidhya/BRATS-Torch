require 'cunn'
require 'cudnn'
require 'optim'
require 'randomkit'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
-- general options:
cmd:option('-batchSize',96,'batchSize: 96')
cmd:option('-nHidden',1000,'Number of hidden neurons')
cmd:option('-loader','preProcessed.t7','Data: preProcessed.t7')
cmd:option('-model','results/model.net', 'Model')
cmd:option('-save','encoded.t7','Save Data: encoded.t7')
cmd:option('-size','normal', 'size: normal | small (for faster experiments)')
cmd:text()

opt = cmd:parse(arg)

print '==> Loading Data..'
data = torch.load(opt.loader)
if opt.loader == 'preProcessed.t7' then
	pSize = data.data:size(3)
	data.data = data.data:reshape(data.data:size(1),pSize*pSize)
	pSize = pSize*pSize
else
	if type(data) ~= "table" then
		data = {
		data = data
		}
	end
	pSize = data.data:size(2)
end

print '==> Data Loaded'

if opt.size == 'small' then
	data.data = data.data[{{1,100000}}]
end

model = torch.load(opt.model)
model:remove(3)

model:evaluate()

encodedData = torch.zeros(data.data:size(1),model:get(1).weight:size(1)):float()


for t = 1,data.data:size(1),opt.batchSize do
	-- disp progress
	xlua.progress(t, data.data:size(1))
	local inputs = data.data[{{t,math.min(t+opt.batchSize-1,data.data:size(1))}}]:cuda()
	encodedData[{{t,math.min(t+opt.batchSize-1,data.data:size(1))}}] = model:forward(inputs):float()
end

torch.save(opt.save,encodedData)