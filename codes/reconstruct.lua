require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'image'

model=torch.load('/home/brats/Kiran/BRATS/results/validate_da/model.net')
-- training_data=torch.load('/home/brats/Kiran/BRATS/data_pretrain.t7')
validation_data=torch.load('/home/brats/Kiran/BRATS/data/p_validate.t7')
-- train=training_data[{{100},{}]
valid=validation_data.data[100]
-- print (valid:size())
-- train_output=model.forward(train)

valid_output=model:forward(valid:cuda())
-- print (valid_output.size)
image.display(valid[{{1,441}}]:reshape(21,21))
image.display(valid_output[{{1,441}}]:reshape(21,21))

