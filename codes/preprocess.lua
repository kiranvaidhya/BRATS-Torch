data = torch.load('train_32x32.t7','ascii')
data.data = data.data:float()

mean = torch.Tensor(60000)
std = torch.Tensor(60000)

for i = 1,60000 do
	mean[i] = data.data[i]:mean()
	std[i] = data.data[i]:std()
end


for i = 1,60000 do
	data.data[i] = data.data[i]:add(-mean:mean())
	data.data[i] = data.data[i]:div(std:mean())
end

torch.save('preProcessed.t7',data)