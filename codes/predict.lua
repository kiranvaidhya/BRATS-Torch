py = require('fb.python')
require 'cudnn'
require 'optim'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing Images')
cmd:text()
cmd:text('Options')
cmd:option('-batchSize',10000)
cmd:option('-patchSize',21)
cmd:option('-path', 'testing', 'training | validation | testing')
cmd:text()

params = cmd:parse(arg)
batchSize = params.batchSize
patchSize = params.patchSize

if params.path == 'training' then
	py.exec([=[
path = '../HeadNeck/data/Training'
		]=])
elseif params.path == 'validation' then
	py.exec([=[
path = '../HeadNeck/data/Validation'
		]=])
else
	py.exec([=[
path = '../../Testing/N4_zscore_testing_t1_t1c_hist_match'
		]=])
end


model = torch.load('results/f4/model.net')

model:evaluate()

py.exec([=[

from sklearn.feature_extraction import image as extractor
import nrrd
import os
import mha
import nibabel as nib
import getopt
import sys
import numpy as np
from scipy.ndimage import zoom

patchSize = 21

images = []
folders = []

Flair = []
T1 = []
T2 = []
T_1c = []
folders = []
Truth=[]
Subdir_array = []

for subdir, dirs, files in os.walk(path):
	for file1 in files:
		if file1[-3:]=='nii' and ( 'Flair' in file1):
			Flair.append(file1)
			folders.append(subdir+'/')
			Subdir_array.append(subdir[-5:])
		elif file1[-3:]=='nii' and ('T1' in file1 and 'T1c' not in file1):
			T1.append(file1)
		elif file1[-3:]=='nii' and ('T2' in file1):
			T2.append(file1)
		elif file1[-3:]=='nii' and ('T1c' in file1 or 'T_1c' in file1):
			T_1c.append(file1)
		elif file1[-3:]=='mha' and 'OT' in file1:
			Truth.append(file1)     

i = 0
]=])

print(py.eval('Flair'))

for imageIterator = 1,py.eval('len(Flair)') do

	py.exec([=[

print
print
print '####################################################################'
print '==> Predicting Image: ', i+1
print '    Folder: ', folders[i]
print '####################################################################'
print
print

flairImage = nib.load(folders[i]+Flair[i])
T1Image = nib.load(folders[i]+T1[i])
T2Image = nib.load(folders[i]+T2[i])
T_1cImage = nib.load(folders[i]+T_1c[i])
folder = folders[i]

flairImage = flairImage.get_data()
T1Image = T1Image.get_data()
T2Image = T2Image.get_data()
T_1cImage = T_1cImage.get_data()

predictedImage = np.zeros(flairImage.shape)

i = i + 1
sliceIterator = 0
]=])


	print '==> Predicting each pixel slice by slice'

	for sliceIterator = 1,155 do

		py.exec([=[

patches = np.zeros((1,patchSize,patchSize))

flairPatch = extractor.extract_patches(flairImage[:,:,sliceIterator], patchSize, extraction_step = 1)
flairPatch = flairPatch.reshape(flairPatch.shape[0]*flairPatch.shape[1],patchSize,patchSize)

T1patch = extractor.extract_patches(T1Image[:,:,sliceIterator], patchSize, extraction_step = 1)
T1patch = T1patch.reshape(T1patch.shape[0]*T1patch.shape[1],patchSize,patchSize)

T2patch = extractor.extract_patches(T2Image[:,:,sliceIterator], patchSize, extraction_step = 1)
T2patch = T2patch.reshape(T2patch.shape[0]*T2patch.shape[1],patchSize,patchSize)

T1cpatch = extractor.extract_patches(T_1cImage[:,:,sliceIterator], patchSize, extraction_step = 1)
T1cpatch = T1cpatch.reshape(T1cpatch.shape[0]*T1cpatch.shape[1],patchSize,patchSize)

patches = np.concatenate([flairPatch,T1patch,T2patch,T1cpatch],axis=1)
sliceIterator = sliceIterator + 1

			]=])

		local patches = py.eval('patches')
		patches = patches:reshape(patches:size(1),patches:size(2)*patches:size(3))

		predictions = torch.Tensor(patches:size(1)):float()

		for i = 1, patches:size(1),batchSize do
			local batch = patches[{{i,math.min(i+batchSize-1,patches:size(1))}}]:cuda()
			posteriorProbabilities, predictedClasses = model:forward(batch):max(2)
			predictions[{{i,math.min(i+batchSize-1,patches:size(1))}}] = predictedClasses:float()
		end

	xlua.progress(sliceIterator,155)

	py.exec([=[
predictedSlice = pred
predictedSlice = predictedSlice.reshape(240-patchSize+1,240-patchSize+1)
predictedSlice = np.lib.pad(predictedSlice,((patchSize-1)/2,(patchSize-1)/2),'constant',constant_values=0)
predictedImage[:,:,sliceIterator-1] = predictedSlice

if sliceIterator == 155:
	predictedImage = predictedImage - 1
	predictedImage[np.where(predictedImage==-1)] = 0
	affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
	img = nib.Nifti1Image(predictedImage, affine)
	img.set_data_dtype(np.int32)
	print 'Saving in: ', folders[i-1]
	nib.save(img,folders[i-1]+'/torchPrediction.nii')
	]=],{pred = predictions})
	end
end