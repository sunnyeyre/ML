% script Arch1.m
% Finds ICA representation of train and test images under Architecture I, 
% described in Bartlett & Sejnowski (1997, 1998), and Bartlett, Movellan & 
% Sejnowski (2002):  In Architecture I, we load N principal component 
% eigenvectors into rows of x, and then run ICA on x.
%
% Put the aligned training images in the rows of C, one image per row.  
% In the following examples, there are 500 images of aligned faces of size 
% 60x60 pixels, so C is 500x3600. 
%
% You can use the following matlab code to create C:
% markFeatures.m collects eye and mouth positions. 
% align_Faces.m crops, aligns, and scales the face images.
% loadFaceMat.m loads the images into the rows of C. 
%
% This script also calls the matrix of PCA eigenvectors organized in 
% the columns of V (3600x499), created by [V,R,E] = pcabigFn(C');
%
% The ICA representation will be in the rows of F (called B in Bartlett, 
% Movellan & Sejnowski, 2002): 

[V,R,E] = pcabigFn(C');
D = zeroMn(C')'; % D is 500x3600 and D = C-ones(500,1)*mean(C);
R = D*V; 	 % R is 500x499 and contains the PCA coefficients;

% We choose to use the first 200 eigenvectors. 
% (If PCA generalizes better by dropping first few eigenvectors, ICA will too).

x = V(:,1:150)'; 		% x is 200x3600
runica 				% calculates wz, w and uu. The matrix x gets 
				% overwritten by a sphered version of x. 
F = R(:,1:150) * inv(w*wz); 	% F is 500x200 and each row contains the 
			    	% ICA1 rep of an image

% Representations of test images under architecture I: 
% Put original aligned test images in rows of Ctest. 

Dtest = zeroMn(Ctest')'; % For proper testing, subtract the mean of the 
			 % training images not the test images: 
			 % Dtest = Ctest-ones(500,1)*mean(C);
Rtest = Dtest*V;
Ftest = Rtest(:,1:150) * inv(w*wz);

% Test nearest neighbor classification using cosine, not euclidean distance, 
% as similarity measure.
%
% First create label vectors. These are column vectors of integers. Lets 
% say our 500 training examples consisted of 500 different people. Then
% trainClass = [1:500]'; 
%
% We also need the correct class labels of the test examples if we want to 
% compute percent correct. Lets say the test examples were two images each 
% of the first 10 individuals. Then 
% testClass = [1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10]';

%We now compute percent correct:
train_ex = F';
test_ex = Ftest';
[pc,rankmat] = nnclassFn(train_ex,test_ex,trainClass,testClass);

%pc is percent correct of first nearest neighbor.
%rankmat gives the top 30 matches for each test image. 

