% function [S] = cosFn(mat1,mat2),
% Computes the cosine (normalized dot product) between training vectors in 
% columns of mat1 and test vectors in columns of mat2. Outputs a matrix of 
% cosines (similarity matrix). 
%
% Written by Ian Fasel.

function [S] = cosFn(mat1,mat2),
  
  denom = sum(mat1.^2,1)*sum(mat2'.^2,2);
  denom (find(denom==0)) = 0.00000000000000000000001;
  numer = mat1'*mat2;
  
  S = numer./denom;
