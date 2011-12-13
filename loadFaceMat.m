%function C = loadFaceMat(imgdir);
%
%Loads a directory of images into the rows of C.
%imgdir is a string with the path to the directory containing the images.
%For example:
imgdir = 'Z:\ICA\imgdir'

cd (imgdir)
files = dir('*.bmp')


C = []

for i = 1:numel(files)
    t = files(i).name;
        I=imread(t);
      tmp=mat2gray(double(I));
      tmp = reshape(tmp,1,size(tmp,1)*size(tmp,2));
      C = [C;tmp];
end