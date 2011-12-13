%Script align_Faces
%
% Aligns the eye positions of a directory of face images. Reads in Labels.mat, 
% obtained using getLabels.m, writes jpegs to a specified directory.
% Specify directory paths at the top of the file. 
%LabelDir is where Labels.mat is.
%imgDir is where the original images are
%DestDir is where you want the cropped jpegs to go. 

LabelDir = 'C:\Users\sy483\Desktop\ICA'
imgdir = 'C:\Users\sy483\Desktop\ICA\imgdir'
DestDir = 'C:\Users\sy483\Desktop\ICA\testdir'
  
homeDir = 'C:\Users\sy483\Desktop\ICA'

% CHANGE THESE VARIABLES AS NEEDED
%XSIZE =  YSIZE = 	%Size of desired cropped image
%EYES = %Number of pixels desired between the eyes
%TEETH_EYES = %Desired no. of pixels from teeth to eyes. 

XSIZE = 128; YSIZE = 128;      
EYES = 30;
TEETH_EYES = 62; 
%SEE ALSO PARAMETERS IN CROP ROUTINE. EYES BELOW MIDPOINT.
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Here is where we load the images and do the preprocessing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      cd (LabelDir)
      load Labels
      cd (imgDir)
      r = dir;

for i = 3:(size(r,1))
   imgName = r(i).name
   %[X,map] = imread([ t ]);
      I=imread(imgName);
      I=rgb2gray(I);

     %Extract face measurements:
     %dxeyes =  x distance between eyes in original image                     
             % Average the locations of inner & outer corners.
     %dyeyes =  y distance between eyes in original image 
     %dEeyes = Euclidean norm distance between eyes
     %dEteeth_eyes = Euclidean norm distance from teeth to midpt b/neyes
                 
     [height, width] = size(I);
     dxeyes = marks(i-2,3) - marks(i-2,1); %Check these. 
     dyeyes = marks (i-2,4) - marks(i-2,2);
     dEeyes = sqrt(dxeyes^2 + dyeyes^2);
     mean_eye_x = mean([marks(i-2,1), marks(i-2,3)]);
     mean_eye_y = mean([marks(i-2,2), marks(i-2,4)]);
     dEteeth_eyes = sqrt((marks(i-2,5)-mean_eye_x)^2 + (marks(i-2,6)-mean_eye_y)^2); 
                                          
     %scale
     yscale = TEETH_EYES / dEteeth_eyes; xscale = EYES / dEeyes;
     height_new = yscale*height; width_new = xscale*width;
     tmp0=imresize(I,[height_new,width_new],'bicubic');
   
      %rotate (Problem: imrotate rotates about the center of the image. 
      %To avoid losing feature position information, must first center the 
      %image on the right eye before rotating. 
      %Then use right eye position to determine cropping.
      Reye_x = marks(i-2,1);
      Reye_y = marks(i-2,2);

      %crop a 200x200 window centered on left eye:
      %Zero-pad to make sure window never falls outside of image. 
      %W = 100; %Window radius
      W = 500;  %For bigger images (Gwen's params).
      padcols = zeros(size(tmp0,1),W); padrows = zeros(W,size(tmp0,2)+W);
      padcols = uint8(padcols); padrows=uint8(padrows);
      tmp = [padrows;padcols,tmp0];

      tmpx = xscale*Reye_x - W +W; tmpy = yscale*Reye_y - W +W;     
      tmp1 = imcrop(tmp,[tmpx,tmpy,2*W,2*W]);
      %figure(2);imshow(tmp1)

      angle = 180/pi*atan((yscale*dyeyes)/(xscale*dxeyes));
      tmp2 = imrotate(tmp1,angle,'bicubic','crop');
      %figure(2); imshow(tmp2);

      %crop
      % x and y give the upper left corner of cropped image
      % Reye is centered at (W,W) = (100,100).
      % For bigger images (W,W) = (500,500)
      x = W - (XSIZE-EYES)/2;
      %y = W - YSIZE/2;   %Eyes at midpoint
      y = W - YSIZE*1/3;  %Face box
      tmp3=imcrop(tmp2,[x,y,XSIZE,YSIZE]);
      figure(1); imshow(tmp3);

      %save
      [imgName, R] = strtok(imgName, '.');
      fname = [DestDir,imgName, '.pgm'];
      imwrite(tmp3,fname,'pgm') 
end
