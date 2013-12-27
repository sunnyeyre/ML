%Script markFeatures
%For marking eye and mouth positions in face images.
%Writes a file called Labels.mat, in which each row indexes an image, and
%the columns are [x,y] positions of subject's right eye, [x,y] left eye,
%and [x,y] of mouth. Specify the image directory and the destination 
%directory (Where you want the labels saved) at the top of the script.


imgdir = 'Z:\ICA\imgdir'
destdir = 'Z:\ICA\testdir'

cd(imgdir)
files = dir('*.bmp')


C = []

%get marks
marks = [];
for i = 1:numel(files)
    t = files(i).name;
    [X,map] = imread(t);

    figure(1);
    colormap gray;
    if isfloat(X)
        image(gray2ind(mat2gray((X))));
    else
        image(X);
    end
    title(t);
    disp 'Click subjects right eye, left eye, then mouth.'
    [m,n] = ginput(3); pos = round([m,n]);
    pos = reshape(pos',1,6);
    marks = [marks; pos];
end

cd (destdir)
save Labels marks r
