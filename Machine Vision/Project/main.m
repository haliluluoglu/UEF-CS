%%Taking path of the image that you want to count bottles in crate.
path=input("Please enter path of the image: ", 's')
fprintf("Your entered this path: %s", path)

%%Read the image and show the image.
original_image = imread(path);
imshow(original_image);
%%Convert to image into black and white for processing and show the image.
black_and_whiteImage = im2bw(original_image);
imshow(black_and_whiteImage);

%%Creating a disk-shaped structure element, where radious=2, the number of
%%line structuring element=6 for dilation process.
SE=strel('disk', 2,6);
%%Applying dilation to complete broken circles of the bottle.
dilated_image=imdilate(black_and_whiteImage, SE);
imshow(dilated_image);
%%Creating a rectangle-shaped structure element, where element size of
%%[2,5] for erosion the image.
SE_2=strel('rectangle', [2,5]);
%%Creating a linear structure element, where length=3, angle degree=3 for
%%dilate the image.
SE_3=strel('line', 3,3);    
%%Applying erosion to clear white points from the image.
eroded_image=imerode(dilated_image, SE_2);
imshow(eroded_image);
dilated_image2=imdilate(eroded_image, SE_3);
%imshow(dilated_image2);
%%Filling the circles from the bottles to get straight circles.
filled_image=imfill(dilated_image, 'holes');
imshow(filled_image);

%%Finding the circles with HOUGH TRANSFORM with radious range=[18,55]
[centers, radii, metric] = imfindcircles(filled_image,[18 55]);
%%draws circles with specified centers and radii onto the current axes.
h = viscircles(centers,radii);
xlabel([num2str(numel(radii)),' bottles']);