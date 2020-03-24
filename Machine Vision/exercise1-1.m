%%
pic1=imread("mammogram.tif");
imshow(f)
j=imadjust(pic1,'',[1 0]);
figure;
imshow(j);
%%
pic2=imread("pollen.tif");
img=histeq(pic2);
figure;
subplot(221); imshow(pic2);
subplot(222); imshow(img);
subplot(223); imhist(pic2);
subplot(224); imhist(img);
%%
pic=imread("pollen.tif");
img=histeq(pic);

im=myFunction(pic,1,200,300,500);
im2=myFunction(img,1,200,300,500);
figure;
subplot(221); imshow(pic);
subplot(222); imshow(img);
subplot(223); imshow(im);
subplot(224); imshow(im2);
