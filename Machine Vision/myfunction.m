function outputImage = myfunction(inputImage,row1,row2,column1,column2)

img = inputImage(row1:row2, column1:column2);

outputImage = histeq(img);

