%% 
close all;
clc;

%% hyper params
sigma       = 3.0;       
high_thresh = 0.0;    % default H and L = 0
low_thresh  = 0.0;

%% read images
%dog image
% img_rgb = imread('dog.jpg');

%cat image
img_rgb = imread('cat.jpg');

% convert to binary image
img_gray = rgb2gray(img_rgb);
img = img_gray;
img(img_gray>250) = 255;
img(img_gray<250) = 0;


%% run devernay function

[x, y] = devernay_edges(img, sigma, high_thresh, low_thresh);  % returns detected edges

%% plot detected edges

figure(1);
imshow(img_rgb); hold on;
plot(x, y, 'm.', 'MarkerSize', 5); hold off;
title('Devernay edge detection');



