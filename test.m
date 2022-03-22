%%
close all;
clc;

%% variables
sigma  = 3.0;   
high_thresh = 0.0;    % default H and L = 0
low_thresh  = 0.0;

%% read image
img_rgb = imread('cat.jpg');
[X,Y] = size(rgb2gray(img_rgb)');
img = load('cat.txt');

%% run devernay function
[x, y] = devernay_edges(img, X, Y, sigma, high_thresh, low_thresh);

%% plot detected edges
figure(1);
imshow(img_rgb); hold on;
plot(x, y, 'm.', 'MarkerSize', 5); hold off;
title('Devernay edge detection');
