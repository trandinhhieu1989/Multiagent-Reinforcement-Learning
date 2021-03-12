%% This file is to test the random number created by using Zipf rand file.
clc; clear; close all;
skewness = 0.9; %Shape factor of Zipf distribution
F = 20; % Total number of files
M = 10; % Number of samples creating each running times
sample = zipf_rand(F, skewness, M);
