function x = zipf_rand(N, expn, M)
% Generate random numbers based on Zipf distribution
% Author: Tuyen Tran (tuyen.tran@rutgers.edu). Oct 2015
%
% Reference: https://en.wikipedia.org/wiki/Zipf's_law
% N={1,2,...,N} denote a set of N files, each of which is randomly
% requested based on a popularity distribution. The n-file has popularity
% profile f_n, where f_n is sorted in a descending order with 
% sum (n=1 to N) fn = 1
% N         total Number of Elements or is the library size
% expn      the value of the Exponent characterizing the distribution
% M         Number of sample to be generated (in range [1,N])
%
% Example: zipf_rand(3,1,4)
% ans = 3 2 1 1

if nargin == 2
    M = 1;
end
%%
%% We solve the problem of how to get M files in total N file which follows
%% Zipf distribution when we know the 0=< CDF <= 1 and M chosen files having 
% M CDF which is uniform distribution, i.e., samples = rand (1,M). This
% becomes inverse CDF problem (we know the CDF value of x and find x )
%%
ranks = 1:1:N;
% pmf is the popularity distribution (request probability) of n-th element
% in a descending order.
pmf = (ranks.^(-expn))/sum(ranks.^(-expn));
% rand function: uniformly distributed random numbers
%% samples: is the CDF request probability of M files.
samples = rand(1,M);
% Cumsum(A) returns a cumulity sum of A at the beginning of the first array
% dimension in A whose size does not equal 1
p = cumsum(pmf(:));
%C = [A; B]  Vertically concatenate A and B
%[0;p/p(end)]Concatenating zero value and the p array vertically: since the
%request probability is from 0 to 1, so they add 0 value to the p array.
% WE CAN USE [0;p] INSTEAD OF [0;p/p(end)]
%[~,x]=histc(samples,[0;p/p(end)]): returns x, Count the number of values in samples that
%are within each specified bin range in[0;p/p(end)] 
%% https://fr.mathworks.com/help/matlab/ref/histc.html
%% bincount count the number of values in sameples that are within each specified bin range. 
%% x (or ind) indicates the bin numbers.
[bincount,x] = histc(samples,[0;p/p(end)])
%[~,x] = histc(samples,[0;p]);
end