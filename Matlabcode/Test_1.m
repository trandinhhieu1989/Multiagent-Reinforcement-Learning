%% Test for n==1: Cache file having high temp_user_request and satisfy the capacity 
% This code cache files based on the number of requests.
clc; clear; close all;
 U=3; F=4;S_u=2;n=1;
Temp_user_req_matrix = [3 2 1 1;2 0 0 0; 1 0 3 0];
 Cache_Matrix = zeros(U,F);
 R_Matrix = zeros(U,F);
for u=1:1:U
    for f=1:1:F
        if n==1
        id_req = find (Temp_user_req_matrix(u,:)~=0,2);% Finding which file f is requested by user u
        R_Matrix(u,id_req )=0;
        if ~isempty(id_req)
           Des_req = sort(Temp_user_req_matrix(u,:),'descend');% Sorting the file requests of user u in descending rows
           limit = min(size(id_req,2),S_u);%Due to the limited cache size of each user
           for i=1:1:limit    
               cache_files = find(Temp_user_req_matrix(u,:)==Des_req(:,i));% Determining which file to cache
               %id_cache_file= min(cache_files );% If more than 1 file have same no of request, choosing the file having higher content popularity
               %In Zipf distribution, the popularity is decreasing. Therefore, f1 has highest popularity
               if (size(cache_files,2) > 1)
                   for a=1:1:size(cache_files,2)
                       if (Cache_Matrix(u,cache_files(:,a))==0)&&(sum(Cache_Matrix(u,:))<limit)
                           Cache_Matrix(u,cache_files(:,a))=1;
                       else
                           break;
                       end
                   end
               else
                   Cache_Matrix(u,cache_files)=1;% In case size(cache_files,2) == 1
               end
           end             
        end
        end%n==1
    end
end