%% Test for n==1: Cache_Matrix=0
clc; clear; close all;
 U=3; F=4;S_u=2;n=1;s_f=1;r=0.5; T=F+1; gamma = 0.56;
N_U = [1 1 1; 1 1 1;1 1 1]; 
R_BS_UE = [3.8527e+07;4.7338e+7;3.8527e+7];
R_D2D   = [Inf 73037159.3473631 53308532.7240431;73037159.3473631 Inf 73037159.3473631;53308532.7240431 73037159.3473631 Inf];
Temp_user_req_matrix = [1 2 0 1;1 1 0 2; 1 2 1 3];
Cache_Matrix =         [1 0 1 0; 0 1 0 1; 1 0 0 1];
History_Cache =        [3 2 4 5;1 4 2 3; 2 3 5 6];
%Cache_Matrix = zeros(U,F);
a_uf = [0 1];
%% 12: Based on 11, we calculate the Q_{u,f}(a_{u,f} = 1) and Q_{u,f}(a_{u,f} = 0) 
%
%In this paper, instead of considering all the users which is increasing
%the super action, we only consider user u and its neighbor to decrease the action space.

R_Matrix_0 = zeros(U,F);%Reward matrix for a_{u,f}=0
R_Matrix_1 = zeros(U,F);%Reward matrix for a_{u,f}=1
C_uf_0 = zeros (U,F); % Count number of time user u take action a_{u_f}=0
C_uf_1 = zeros (F,F); % Count number of time user u take action a_{u_f}=1
for a = 0:1:1
   for u=1:1: U
      for f=1:1:F
        latency_BS_UE = s_f/R_BS_UE(u,:); 
        if (n==1)
            id_req = find (Temp_user_req_matrix(u,f)~=0,2);% Declare user u request file f or not
            if ~isempty(id_req)% isempty: If Nei_u_f empty, it return 1, otherwise 0
                if a==1 % user u cache file f, it serves for itself with no delay
                    R_Matrix_1(u,f) = R_Matrix_1(u,f)+ latency_BS_UE;
                else % a==0, n==1 so we have no cache. Transmission from BS to user u            
                    R_Matrix_0(u,f) = R_Matrix_0(u,f);
                end % a==1
            end % ~isempty(id_req) 
        end % n==1
      end % for f=1:1:F
    end % for u=1:1:U 
end% for a = 1:1:size(a_uf,2)
%% 8: Update Q_{u,f}(b_{u,f}):avarage reward of joint action b_{u,f} of user u and its neighbors (i.e., u2 and u3) with each file f
 % First, we need to clarify C_{u,f}(b_{u,f}) based on Temp_user_req_matrix   
 % Second, we can calculate Q_(b_{u,f}) as step 3 in belief-based modified CUCB based on R_Matrix
 % Third, updating the Q_{u,f}(b_{u,f}) as in Algorithm 1
        C_uf_b_uf = zeros (U,F);
        Q_b_uf_0 = zeros (U,F);
        Q_b_uf_1 = zeros (U,F);
        Q_uf_b_uf_0 = zeros (U,F);
        Q_uf_b_uf_1 = zeros (U,F);
     for u=1:1:U
        for f=1:1:F
            Temp_C_uf_b_uf = sum(Temp_user_req_matrix(:,f));
            Temp_Q_b_uf_0 = sum(R_Matrix_0(:,f));
            Temp_Q_b_uf_1 = sum(R_Matrix_1(:,f));
            C_uf_b_uf(u,f)= C_uf_b_uf(u,f) + Temp_C_uf_b_uf;
            Q_b_uf_0(u,f)= Q_b_uf_0(u,f) + (Temp_Q_b_uf_0)/U;
            Q_b_uf_1(u,f)= Q_b_uf_1(u,f) + (Temp_Q_b_uf_1)/U;
            Q_uf_b_uf_0(u,f) = Q_b_uf_0(u,f) + 1/(C_uf_b_uf(u,f)+1)*(R_Matrix_0(u,f)-Q_b_uf_0(u,f));
            Q_uf_b_uf_1(u,f) = Q_b_uf_1(u,f) + 1/(C_uf_b_uf(u,f)+1)*(R_Matrix_1(u,f)-Q_b_uf_1(u,f));
 % 9. Update C_{u,f}(a_{u,f}) and C__{u,v,f}(a_{u,f}).       
            C_uf_a_uf(u,f) = Temp_user_req_matrix(u,f);% Number of times actions a_{u,f} is selected by agent u
            C_uvf_a_uf(u,f) = C_uf_b_uf(u,f) - C_uf_a_uf(u,f);% Number of times action a_{u,f} is selected by neighbors of u, i.e., agent v
        end
     end
        
   %% 10 Compute Pr_{u,f}(b_{u,f}^(-u))
        Pr_uvf_a_vf = C_uvf_a_uf/T;
        Pr_uf_b_uf = prod(Pr_uvf_a_vf,2); % prof function:  the product of the elements in each row. 
%% 12: Based on 11, we calculate the Q_{u,f}(a_{u,f} = 1) and Q_{u,f}(a_{u,f} = 0) 
       Q_uf_0 = Q_uf_b_uf_0.* Pr_uf_b_uf;
       Q_uf_1 = Q_uf_b_uf_1.* Pr_uf_b_uf;
       Q_uf   = Q_uf_1 - Q_uf_0;
       l = max (Q_uf/s_f); % Because we only use one s_f value


          %if (Temp_user_req_matrix(u,f)~=0)
          % Finding the neighbor of user u in N_U matrix
            %Nei_u =  find(N_U(u,:)~=0);%Finding Neighbors ID of user u          
                %if ~isempty(Nei_u)% isempty: If Nei_u_f empty, it return 1, otherwise 0
                    %for i=1:1:size(Nei_u,2)
                    %temp_latency = s_f/R_D2D(u,Nei_u(:,i));
                    %end
                    
                %else % u have no neighbors, transmission from BS. R_Matrix unchange
                    
                %end
            %end