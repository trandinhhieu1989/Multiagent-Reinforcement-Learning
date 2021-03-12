% This code is written by Tran Dinh Hieu on October 2, 2018.
% I tried to coded the paper: Efficient D2D content caching using multi-agent reinforcemnt learning, INFOCOM 2018.
%% WE TRY TO COMPARE BETWEEN BMCU ALGORITHM AND TOTAL (CUMMULITIVE) CONTENT POPULARITY ALGORITHM (CACHE FILES BASED ON CONTENT 
%% POPULARITY OF FILE AT ALL TIME PERIODS) AND TEMPORARY CONTENT POPULARITY (CACHE FILES BASED ON CONTENT 
%% POPULARITY OF FILE AT EACH TIME PERIODS) OVER TIME PERIODS N
%% THIS CODE DID NOT USE INITIALIZE STEP TO CREATE THE HIST_CACHE_MATRIX, R_MATRIX_0, R_MATRIX_1,USER_REQ_MATRIX,
%% Q_UF_B_UF_0, Q_UF_B_UF_1. 
clear; clc; close all;tic
%
%%  SIMULATION PARAMETERS
% 
Radius =  350; % Macrocell Radius
C = [400 400]; %// center [x y] of Macrocell, position of BS


P_BS_UE = 40; % transmission power of BS 40 Watt
P_D2D= 0.25; % transmission power of D2D 0.25 Watt
B = 10e6; % Total bandwidth for BS-UE links = Total bandwidth for D2D links
SPRU_BS_UE = 1; % Spatial reuse of BS-UE links
SPRU_D2D = 3; % Spatial reuse of D2D links
Sigma = -174; % Noise power -174 dBm/Hz

U = 4; % Total number of users in Macrocell
F = 100;   % Total number of files

s_f = 1; % Size of files
kappa = 1e-2;% Pathloss constant
epsilon = 4; % Pathloss exponent
gamma = 0.9; %Shape factor of Zipf distribution
mu   = F; % Avarage arrival rate of user's request 
r = 0.5; % Proportional factor of payment
T = F+1;% In this paper, they defined T as the total number of requests in time T

%The allocated bandwidth to each link given by total bandwidth x spatial reuse/number of users
B_BS_UE = B*SPRU_BS_UE/U;
B_D2D = B*SPRU_D2D/U;

% Transfer the noise power from dBm/Hz to dBm
P_dBm_BS_UE = Sigma + 10*log10(B_BS_UE);
P_dBm_D2D = Sigma + 10*log10(B_D2D);

% Transfer the noise power from dBm to Watt
P_W_BS_UE = 10^(P_dBm_BS_UE/10)*10^-3;
P_W_D2D = 10^(P_dBm_D2D/10)*10^-3;
M = 2; % Number of samples creating each running times. In this paper, they defined T=F+1 as the total number of requests in time T
N = 1000;%N: total number of time period T 

%% Creating the user's coordinate matrix based on function Topology(Radius,C,U)
%u_coor_matr = Fixed_Topology(Radius,C,U);
u_coor_matr = Topology(Radius,C,U);
%%
    S_uVec = [5];
    S_len = length(S_uVec);
    N_Vec = [1:1:N];
    N_len = length(N_Vec);
    Down_Latency = zeros(1,N_len); % Initialize the downloading latency in Fig. 2
    Down_Latency_UB = zeros(1,N_len); % Immediately content popularity
    Down_Latency_CP = zeros(1,N_len); % Total content popularity
    Number_Ser_D2D = zeros(1,N_len); % Initialize the number of file served by D2D or self-offloading
    Number_Ser_D2D_UB = zeros(1,N_len);% Immediately content popularity
    Number_Ser_D2D_CP = zeros(1,N_len);% Total content popularity
    CHR               = zeros(3,N_len);
    %S_idx = 1:S_len
for S_idx = 1:S_len 
 
    %u_coor_matr = Topology(Radius,C,U);
    User_req_matrix = zeros(U,F);%Creating the user request matrix: rows users, columns: files, elements are the number of time user i request file fn
    User_req_UB = zeros(U,F);
    Hist_Cache_Matrix = zeros(U,F); % BMCU: Cumulitive cache matrix of all the users
    Hist_Cache_UB = zeros(U,F); % immediately content popularity : Cumulitive cache matrix of all the users
    Hist_Cache_CP = zeros(U,F); % total content popularity : Cumulitive cache matrix of all the users
    Q_uf_b_uf_0 = zeros (U,F);
    Q_uf_b_uf_1 = zeros (U,F);
    Q_uf_1      = zeros (U,F);
    Q_uf_0      = zeros (U,F);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 2: OBSERVE (USER REQUEST PHASE IN PERIOD T)
% To prevent the case that cache capacity of user not full but we reset in
% new time slots. This is unreasonable
Cache_Matrix = zeros(U,F); % BMCU: cache matrix of each n value
Cache_UB = zeros(U,F); % UB algorithm: immediately content popularity
Cache_CP = zeros(U,F); % CP algorithm: total content popularity
for n=1:1:N
%% Finding the Temp_user_req_matrix and User_req_matrix
R_Matrix_0 = zeros(U,F);
R_Matrix_1 = zeros(U,F);
% Calculating the transmission rate between BS-UE and D2D
d_BS_UE = [];
    for i=1:1:U
    % Distance between BS to each UE
        d = sqrt((C(1)-u_coor_matr(i,1))^2+(C(2)-u_coor_matr(i,2))^2);
        d_BS_UE = [d_BS_UE; d];
    end
R_BS_UE = B_BS_UE.*log2(1+P_BS_UE.*kappa.*d_BS_UE.^(-epsilon)./P_W_BS_UE );
% Distance between each user and its neighbors
d_D2D = zeros(U,U);
for u=1:1:U
for v=1:1:U 
    if v==u
 d_D2D(u,v) = 0;
    else             
d = sqrt((u_coor_matr(u,1)-u_coor_matr(v,1))^2+(u_coor_matr(u,2)-u_coor_matr(v,2))^2);
d_D2D(u,v) = d;
    end
end 
end
R_D2D = B_D2D.*log2(1+P_D2D.*kappa.*d_D2D.^(-epsilon)./P_W_D2D );
%
%Finding the neighbors of user u1,u2,...,U, i.e., N'(u1): neighbor of u1
N_U= zeros(U,U);% rows: users, columns: neighbors, while the diagonal line always equal to 0: each user always have transmission rate higher than transmission from BS.
for u=1:1:U
    for v=1:1:U
    if (R_D2D(u,v) > R_BS_UE(u,:))&& (v~=u)%comparing between transmission rate 
        N_U(u,v) = 1;
    end
    end
end

% Creating the demand matrixs only for subset of user u1 and its neighbors IB_B_U
demand = zeros(U,100*F);
%d_sample = zeros(U,3*F);%Creating a square zero matrix in which the dimension is much bigger than F value
%ID_B_U = find(N_U(1,:)==1);% Subset of user u1 and its neighbors
for i =1:1:U
% Total number of request follows Possion process with rate mu
%M = poissrnd(mu);
sample = zipf_rand(F, gamma, M);
modified_sample = horzcat(sample, zeros(1,size(demand,2)-size(sample,2))); % Add to the same rows
demand(i,:) = modified_sample;
end

%   Creating the user request matrix: rows users, columns: files, elements are the number of time user i request file fn
% This matrix represent for Cu,f(a_{u,f}) and Cu,v,f(a_{u,f})
Temp_user_req_matrix = zeros(U,F);%User request matrix in this time T, this repesents for joint action matrix
% B_{u,f} of the observed user u and its neighbors v_i
for i=1:1:U
    for f=1:1:F
      %numel(find(d_sample(i,:)==f)) % number of elements of row i in matrix d_sample equal to file f's value
      Temp_user_req_matrix(i,f)= Temp_user_req_matrix(i,f)+numel(find(demand(i,:)==f));     
    end
end
%Count matrix: Cumulitive user request matrix in all time periods 
User_req_matrix = User_req_matrix + Temp_user_req_matrix;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize value for UPER BOUND algorithm IN WHICH CACHE THE HIGH
% POPULARITY FILES
    %Cache_UB = Cache_Matrix;
    %Hist_Cache_UB = Hist_Cache_Matrix;
    Temp_user_req_UB = Temp_user_req_matrix;
    User_req_UB = User_req_matrix;    
    
%% AVERAGE DOWNLOADING LATENCY: SUM DOWNLOADING LATENCY FOR ALL CONTENT REQUESTS TO THE TOTAL NUMBER OF CONTENT REQUESTS
%% CACHE HIT RATE: THE TOTAL NUMBER OF REQUESTS SERVED BY SELF-OFFLOADING OR D2D TRANMISSION
for u=1:1:U
    for f=1:1:F
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IMMEDIATELY CONTENT POPULARITY ALGORITHM 
 % CALCULATING ADL AND CHR IN UB
    latency_BS_UE = s_f/R_BS_UE(u,:);
    id_req_UB = find(Temp_user_req_UB(u,f)~=0);% Finding which file f is requested by user u
      if ~isempty(id_req_UB)
          if Cache_UB(u,f)==1 % If user u cache file f
             Number_Ser_D2D_UB(1,n)  = Number_Ser_D2D_UB(1,n)  + 1; % Increasing the number of self-offloading or D2D transmission
          else
             Nei_u_UB = find(N_U(u,:)==1);% Subset of user u1 and its neighbors
             if ~isempty(Nei_u_UB)
                 Nei_u_f_UB = [];
                 for i=1:1:size(Nei_u_UB,2)  
                    Nei_u_i_f_UB = find (Cache_UB(Nei_u_UB(:,i),f)~=0);% Finding which neighbors contain file f
                    % Add user ID which contains f file in Nei_u_f
                    if ~isempty(Nei_u_i_f_UB)
                        Nei_u_f_UB = [Nei_u_f_UB,Nei_u_UB(:,i)];
                    else
                        Nei_u_f_UB = [Nei_u_f_UB,Nei_u_i_f_UB];  
                    end                    
                 end %for i=1:1:size(Nei_u,2)   
% Finding the lowest latency from subset Nei_u_f to user u   
                if ~isempty(Nei_u_f_UB)
                    latency_UB = 1e10;% Initiating a very large number
                    for i=1:1:size(Nei_u_f_UB,2)
                        temp_latency_UB = s_f/R_D2D(u,Nei_u_f_UB(:,i));
                        if (temp_latency_UB < latency_UB )
                            latency_UB = temp_latency_UB;
                            chosen_nei_UB = Nei_u_f_UB(:,i);% Chosen neighbor is the neighbor nearest to user u and cache file f
                        end
                    end
                    Down_Latency_UB(1,n)  = Down_Latency_UB(1,n)  + latency_UB;
                    Number_Ser_D2D_UB(1,n)  = Number_Ser_D2D_UB(1,n)  + 1;
                else % No any neighbors of u cache file f, so transmission from BS
                    Down_Latency_UB(1,n)  = Down_Latency_UB(1,n)  + latency_BS_UE;                    
                end %if ~isempty(Nei_u_f_UB)
             else % transmission from BS
                 Down_Latency_UB(1,n)  = Down_Latency_UB(1,n)  + latency_BS_UE;
             end %if ~isempty(Nei_u_UB)
          end  %if Cache__UB(u,f)==1           
      end %if ~isempty(id_req_UB)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TOTAL CONTENT POPULARITY ALGORITHM 
 % CALCULATING ADL AND CHR IN UB
    latency_BS_UE = s_f/R_BS_UE(u,:);
    id_req_CP = find (Temp_user_req_UB(u,f)~=0);% Finding which file f is requested by user u
      if ~isempty(id_req_CP)
          if Cache_CP(u,f)==1 % If user u cache file f
             Number_Ser_D2D_CP(1,n)  = Number_Ser_D2D_CP(1,n)  + 1; % Increasing the number of self-offloading or D2D transmission
          else
             Nei_u_CP = find(N_U(u,:)==1);% Subset of user u1 and its neighbors
             if ~isempty(Nei_u_CP)
                 Nei_u_f_CP = [];
                 for i=1:1:size(Nei_u_CP,2)  
                    Nei_u_i_f_CP = find (Cache_CP(Nei_u_CP(:,i),f)~=0);% Finding which neighbors contain file f
                    % Add user ID which contains f file in Nei_u_f
                    if ~isempty(Nei_u_i_f_CP)
                        Nei_u_f_CP = [Nei_u_f_CP,Nei_u_CP(:,i)];
                    else
                        Nei_u_f_CP = [Nei_u_f_CP,Nei_u_i_f_CP];  
                    end                    
                 end %for i=1:1:size(Nei_u,2)   
% Finding the lowest latency from subset Nei_u_f to user u   
                if ~isempty(Nei_u_f_CP)
                    latency_CP = 1e10;% Initiating a very large number
                    for i=1:1:size(Nei_u_f_CP,2)
                        temp_latency_CP = s_f/R_D2D(u,Nei_u_f_CP(:,i));
                        if (temp_latency_CP < latency_CP )
                            latency_CP = temp_latency_CP;
                            chosen_nei_CP = Nei_u_f_CP(:,i);% Chosen neighbor is the neighbor nearest to user u and cache file f
                        end
                    end
                    Down_Latency_CP(1,n)  = Down_Latency_CP(1,n)  + latency_CP;
                    Number_Ser_D2D_CP(1,n)  = Number_Ser_D2D_CP(1,n)  + 1;
                else % No any neighbors of u cache file f, so transmission from BS
                    Down_Latency_CP(1,n)  = Down_Latency_CP(1,n)  + latency_BS_UE;                    
                end %if ~isempty(Nei_u_f_UB)
             else % transmission from BS
                 Down_Latency_CP(1,n)  = Down_Latency_CP(1,n)  + latency_BS_UE;
             end %if ~isempty(Nei_u_UB)
          end  %if Cache__UB(u,f)==1           
      end %if ~isempty(id_req_UB)      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PROPOSED BELIEF-BASED MODIFIED COMBINATORIAL UPER CONFIDENCE BOUND (BMCU)
      %latency_BS_UE = s_f/R_BS_UE(u,:);
      id_req = find (Temp_user_req_matrix(u,f)~=0);% Finding which file f is requested by user u
      if ~isempty(id_req)
          if Cache_Matrix(u,f)==1 % If user u cache file f
             Number_Ser_D2D(1,n)  = Number_Ser_D2D(1,n)  + 1; % Increasing the number of self-offloading or D2D transmission
          else
             Nei_u= find(N_U(u,:)==1);% Subset of user u1 and its neighbors
             if ~isempty(Nei_u)
                 Nei_u_f = [];
                 for i=1:1:size(Nei_u,2)  
                    Nei_u_i_f = find (Cache_Matrix(Nei_u(:,i),f)~=0);% Finding which neighbors contain file f
                    % Add user ID which contains f file in Nei_u_f
                    if ~isempty(Nei_u_i_f)
                        Nei_u_f = [Nei_u_f,Nei_u(:,i)];
                    else
                        Nei_u_f = [Nei_u_f,Nei_u_i_f];  
                    end                    
                 end %for i=1:1:size(Nei_u,2)   
% Finding the lowest latency from subset Nei_u_f to user u   
                if ~isempty(Nei_u_f)
                    latency = 1e10;% Initiating a very large number
                    for i=1:1:size(Nei_u_f,2)
                        temp_latency = s_f/R_D2D(u,Nei_u_f(:,i));
                        if (temp_latency < latency )
                            latency = temp_latency;
                            chosen_nei = Nei_u_f(:,i);% Chosen neighbor is the neighbor nearest to user u and cache file f
                        end
                    end
                    Down_Latency(1,n)  = Down_Latency(1,n)  + latency;
                    Number_Ser_D2D(1,n)  = Number_Ser_D2D(1,n)  + 1;
                else % No any neighbors of u cache file f, so transmission from BS
                    Down_Latency(1,n)  = Down_Latency(1,n)  + latency_BS_UE;                    
                end %if ~isempty(Nei_u_f)
             else % transmission from BS
                 Down_Latency(1,n)  = Down_Latency(1,n)  + latency_BS_UE;
             end %if ~isempty(Nei_u)
          end  %if Cache_Matrix(u,f)==1           
      end %if ~isempty(id_req)
    end %for f=1:1:F

end %for u=1:1:U    
    
%% CACHE PLACEMENT PHASE
for u=1:1:U    
    for f=1:1:F        
      %% BMCU: FINDING WHICH FILES NEED TO BE CACHED
      %% Finding the R_matrix, R_Matrix_0, R_Matrix_1
      latency_BS_UE = s_f/R_BS_UE(u,:);
         for a=0:1:1 
            if a==0
            % Finding the neighbor of user u in N_U matrix
            Nei_u =  find(N_U(u,:)~=0);%Finding Neighbors ID of user u          
            if ~isempty(Nei_u) % isempty: If Nei_u_f empty, it return 1, otherwise 0
            % Finding the neighbors contain file f or not, if yes we select the neighbor having lowest latency   
                Nei_u_f = [];
                chosen_nei = [];
                for i=1:1:size(Nei_u,2)  
                    Nei_u_i_f = find (Cache_Matrix(Nei_u(:,i),f)~=0);% Finding neighbor Nei_u(:,i) contain file f or not
                    % Add user ID which contains f file in Nei_u_f
                    if ~isempty(Nei_u_i_f)% Yes
                        Nei_u_f = [Nei_u_f,Nei_u(:,i)];
                    else
                        Nei_u_f = [Nei_u_f,Nei_u_i_f];  
                    end                    
                end    
% Finding the lowest latency from subset Nei_u_f to user u   
                if ~isempty(Nei_u_f)
                    latency = 1e10;% Initiating a very large number
                    for i=1:1:size(Nei_u_f,2)
                    temp_latency = s_f/R_D2D(u,Nei_u_f(:,i));
                        if (temp_latency < latency )
                            latency = temp_latency;
                            chosen_nei = Nei_u_f(:,i);% Chosen neighbor
                        end
                    end
                 % Calculating the Reward as in (1) and (2)
                      R_u_f = latency_BS_UE - latency;% Eq.(1)
                      if u~= chosen_nei
                        R_Matrix_0(u,f) = R_Matrix_0(u,f)+(1-r)*R_u_f; %Eq. (2), user u pay incentive r for chosen_nei
                        R_Matrix_0(chosen_nei,f)= R_Matrix_0(chosen_nei,f)+r*R_u_f; %Eq. (2), the chosen neighbor can receive incentive r   
                      end              
                    
                end %~isempty(Nei_u_f)
            end %~isempty(Nei_u)
            elseif a==1 % Self-offloading
               R_Matrix_1(u,f) = latency_BS_UE;
            end % end a==0
         end % for a=0:1:1  
       %% Caculate the Q value as in multi-arm bandit: Q_n+1 = Q_n + (R_n - Q_n)/n 
        ID_B_U = find(N_U(u,:)==1);% Subset of user u1 and its neighbors
        N_U_1 = 0;
        N_U_0 = 0; 
        Requ_times = 0;
%Finding the total number of times user u cache file f so far (in the past)
        N_U_1 = N_U_1 + Hist_Cache_Matrix(u,f);
%Finding the total number of times user u did not cache file f so far (in the past)        
        Requ_times = Requ_times + User_req_matrix(u,f); 
        if ~isempty(ID_B_U) % Eliminate the case if have neighbors, we can
        %calculate the Q values
            Pr_chosen_nei_0 = 1;
            for i=1:1:length(ID_B_U)
 %Finding the total number of times user u and its neighbor N_U cache file f so far (in the past)           
                N_U_1 = N_U_1 + Hist_Cache_Matrix(ID_B_U(i),f); 
 %Finding the total number of times user u and its neighbor N_U request file f so far (in the past)
                Requ_times = Requ_times + User_req_matrix(ID_B_U(i),f);   
 % Probability that all neighbors did not cache file f
                % Utilizing the local information which is the total number of request of all users
                %% To prevent caculate Ave_Q_uf(u,b)=-inf when log(sum(User_req_matrix(u,:))=-inf when sum(User_req_matrix(u,:)=0
                if (sum(User_req_matrix(ID_B_U(i),:))==0)
                    Pr_chosen_nei_0 =0;
                else
                    Pr_chosen_nei_0 = Pr_chosen_nei_0*(sum(User_req_matrix(ID_B_U(i),:))-Hist_Cache_Matrix(ID_B_U(i),f))/sum(User_req_matrix(ID_B_U(i),:));
                end
                 % Utilizing the global information which is the total number of request of all users
                %Pr_chosen_nei_0 = Pr_chosen_nei_0*(sum(sum(User_req_matrix))-Hist_Cache_Matrix(ID_B_U(i),f))/sum(sum(User_req_matrix));
            end
 %Finding the total number of times user u and its neighbor N_U did not cache file f so far (in the past)
                if (Requ_times==0)|| (Requ_times <N_U_1)% When S_u=F some files did not request but still cached, it lead to Requ_times <N_U_1Requ_times <N_U_1
                    N_U_0 = 0;
                elseif  (Requ_times >= N_U_1)
                    N_U_0 = Requ_times - N_U_1;
                else
                    N_U_0 = 0;
                end
 %% 8. Calculating as in Algorithm 1    
        Q_uf_b_uf_0(u,f)=  Q_uf_b_uf_0(u,f) + (R_Matrix_0(u,f) - Q_uf_b_uf_0(u,f))/(N_U_0+1);            
        Q_uf_b_uf_1(u,f)= Q_uf_b_uf_1(u,f) + (R_Matrix_1(u,f) - Q_uf_b_uf_1(u,f))/(N_U_1+1);     
 %% 10. We only compute the Probability of nearest neighbor cache file f
        if ~isempty(chosen_nei)
            % Utilizing the local information which is the total number of request of all users
            if sum(User_req_matrix(chosen_nei,:))==0
                Pr_chosen_nei_1 = 0;
            else
                Pr_chosen_nei_1 = Hist_Cache_Matrix(chosen_nei,f)/sum(User_req_matrix(chosen_nei,:));
            end
            % Utilizing the global information which is the total number of request of all users
            %Pr_chosen_nei_1 = Hist_Cache_Matrix(chosen_nei,f)/sum(sum(User_req_matrix));
            Q_uf_0(u,f) = Q_uf_b_uf_0(u,f)*Pr_chosen_nei_1;
        else % if no any neighbors cache file f, we need to transmit from BS           
            Q_uf_0(u,f) = Q_uf_b_uf_0(u,f)*Pr_chosen_nei_0; 
        end        
 %% 11.       
        Q_uf_1(u,f) = Q_uf_b_uf_1(u,f); 
        else % If we do not have any neighbors or no any neighbors satisfy the transmission rate higher than from BS
            Q_uf_0(u,f) = 0;
            Q_uf_b_uf_1(u,f)= Q_uf_b_uf_1(u,f) + (R_Matrix_1(u,f) - Q_uf_b_uf_1(u,f))/(N_U_1+1);
            Q_uf_1(u,f) = Q_uf_b_uf_1(u,f); 
        end %~isempty(ID_B_U)
%% 12
        % Benifits
        Q_uf(u,f) = Q_uf_1(u,f)-Q_uf_0(u,f);
    end % f=1:1:F
        % Calculate the average benifits using belief-based modified CUCB algorithm
        l = max(Q_uf(u,:)/s_f); % Because we only use one s_f value
        C_uf_1 = Hist_Cache_Matrix; % Instead of increase value of Hist_Cache_Matrix from 0 to 1, we change value of C_uf_1 matrix
        for b=1:1:F
            if C_uf_1(u,b)==0
                C_uf_1(u,b)=0.5;
            end
            % Utilizing the local information which is the total number of request of all users
            if sum(User_req_matrix(u,:))==0
                Ave_Q_uf(u,b) = Q_uf(u,b);
            else
                Ave_Q_uf(u,b) = Q_uf(u,b) + l*(numel(ID_B_U)+1)*s_f/(F^gamma)*sqrt(3*log(sum(User_req_matrix(u,:)))/(2*C_uf_1(u,b)));
            end
             %Utilizing the local information which is the total number of request of all users
            %Ave_Q_uf(u,b) = Q_uf(u,b) + l*(numel(ID_B_U)+1)*s_f/(F^gamma)*sqrt(3*log(sum(sum(User_req_matrix)))/(2*C_uf_1(u,b)));
        end
%% STEP 3: OPTIMIZE (CACHE PLACEMENT PHASE IN T)
        Cache_Matrix(u,:) = zeros(1,F);% Reset cache matrix of row u
        ID_req_dif_0 =   find (Temp_user_req_matrix(u,:));
        if ~isempty(ID_req_dif_0)
        Des_Ave_Q_uf = sort(Ave_Q_uf(u,:)/s_f,'descend');% Sorting the file requests of user u in descending rows
        %limit = min([length(ID_req_dif_0),size(Des_Ave_Q_uf,2),S_uVec(S_idx)]);%Due to the limited cache size of each user
            if (length(Des_Ave_Q_uf) > S_uVec(S_idx))
                limit = min([length(Des_Ave_Q_uf),S_uVec(S_idx)]);
            else
                limit = S_uVec(S_idx);
            end
            for i=1:1:limit    
               cache_files = find(Ave_Q_uf(u,:)==Des_Ave_Q_uf(:,i));% Determining which file to cache
               if (1 < length(cache_files))&&( length(cache_files)<= (limit - sum(Cache_Matrix(u,:))))
                   for a=1:1:length(cache_files)
                       if (Cache_Matrix(u,cache_files(:,a))==0)&&(sum(Cache_Matrix(u,:))<limit)
                           Cache_Matrix(u,cache_files(:,a))=1;
                       else
                           break;
                       end
                   end
               elseif (1 < length(cache_files))&&( length(cache_files)> (limit - sum(Cache_Matrix(u,:))))
                    random_node = randperm(length(cache_files)); % random chosen nodes for caching
                   for a=1:1: (limit-sum(Cache_Matrix(u,:)))
                       if (Cache_Matrix(u,cache_files(:,random_node(a)))==0)&&(sum(Cache_Matrix(u,:))<limit)
                       Cache_Matrix(u,cache_files(:,random_node(a)))=1;
                       else
                           break;
                       end
                   end 
               elseif (length(cache_files)==1)
                   Cache_Matrix(u,cache_files)=1;% In case size(cache_files,2) == 1
               end
            end  %for i=1:1:limit
        end %
       % Cumulitive cache matrix
     Hist_Cache_Matrix(u,:) = Hist_Cache_Matrix(u,:) + Cache_Matrix(u,:);  
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %% IMMEDIATETLY CONTENT POPULARITY ALGORITHM 
        % FINDING WHICH FILES NEED TO BE CACHED IN ORDER OF POPULAR FILES
        Cache_UB(u,:) = zeros(1,F);% Reset cache matrix of row u
        id_req_UB = find (Temp_user_req_UB(u,:));% Finding which file f is requested by user u
        %R_Matrix_0(u,id_req )=0;% If k==1, R_matrix unchange ==0
        if ~isempty(id_req_UB)
           Des_req_UB = sort(Temp_user_req_UB(u,:),'descend');% Sorting the file requests of user u in descending rows
           if (length(id_req_UB)>S_uVec(S_idx))
               limit = min(length(id_req_UB),S_uVec(S_idx));%Due to the limited cache size of each user
           else
               limit = S_uVec(S_idx);
           end           
           for i=1:1:limit    
               cache_files_UB = find(Temp_user_req_UB(u,:)==Des_req_UB(:,i));% Determining which file to cache
               if (1 < length(cache_files_UB))&&( length(cache_files_UB)<= (limit - sum(Cache_UB(u,:))))
                   for a=1:1:length(cache_files_UB)
                       if (Cache_UB(u,cache_files_UB(:,a))==0)&&(sum(Cache_UB(u,:))<limit)
                           Cache_UB(u,cache_files_UB(:,a))=1;
                       else
                           break; 
                       end
                   end
               elseif (1 < length(cache_files_UB))&&( length(cache_files_UB)> (limit - sum(Cache_UB(u,:))))
                   random_UB = randperm(length(cache_files_UB)); % random chosen nodes for caching
                   for a=1:1: (limit-sum(Cache_UB(u,:)))
                       if (Cache_UB(u,cache_files_UB(:,random_UB(a)))==0)&&(sum(Cache_UB(u,:))<limit)
                       Cache_UB(u,cache_files_UB(:,random_UB(a)))=1;
                       else
                           break;
                       end
                   end                   
               elseif (length(cache_files_UB)==1)
                   Cache_UB(u,cache_files_UB)=1;% In case size(cache_files,2) == 1
               end
           end   %for i=1:1:limit           
        end
        Hist_Cache_UB(u,:) = Hist_Cache_UB(u,:)+ Cache_UB(u,:);
        %% TOTAL CONTENT POPULARITY ALGORITHM 
        % FINDING WHICH FILES NEED TO BE CACHED IN TOTAL POPULAR FILES IN User_req_matrix
        Cache_CP(u,:) = zeros(1,F);
        id_req_CP = find (User_req_matrix(u,:));% Finding which file f is requested by user u
        %R_Matrix_0(u,id_req )=0;% If k==1, R_matrix unchange ==0
        if ~isempty(id_req_CP)
           Des_req_CP = sort(User_req_matrix(u,:),'descend');% Sorting the file requests of user u in descending rows
           %limit = min(length(id_req_CP),S_uVec(S_idx));%Due to the limited cache size of each user
           if (length(id_req_CP)>S_uVec(S_idx))
               limit = min(length(id_req_CP),S_uVec(S_idx));%Due to the limited cache size of each user
           else
               limit = S_uVec(S_idx);
           end   
           for i=1:1:limit    
               cache_files_CP = find(User_req_matrix(u,:)==Des_req_CP(:,i));% Determining which file to cache
               if (1 < length(cache_files_CP))&&( length(cache_files_CP)<= (limit - sum(Cache_CP(u,:))))
                   for a=1:1:size(cache_files_CP,2)
                       if (Cache_CP(u,cache_files_CP(:,a))==0)&&(sum(Cache_CP(u,:))<limit)
                           Cache_CP(u,cache_files_CP(:,a))=1;
                       else
                           break; 
                       end
                   end
               elseif (1 < length(cache_files_CP))&&( length(cache_files_CP)> (limit - sum(Cache_CP(u,:))))
                   random_CP = randperm(length(cache_files_CP)); % random chosen nodes for caching
                   for a=1:1: (limit-sum(Cache_CP(u,:)))
                       if (Cache_CP(u,cache_files_CP(:,random_CP(a)))==0)&&(sum(Cache_CP(u,:))<limit)
                       Cache_CP(u,cache_files_CP(:,random_CP(a)))=1;
                       else
                           break;
                       end
                   end                   
               elseif (length(cache_files_CP)==1)
                   Cache_CP(u,cache_files_CP)=1;% In case size(cache_files,2) == 1
               end
           end   %for i=1:1:limit           
        end
        Hist_Cache_CP(u,:) = Hist_Cache_CP(u,:)+ Cache_CP(u,:);
end % u=1:1:U    

%
    
%% CALCULTATING THE AVERAGE DOWNLOADING LATENCY AND CACHE BYTE HIT RATE AT EACH TIME SLOT
Cum_Nu_Req = cumsum(Number_Ser_D2D);
Cum_Nu_Req_UB = cumsum(Number_Ser_D2D_UB);
Cum_Nu_Req_CP = cumsum(Number_Ser_D2D_CP);
Cum_Down_Latency = cumsum(Down_Latency);
Cum_Down_Latency_UB = cumsum(Down_Latency_UB);
Cum_Down_Latency_CP = cumsum(Down_Latency_CP);
%Number_Ser_D2D_UB(1,n) = Cum_Nu_Req_UB;
%Number_Ser_D2D_CP(1,n) = Cum_Nu_Req_CP;
%ADL(1,n)  = cumsum(Down_Latency(1,:))/sum(sum(User_req_matrix));
%ADL(2,n)  = cumsum(Down_Latency_UB(1,:))/sum(sum(User_req_matrix));
%ADL(3,n)  = cumsum(Down_Latency_CP(1,:))/sum(sum(User_req_matrix));
CHR(1,n) = Cum_Nu_Req(1,n)/sum(sum(User_req_matrix));
CHR(2,n) = Cum_Nu_Req_UB(1,n)/sum(sum(User_req_matrix));
CHR(3,n) = Cum_Nu_Req_CP(1,n)/sum(sum(User_req_matrix));
end %n=1:1:N

end % for S_u=5:5:Smax
%% CALCULTATING THE CUMULITIVE AVERAGE DOWNLOADING LATENCY AND CACHE BYTE HIT RATE

%N = 10*[1:1:100];
S_u = S_uVec ;
N_u = N_Vec;
%figure
%semilogy(S_u,ADL(1,:),'ro-'); grid on;hold on;
%semilogy(S_u,ADL(2,:),'b*-'); grid on;hold on;
%xlabel('Cache size of UEs')
%ylabel('Average Downloading Latency (s)')
figure
semilogy(N_u,CHR(1,:),'ro-'); grid on;hold on;
semilogy(N_u,CHR(2,:),'b*-'); grid on;hold on;
semilogy(N_u,CHR(3,:),'g^-'); grid on;hold on;
xlabel('Time periods (N)')
ylabel('CHR')
toc
