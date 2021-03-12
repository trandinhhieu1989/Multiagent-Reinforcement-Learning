%This code is written by Tran Dinh Hieu on October 1, 2018.
%I tried to coded the paper: Efficient D2D content caching using multi-agent reinforcemnt learning, INFOCOM 2018.
% THIS IS A SAMPLES CODE: WE ONLY CONSIDER A FEW NUMBER OF USERS 
% WE TRY TO PRODUCE AGAIN THE FIGURE 2,3 . AVERAGE DOWNLOADING LATENCY AND CHR VERSUS CACHE SIZE OF UEs
%% IN THIS CODE, WE COMPARED THE BMCU WITH UPER BOURND ALGORITHM, I.E., CACHE HIGH CONTENT POPULARITY FILES 
%% AND THE HISTORICAL MATRIX OF THE UPER BOURND ALGORITHM EQUAL TO BMCU'S HISTORICAL MATRIX
%% THIS CODE USE INITIALIZE STEP TO CREATE THE HIST_CACHE_MATRIX, R_MATRIX_0, R_MATRIX_1,USER_REQ_MATRIX,
%% Q_UF_B_UF_0, Q_UF_B_UF_1
clear; clc; close all;
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

U =  3; % Total number of users in Macrocell
F = 20;   % Total number of files

s_f = 1; % Size of files
%S_u = 2; % Cache size of each user
kappa = 1e-2;% Pathloss constant
epsilon = 4; % Pathloss exponent
gamma = 0.56; %Shape factor of Zipf distribution
mu   = F; % Avarage arrival rate of user's request 
r = 0.5; % Proportional factor of payment
T = F+1;% Time period
%N = 4;% Number of time periods

%The allocated bandwidth to each link given by total bandwidth x spatial reuse/number of users
B_BS_UE = B*SPRU_BS_UE/U;
B_D2D = B*SPRU_D2D/U;

% Transfer the noise power from dBm/Hz to dBm
P_dBm_BS_UE = Sigma + 10*log10(B_BS_UE);
P_dBm_D2D = Sigma + 10*log10(B_D2D);

% Transfer the noise power from dBm to Watt
P_W_BS_UE = 10^(P_dBm_BS_UE/10)*10^-3;
P_W_D2D = 10^(P_dBm_D2D/10)*10^-3;
M = 80; % Number of samples creating each running times
N = 100;%N: total number of time period T 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 1: 3. INITIALIZE BY APPLYING MOST POPULAR FILES ALGORITHM, EACH USER CACHED SOME FILES (NOT ALL FILES) AT LEAST ONCE.
%%INITIALIZING THE COUNT MATRIX, Q-VALUES MATRIX, HISTORICAL CACHED FILES
K =1; % Number of running trial times to create the initial data 
%% Creating the user's coordinate matrix
u_coor_matr = Fixed_Topology(Radius,C,U);
%u_coor_matr = Topology(Radius,C,U);
    max_S_u = 20;
    Down_Latency = zeros(1,max_S_u/5); % Initialize the downloading latency in Fig. 2
    Down_Latency_UB = zeros(1,max_S_u/5);
    Number_Ser_D2D = zeros(1,max_S_u/5); % Initialize the number of file served by D2D or self-offloading
    Number_Ser_D2D_UB = zeros(1,max_S_u/5);
   
for S_u=5:5:max_S_u
    %u_coor_matr = Topology(Radius,C,U);
    User_req_matrix = zeros(U,F);%Creating the user request matrix: rows users, columns: files, elements are the number of time user i request file fn
    Hist_Cache_Matrix = zeros(U,F); % Cumulitive cache matrix of all the users 
    Q_uf_b_uf_0 = zeros (U,F);
    Q_uf_b_uf_1 = zeros (U,F);
    Q_uf_1      = zeros (U,F);
    Q_uf_0      = zeros (U,F);
for k=1:1:K %K: total number of trials to create initialize phase 
Cache_Matrix = zeros(U,F); % Defining the cache matrix of all the users, each user limit by cache size
R_Matrix_0 = zeros(U,F);
R_Matrix_1 = zeros(U,F);
%% Calculating the transmission rate between BS-UE and D2D
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
%%
%%Finding the neighbors of user u1,u2,...,U, i.e., N'(u1): neighbor of u1
N_U= zeros(U,U);% rows: users, columns: neighbors, while the diagonal line always equal to 0: each user always have transmission rate higher than transmission from BS.
for u=1:1:U
    for v=1:1:U
    if (R_D2D(u,v) > R_BS_UE(u,:))&& (v~=u)%comparing between transmission rate 
        N_U(u,v) = 1;
    end
    end
end
%% 
%
%% Creating the demand matrixs only for subset of user u1 and its neighbors IB_B_U
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

%%   Creating the user request matrix: rows users, columns: files, elements are the number of time user i request file fn
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
%Cache_Matrix = zeros(U,F); % Defining the cache matrix of all the users
% Cache files following the MOST POPULAR FILES algorithm
%%
for u=1:1:U
    for f=1:1:F
      % If k=1, transmission from BS
      if k==1
        id_req = find (Temp_user_req_matrix(u,:));% Finding which file f is requested by user u
        %R_Matrix_0(u,id_req )=0;% If k==1, R_matrix unchange ==0
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
                           %Hist_Cache_Matrix(u,cache_files(:,a)) = Hist_Cache_Matrix(u,cache_files(:,a)) + Cache_Matrix(u,cache_files(:,a));
                       else
                           break; 
                       end
                   end
               else
                   Cache_Matrix(u,cache_files)=1;% In case size(cache_files,2) == 1
                   %Hist_Cache_Matrix(u,cache_files) = Hist_Cache_Matrix(u,cache_files) + Cache_Matrix(u,cache_files);
               end
           end   %for i=1:1:limit           
        end
      end % k==1
      %
      latency_BS_UE = s_f/R_BS_UE(u,:); 
      if (k>1)&& (Temp_user_req_matrix(u,f)~=0)  
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         %% Cache files follow CONTENT POPULARITY OF EACH REQUEST  
         id_req = find (Temp_user_req_matrix(u,:));% Finding which file f is requested by user u
         %R_Matrix_0(u,id_req )=0;% If k==1, R_matrix unchange ==0
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
                           %Hist_Cache_Matrix(u,cache_files(:,a)) = Hist_Cache_Matrix(u,cache_files(:,a)) + Cache_Matrix(u,cache_files(:,a));
                       else
                           break; 
                       end
                   end
               else
                   Cache_Matrix(u,cache_files)=1;% In case size(cache_files,2) == 1
                   %Hist_Cache_Matrix(u,cache_files) = Hist_Cache_Matrix(u,cache_files) + Cache_Matrix(u,cache_files);
               end
           end   %for i=1:1:limit           
         end
         %
         %% Finding the R_matrix
         for a=0:1:1 
            %%
            if a==0
            % Finding the neighbor of user u in N_U matrix
            Nei_u =  find(N_U(u,:)~=0);%Finding Neighbors ID of user u          
            if ~isempty(Nei_u)% isempty: If Nei_u_f empty, it return 1, otherwise 0
            % Finding the neighbors contain file f or not, if yes we select the neighbor having lowest latency   
                Nei_u_f = [];
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
            elseif a==1
               R_Matrix_1(u,f) = latency_BS_UE;
            end % end a==0
         end % for a=0:1:1
        %else %Temp_user_req_matrix(u,f)=0 && (n>1))
            %R_Matrix(u,f) = R_Matrix(u,f);
      end %(k>1)&& (Temp_user_req_matrix(u,f)~=0) 
    % Cumulitive cache matrix
     Hist_Cache_Matrix(u,f) = Hist_Cache_Matrix(u,f) + Cache_Matrix(u,f);
    
    end % for f=1:1:F
end % for u=1:1:U
    %% Caculate the Q value as in multi-arm bandit: Q_n+1 = Q_n + (R_n - Q_n)/n
    % When we have complete Hist_Cache_Matrix and R_Matrix, we will
    % calculate the Q value
     if (k>1)%&&(u == U)&&(f==F)
        for a=1:1:U
            for b=1:1:F
        ID_B_U = find(N_U(a,:)==1);% Subset of user u1 and its neighbors
        N_U_1 = 0;
        Requ_times = 0;
%Finding the total number of times user u cache file f so far (in the past)
        N_U_1 = N_U_1 + Hist_Cache_Matrix(a,b);
%Finding the total number of times user u did not cache file f so far (in the past)        
        Requ_times = Requ_times + User_req_matrix(a,b); 
        if ~isempty(ID_B_U)
            for i=1:1:size(ID_B_U,2)
 %Finding the total number of times user u and its neighbor N_U cache file f so far (in the past)           
                N_U_1 = N_U_1 + Hist_Cache_Matrix(ID_B_U(i),b); 
 %Finding the total number of times user u and its neighbor N_U request file f so far (in the past)
                Requ_times = Requ_times + User_req_matrix(ID_B_U(i),b);           
            end
 %Finding the total number of times user u and its neighbor N_U did not cache file f so far (in the past)
                N_U_0 = Requ_times - N_U_1;
 %% Calculating as 8. in Algorithm 1    
        %if (k>1)%&&(N_U_0~=0)%&&(R_Matrix_0(u,f)~=0)
        Q_uf_b_uf_0(a,b)=  Q_uf_b_uf_0(a,b) + (R_Matrix_0(a,b) - Q_uf_b_uf_0(a,b))/(N_U_0+1);            
        %end
        %if (k>=2)%&&(N_U_1~=0)%&&(R_Matrix_1(u,f)~=0)
        Q_uf_b_uf_1(a,b)= Q_uf_b_uf_1(a,b) + (R_Matrix_1(a,b) - Q_uf_b_uf_1(a,b))/(N_U_1+1);              
        %end 
        end %~isempty(ID_B_U)
            end %for b=1:1:F
        end %for a=1:1:U
     end %if (k>1)&&(u == U)&&(f==F)
end %for k=1:1:K

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 2: OBSERVE (USER REQUEST PHASE IN PERIOD T)

for n=1:1:N
%% Finding the Temp_user_req_matrix and User_req_matrix
%Cache_Matrix = zeros(U,F); % We need Cache_Matrix value to find the
%nearest neighbor which contains file f
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
%M = 40; % Number of samples creating each running times
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
    Cache_UB = Cache_Matrix;
    Hist_Cache_UB = Hist_Cache_Matrix;
    Temp_user_req_UB = Temp_user_req_matrix;
    User_req_UB = User_req_matrix;
%% AVERAGE DOWNLOADING LATENCY: SUM DOWNLOADING LATENCY FOR ALL CONTENT REQUESTS TO THE TOTAL NUMBER OF CONTENT REQUESTS
%% CACHE HIT RATE: THE TOTAL NUMBER OF REQUESTS SERVED BY SELF-OFFLOADING OR D2D TRANMISSION
for u=1:1:U
    for f=1:1:F
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONTENT POPULAR AT EACH REQUEST ALGORITHM
 % FINDING WHICH FILES NEED TO BE CACHED IN MOST POPULAR FILESS
    id_req_UB = find (Temp_user_req_UB(u,:));% Finding which file f is requested by user u
        %R_Matrix_0(u,id_req )=0;% If k==1, R_matrix unchange ==0
        if ~isempty(id_req_UB)
           Des_req_UB = sort(Temp_user_req_UB(u,:),'descend');% Sorting the file requests of user u in descending rows
           limit = min(size(id_req_UB,2),S_u);%Due to the limited cache size of each user
           for i=1:1:limit    
               cache_files_UB = find(Temp_user_req_UB(u,:)==Des_req_UB(:,i));% Determining which file to cache
               if (size(cache_files_UB,2) > 1)
                   for a=1:1:size(cache_files_UB,2)
                       if (Cache_UB(u,cache_files_UB(:,a))==0)&&(sum(Cache_UB(u,:))<limit)
                           Cache_UB(u,cache_files_UB(:,a))=1;
                           %Hist_Cache_Matrix(u,cache_files(:,a)) = Hist_Cache_Matrix(u,cache_files(:,a)) + Cache_Matrix(u,cache_files(:,a));
                       else
                           break; 
                       end
                   end
               else
                   Cache_UB(u,cache_files_UB)=1;% In case size(cache_files,2) == 1
                   %Hist_Cache_Matrix(u,cache_files) = Hist_Cache_Matrix(u,cache_files) + Cache_Matrix(u,cache_files);
               end
           end   %for i=1:1:limit           
        end
 % CALCULATING ADL AND CHR IN UB
    latency_BS_UE = s_f/R_BS_UE(u,:);
    id_req_UB = find (Temp_user_req_UB(u,f)~=0);% Finding which file f is requested by user u
      if ~isempty(id_req_UB)
          if Cache_UB(u,f)==1 % If user u cache file f
             Number_Ser_D2D_UB(1,S_u/5)  = Number_Ser_D2D_UB(1,S_u/5)  + 1; % Increasing the number of self-offloading or D2D transmission
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
                    Down_Latency_UB(1,S_u/5)  = Down_Latency_UB(1,S_u/5)  + latency_UB;
                    Number_Ser_D2D_UB(1,S_u/5)  = Number_Ser_D2D_UB(1,S_u/5)  + 1;
                else % No any neighbors of u cache file f, so transmission from BS
                    Down_Latency_UB(1,S_u/5)  = Down_Latency_UB(1,S_u/5)  + latency_BS_UE;                    
                end %if ~isempty(Nei_u_f_UB)
             else % transmission from BS
                 Down_Latency_UB(1,S_u/5)  = Down_Latency_UB(1,S_u/5)  + latency_BS_UE;
             end %if ~isempty(Nei_u_UB)
          end  %if Cache__UB(u,f)==1           
      end %if ~isempty(id_req_UB)
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PROPOSED BELIEF-BASED MODIFIED COMBINATORIAL UPER CONFIDENCE BOUND 
      %latency_BS_UE = s_f/R_BS_UE(u,:);
      id_req = find (Temp_user_req_matrix(u,f)~=0);% Finding which file f is requested by user u
      if ~isempty(id_req)
          if Cache_Matrix(u,f)==1 % If user u cache file f
             Number_Ser_D2D(1,S_u/5)  = Number_Ser_D2D(1,S_u/5)  + 1; % Increasing the number of self-offloading or D2D transmission
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
                    Down_Latency(1,S_u/5)  = Down_Latency(1,S_u/5)  + latency;
                    Number_Ser_D2D(1,S_u/5)  = Number_Ser_D2D(1,S_u/5)  + 1;
                else % No any neighbors of u cache file f, so transmission from BS
                    Down_Latency(1,S_u/5)  = Down_Latency(1,S_u/5)  + latency_BS_UE;                    
                end %if ~isempty(Nei_u_f)
             else % transmission from BS
                 Down_Latency(1,S_u/5)  = Down_Latency(1,S_u/5)  + latency_BS_UE;
             end %if ~isempty(Nei_u)
          end  %if Cache_Matrix(u,f)==1           
      end %if ~isempty(id_req)
    end %for f=1:1:F
    Hist_Cache_UB(u,:) = Hist_Cache_UB(u,:)+ Cache_UB(u,:);
end %for u=1:1:U
%
%% FINDING WHICH FILES NEED TO BE CACHED
for u=1:1:U
    for f=1:1:F
      %% Finding the R_matrix, R_Matrix_0, R_Matrix_1
         for a=0:1:1 
            if a==0
            % Finding the neighbor of user u in N_U matrix
            Nei_u =  find(N_U(u,:)~=0);%Finding Neighbors ID of user u          
            if ~isempty(Nei_u) % isempty: If Nei_u_f empty, it return 1, otherwise 0
            % Finding the neighbors contain file f or not, if yes we select the neighbor having lowest latency   
                Nei_u_f = [];
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
            elseif a==1
               R_Matrix_1(u,f) = latency_BS_UE;
            end % end a==0
         end % for a=0:1:1  
       %% Caculate the Q value as in multi-arm bandit: Q_n+1 = Q_n + (R_n - Q_n)/n 
        ID_B_U = find(N_U(u,:)==1);% Subset of user u1 and its neighbors
        N_U_1 = 0;
        Requ_times = 0;
%Finding the total number of times user u cache file f so far (in the past)
        N_U_1 = N_U_1 + Hist_Cache_Matrix(u,f);
%Finding the total number of times user u did not cache file f so far (in the past)        
        Requ_times = Requ_times + User_req_matrix(u,f); 
        if ~isempty(ID_B_U)
            for i=1:1:size(ID_B_U,2)
 %Finding the total number of times user u and its neighbor N_U cache file f so far (in the past)           
                N_U_1 = N_U_1 + Hist_Cache_Matrix(ID_B_U(i),f); 
 %Finding the total number of times user u and its neighbor N_U request file f so far (in the past)
                Requ_times = Requ_times + User_req_matrix(ID_B_U(i),f);           
            end
 %Finding the total number of times user u and its neighbor N_U did not cache file f so far (in the past)
                N_U_0 = Requ_times - N_U_1;
 %% 8. Calculating as in Algorithm 1    
        Q_uf_b_uf_0(u,f)=  Q_uf_b_uf_0(u,f) + (R_Matrix_0(u,f) - Q_uf_b_uf_0(u,f))/(N_U_0+1);            
        Q_uf_b_uf_1(u,f)= Q_uf_b_uf_1(u,f) + (R_Matrix_1(u,f) - Q_uf_b_uf_1(u,f))/(N_U_1+1);     
 %% 10. We only compute the Probability of nearest neighbor cache file f
        %Pr_chosen_nei_1 = Hist_Cache_Matrix(chosen_nei,f)/sum(User_req_matrix(chosen_nei,:));
         if ~isempty(chosen_nei)
            Pr_chosen_nei_1 = Hist_Cache_Matrix(chosen_nei,f)/sum(User_req_matrix(chosen_nei,:));
        else % if we the chosen_nei is empty
            Pr_chosen_nei_1 = 0;
        end
 %% 11.       
        Q_uf_1(u,f) = Q_uf_b_uf_1(u,f);
        Q_uf_0(u,f) = Q_uf_b_uf_0(u,f)*Pr_chosen_nei_1;
        end %~isempty(ID_B_U)
%% 12
        % Benifits
        Q_uf(u,f) = Q_uf_1(u,f)-Q_uf_0(u,f);
    end % f=1:1:F
        % Calculate the average benifits using belief-based modified CUCB algorithm
        l = max(Q_uf(u,:)/s_f); % Because we only use one s_f value
        %for b=1:1:F
            %if Hist_Cache_Matrix(u,b)==0
                %Hist_Cache_Matrix(u,b)= 1;
            %end
            %Ave_Q_uf(u,b) = Q_uf(u,b) + l*(numel(ID_B_U)+1)*s_f/(F^gamma)*sqrt(3*log(sum(User_req_matrix(u,:)))/(2*Hist_Cache_Matrix(u,b)));
        %end
        C_uf_1 = Hist_Cache_Matrix; % Instead of increase value of Hist_Cache_Matrix from 0 to 1, we change value of C_uf_1 matrix
        for b=1:1:F
            if C_uf_1(u,b)==0
                C_uf_1(u,b)= 1;
            end
            Ave_Q_uf(u,b) = Q_uf(u,b) + l*(numel(ID_B_U)+1)*s_f/(F^gamma)*sqrt(3*log(sum(User_req_matrix(u,:)))/(2*C_uf_1(u,b)));
        end
%% STEP 3: OPTIMIZE (CACHE PLACEMENT PHASE IN T)
        Cache_Matrix(u,:) = zeros(1,F);% Reset cache matrix of row u
        ID_req_dif_0 =   find (Temp_user_req_matrix(u,:));
        Des_Ave_Q_uf = sort(Ave_Q_uf(u,:)/s_f,'descend');% Sorting the file requests of user u in descending rows
        limit = min([numel(ID_req_dif_0),size(Des_Ave_Q_uf,2),S_u]);%Due to the limited cache size of each user
              for i=1:1:limit    
               cache_files = find(Ave_Q_uf(u,:)==Des_Ave_Q_uf(:,i));% Determining which file to cache
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
         end  %for i=1:1:limit
       % Cumulitive cache matrix
     Hist_Cache_Matrix(u,:) = Hist_Cache_Matrix(u,:) + Cache_Matrix(u,:);  
end % u=1:1:U    

end %n=1:1:N
%% CALCULTATING THE AVERAGE DOWNLOADING LATENCY AND CACHE BYTE HIT RATE
ADL(1,S_u/5)  = Down_Latency(1,S_u/5)/sum(sum(User_req_matrix));
ADL(2,S_u/5)  = Down_Latency_UB(1,S_u/5)/sum(sum(User_req_matrix));
CHR(1,S_u/5) = Number_Ser_D2D(1,S_u/5)/sum(sum(User_req_matrix));
CHR(2,S_u/5) = Number_Ser_D2D_UB(1,S_u/5)/sum(sum(User_req_matrix));

end % for S_u=1:1:3
S_u = [5:5:max_S_u];
figure
semilogy(S_u,ADL(1,:),'ro-'); grid on;hold on;
semilogy(S_u,ADL(2,:),'b*-'); grid on;hold on;
xlabel('Cache size of UEs')
ylabel('Average Downloading Latency (s)')
figure
semilogy(S_u,CHR(1,:),'ro-'); grid on;hold on;
semilogy(S_u,CHR(2,:),'b*-'); grid on;hold on;
xlabel('Cache size of UEs')
ylabel('CHR')
