[%% Based on  Temp_user_req_matrix, we calculate the reward user u receives for file f as in Eq. (2)
%In this paper, instead of considering all the users which is increasing
%the super action, we only consider user u and its neighbor to decrease the action space.
ID_B_U = find(N_U(1,:)==1);% Subset of user u1 and its neighbors
R_Matrix = zeros(U,F); %Reward matrix
for u=1:1:size(ID_B_U,2)
   for f=1:1:F
   latency_BS_UE = s_f/R_BS_UE(u,:); 
 % At the beginning phase, we do not have any cached files. Therefore, we need to transmit from BS
 % Cache file having high temp_user_request and satisfy the capacity 
     if (n==1)
        id_req = find (Temp_user_req_matrix(u,:)~=0,2);% Finding which file f is requested by user u
        R_Matrix(u,id_req )=0;% Reward matrix at the first time slot T
        if ~isempty(id_req)
           Des_req = sort(Temp_user_req_matrix(u,:),'descend');% Sorting the file requests of user u in the descending rows
           limit = min(size(id_req,2),S_u);% Due to the limited cache size of each user
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
 
    elseif (n>1)&& (Temp_user_req_matrix(u,f)~=0)
% Finding the neighbor of user u in N_U matrix
            Nei_u =  find(N_U(u,:)~=0);%Finding Neighbors ID of user u          
            if ~isempty(Nei_u)% isempty: If Nei_u_f empty, it return 1, otherwise 0
% Finding the neighbors contain file f or not, if yes we select the neighbor having lowest latency   
                Nei_u_f = [];
                for i=1:1:size(Nei_u,2)  
                    Nei_u_i_f = find (Cache_Matrix(Nei_u(:,i),f)~=0);% Finding which neighbors contain file f
                    % Add user ID which contains f file in Nei_u_f
                    if ~isempty(Nei_u_i_f)
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
                        R_Matrix(u,f) = R_Matrix(u,f)+(1-r)*R_u_f; %Eq. (2), user u pay incentive r for chosen_nei
                        R_Matrix(chosen_nei,f)= R_Matrix(chosen_nei,f)+r*R_u_f; %Eq. (2), the chosen neighbor can receive incentive r   
                %else %~isempty(Nei_u_f)
                      else%u== chosen_nei: We needn't to pay r for neighbor node
                        R_Matrix(u,f) = R_Matrix(u,f)+ R_u_f;%
                      end
                      %latency = latency_BS_UE;
                     %R_Matrix(u,f) = R_Matrix(u,f);
                end %~isempty(Nei_u_f)
               
            %else %~isempty(Nei_u)
                %R_Matrix(u,f) = R_Matrix(u,f);
            end %~isempty(Nei_u)
      end %(n==1)
   end % for f=1:1:F
end % for u=1:1:U
%% 8: Update Q_{u,f}(b_{u,f}):avarage reward of joint action b_{u,f} of user u and its neighbors (i.e., u2 and u3) with each file f
 % First, we need to clarify C_{u,f}(b_{u,f}) based on Temp_user_req_matrix   
 % Second, we can calculate Q_(b_{u,f}) as step 3 in belief-based modified CUCB based on R_Matrix
 % Third, updating the Q_{u,f}(b_{u,f}) as in Algorithm 1
        C_uf_b_uf = zeros (1,F);
        Q_b_uf = zeros (1,F);
        Q_uf_b_uf = zeros (1,F);
        for f=1:1:F
            Temp_C_uf_b_uf = sum(Temp_user_req_matrix(:,f));
            Temp_Q_b_uf = sum(R_Matrix(:,f));
            C_uf_b_uf(1,f)= C_uf_b_uf(1,f) + Temp_C_uf_b_uf;
            Q_b_uf(1,f)= (Q_b_uf(1,f) + Temp_Q_b_uf)/size(ID_B_U,2);
            Q_uf_b_uf(1,f) = Q_b_uf(1,f) + 1/(C_uf_b_uf(1,f)+1)*(R_Matrix(1,f)-Q_b_uf(1,f));
        end
  %% 9. Update C_{u,f}(a_{u,f}) and C__{u,v,f}(a_{u,f}).       
        C_uf_a_uf = Temp_user_req_matrix(1,:);% Number of times actions a_{u,f} is selected by agent u
        C_uvf_a_uf = C_uf_b_uf - C_uf_a_uf;% Number of times action a_{u,f} is selected by neighbors of u, i.e., agent v
 %% 10 Compute Pr_{u,f}(b_{u,f}^(-u))
        Pr_uvf_a_vf = C_uvf_a_uf/T;
        Pr_uf_b_uf = prod(Pr_uvf_a_vf,2); % prof function:  the product of the elements in each row.      
