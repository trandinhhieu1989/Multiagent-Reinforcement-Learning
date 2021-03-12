for u=1:1:U
    for f=1:1:F
        latency_BS_UE = s_f/R_BS_UE(u,:);
       % if  ((Temp_user_req_matrix(u,f)~=0)&& (n==1))||((Temp_user_req_matrix(u,f)==0)&& (n==1))
        if (n==1)
 % At the beginning phase, we do not have any cached files. Therefore, we need to transmit from BS
        R_Matrix(u,f) = 0;
 % Cache file having high temp_user_request and satisfy the capacity  
        Des_Temp_user_req_matrix = ;% 
        elseif (n>1)&& (Temp_user_req_matrix(u,f)~=0)
 % Finding the neighbor of user u in N_U matrix
        Nei_u =  find(N_U(u,:)~=0);%Finding Neighbors of user u          
            if ~isempty(Nei_u)% isempty: If Nei_u_f empty, it return 1, otherwise 0
% Finding the neighbors contain file f or not, if yes we select the neighbor having lowest latency   
                Nei_u_f = [];
                for i=1:1:size(Nei_u,2)  
                    Nei_u_i_f = find (Cache_Matrix(Nei_u(:,i),f)~=0);% Finding which neighbors contain file f 
                    Nei_u_f = [Nei_u_f,Nei_u_i_f];
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
                      R_Matrix(u,f) = (1-r)*R_u_f; %Eq. (2), user u pay incentive r for chosen_nei
                      R_Matrix(chosen_nei,f)= r*R_u_f; %Eq. (2), the chosen neighbor can receive incentive r   
                else %~isempty(Nei_u_f)
                      latency = latency_BS_UE;
                      R_Matrix(u,f) = 0;
                end %~isempty(Nei_u_f)
               
            else %~isempty(Nei_u)
                R_Matrix(u,f) = 0;
            end %~isempty(Nei_u)
        else %Temp_user_req_matrix(u,f)=0 && (n>1))
            R_Matrix(u,f) = 0;
        end %(n==1)
    end
end