import pandapower as pp
import numpy as np
import pandas as pd
import pandapower.networks as nw
import matplotlib.pyplot as plt
# import net_modify

net= nw.case14()
# net = net_modify.net
timestamps =1
v_est = pd.read_csv('estimated_v3.csv').drop('Unnamed: 0', axis ='columns')
theta_est = pd.read_csv('estimated_angle3.csv').drop('Unnamed: 0', axis ='columns')
# ===============================================================================
column_a = []
column_vm = []
for n in range(len(net.bus.index)):
    column_a.append('a' + str(n))
    column_vm.append('v' + str(n))
# ===========================================================================================================
bst_idx_list=[]

for t in range(timestamps):
    print(t)
    file_v = 'final_v_opt'+str(t)+'.csv'
    file_theta ='final_theta_opt'+ str(t)+'.csv'
    # v_opt = pd.read_csv(file_v).drop('Unnamed: 0', axis ='columns')
    

    v_o = np.array(v_est.iloc[t,:]).flatten() 
    # v_o_mat = np.reshape(v_o, (len(v_o),1))
    # print(v_o_mat)
    # theta_o = np.array(theta_est.iloc[t,:]).flatten()
    # theta_o_mat = np.reshape(theta_o, (len(theta_o),1))
    optm_v = pd.read_csv(file_v).drop('Unnamed: 0', axis='columns')
    # print(optm_v)
    obj_list = []
    for i in range(len(optm_v)):
        v_temp = np.array(optm_v.iloc[i,0]).flatten()
        # v_temp_mat = np.reshape(v_temp,(len(v_temp),1))
        obj_temp = np.subtract(v_o, v_temp)
        obj =[ abs(x) for x in obj_temp]
        objective = np.linalg.norm(obj,2)
        obj_list.append(objective)
    # print(obj_list)
    
    best_idx = obj_list.index(max(obj_list))
    bst_idx_list.append(best_idx)
    # print(best_idx)

# print(bst_idx_list)

final_optmzd_v = pd.DataFrame(columns=column_vm)
final_optmzd_theta = pd.DataFrame(columns=column_a)

for t in range(timestamps):
    print('n is', n)
    file_v = 'final_v_opt'+str(t)+'.csv'
    file_theta ='final_theta_opt'+ str(t)+'.csv'
    v_opt = pd.read_csv(file_v).drop('Unnamed: 0', axis ='columns')
    theta_opt =pd.read_csv(file_theta).drop('Unnamed: 0', axis ='columns')
    # for n in bst_idx_list:
    final_optmzd_v.loc[t] = np.array(v_opt.iloc[bst_idx_list[t],:]).flatten()
    final_optmzd_theta.loc[t]=np.array(theta_opt.iloc[bst_idx_list[t],:]).flatten()

print(final_optmzd_v)  
print(final_optmzd_theta)  

# final_optmzd_v.to_csv('optmz_v_at_attk_inst.csv')
# final_optmzd_theta.to_csv('optmz_theta_at_attk_inst.csv')


    # optm_theta = pd.read_csv(file_v).drop('Unnamed: 0', axis='columns')


