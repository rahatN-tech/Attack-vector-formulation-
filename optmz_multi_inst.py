from platypus import Problem, Real, NSGAII, Solution, SPEA2, MOEAD, Constraint
import pandapower as pp
import numpy as np
import pandas as pd
import pandapower.networks as nw
import simbench as sb
from numpy.linalg import norm
import pandapower.estimation as est
import bounds
import constraints_limit2
import math




timestamps=4

z_meas_ordered = (pd.read_csv('./z_meas_ordered.csv').drop('Unnamed: 0', axis ='columns'))
v_est = pd.read_csv('./estimated_v3.csv').drop('Unnamed: 0', axis='columns')
theta_est = pd.read_csv('./estimated_angle3.csv').drop('Unnamed: 0', axis='columns')





net = nw.case14()
# net = net_modify.net

pp.runpp(net)

# =====17.2%==================
# attk_list = [9,10,11,12,13]
# total meas= 58

# ====================for 28% intrusion===========================================

attk_list = [9,10,11,12,13,8,5,6]
# attk_list = data.pq_bus

attack_list_col_p = []
attack_list_col_q = []
for x in attk_list:
    attack_list_col_p.append('p'+ str(x))
    attack_list_col_q.append('q'+ str(x))

attack_list_col = attack_list_col_p + attack_list_col_q
# print("attack_list_col_p",attack_list_col_p)
    
# finding indices of the columns attacked------------------------------


column_indices = [z_meas_ordered.columns.get_loc(col) for col in attack_list_col ]
print(column_indices)


gen_bus =[]
load_bus =[]
zero_inj = []

for n in range(len(net.bus)):

   if (net.res_bus.loc[n, 'p_mw'])< 0:
       gen_bus.append(n)

   elif (net.res_bus.loc[n, 'p_mw'])> 0:
       load_bus.append(n)

   else:
      zero_inj.append(n)

# ----------------------------------------------------------------------------------------------------------------------

column_a =[]
column_vm =[]
for n in gen_bus:
    column_a.append('a' + str(n))
    column_vm.append('v' + str(n))
    se_a = pd.DataFrame(columns=column_a)
    se_v = pd.DataFrame(columns=column_vm)

# ----------------------------------------------------------------------------------------------------------------


ybus = net._ppc["internal"]["Ybus"].todense()
# print(ybus)
G = (ybus.real)
B = ybus.imag


col_p =[]
col_v =[]
col_theta =[]
for n in range(len(net.bus)):
    col_p.append(('p'+str(n)))
    col_v.append(('v'+ str(n)))
    col_theta.append(('a'+ str(n)))
col_states= col_v + col_theta

col_q =[]
for n in range(len(net.bus)):
    col_q.append(('q'+str(n)))



col_load_p =[]
for n in load_bus:
   col_load_p.append(('p'+str(n)))
   
# print(col_load_p)
col_load_q =[]
for n in load_bus:
   col_load_q.append(('q'+str(n)))


col = col_p + col_q
col_load = col_load_p+col_load_q


z_meas = np.array(z_meas_ordered).flatten()
# print(z_meas)
z_meas_mat = np.reshape(z_meas, (len(z_meas), 1))

# 
var_bounds=zip(bounds.lwr_bnd, bounds.upr_bnd)
variable_range = (tuple(var_bounds))
# print("length of vriable_range",len(variable_range))




#
def generate_meas(x):
    v= x[0:14]
    theta =x[14:28]

    z_arr =[]
    z_p = []

    for i in range(len(net.bus)):
            p_inj = 0
            for j in range(len(net.bus)):
                p_bus = v[i]*v[j]*(G[i,j]*math.cos(math.radians(theta[i]-theta[j]))+ B[i,j]*math.sin(math.radians(theta[i] - theta[j])))
                p_inj = p_inj + p_bus

            z_arr.append(p_inj)
            z_p.append(p_inj)
    # ------------------------------------------------------------------------------------------------------------------------------------------
    z_q =[]
    for i in range(len(net.bus)):
        q_inj =0

        for j in range(len(net.bus)):
            q_bus = v[i]*v[j]*(G[i,j]*math.sin(math.radians(theta[i]-theta[j]))- B[i,j]*math.cos(math.radians(theta[i] - theta[j])))
            q_inj = q_inj + q_bus

        z_arr.append(q_inj)
        z_q.append(q_inj)
   
    return z_arr, z_p, z_q







z_meas_one =(pd.read_csv('./z_meas_ordered.csv').drop('Unnamed: 0', axis ='columns'))

num_var = 28


def objectives(vars):
    
        
    # z_meas_p = z_meas_one[col_p]
    z_meas_p = z_meas_one[attack_list_col_p]
    # z_meas_q = z_meas_one[col_q]
    z_meas_q = z_meas_one[attack_list_col_q]
    z_meas_pq = pd.concat([z_meas_p,z_meas_q], axis= 'columns')
    z_meas_pq_load_arr = np.array(z_meas_pq.iloc[t, :])
    z_meas_pq_load_mat = (-1)*np.reshape(z_meas_pq_load_arr, (len(z_meas_pq_load_arr), 1))
    z_hat, zp_hat, zq_hat = generate_meas(vars)
    
    zp_sgen_hat = list(np.take(zp_hat,attk_list))
    zq_sgen_hat = list(np.take(zq_hat, attk_list))

    z_attk_hat = zp_sgen_hat +zq_sgen_hat
    # print(z_attk_hat)
    z_hat_mat = (np.reshape(z_attk_hat, (len(z_attk_hat),1)))
    # gen==> +ive, load ==> -ive

# -------------------------------------------------------------------------------------------------------------------------


# --------------------------------second term------------------------------------------------------------------------------------------------------------
    
    
    x_v =(np.array(v_est.iloc[t,:]))
    x_theta = (np.array(theta_est.iloc[t,:]))



    x_v_mat = np.reshape(x_v , (len(x_v),1))
    x_theta_mat = (np.reshape(x_theta, (len(x_theta),1)))

    # v_hat, theta_hat = vars
    v_hat= vars[0:14]
    theta_hat =vars[14:28]
    v_hat_mat =np.reshape(v_hat, (len(v_hat), 1))
    theta_hat_mat =np.reshape(theta_hat, (len(theta_hat), 1))

    error_x_v = list((v_hat_mat - x_v_mat).flatten())
    error_v = [100*(x) for x in error_x_v]
    # error_x_v_arr =1000* np.array(v_hat_mat-x_v_mat ).flatten()
    # x_v_norm = 1000*np.linalg.norm(error_x_v,2)
    error_x_theta= list((theta_hat_mat-x_theta_mat).flatten())
    error_theta = [1*(x) for x in error_x_theta]
    
    
# ----------------------------------------------------------------------------------------------------------------------------------------
    error_z = 10*(z_meas_pq_load_mat - z_hat_mat)
    norm_z =np.linalg.norm(error_z,2)

    # =========================================================================================================================================
    # --------------------obj func 1---------------------------------------------------------------------------------------------------------
    f1 = norm_z

    # --------------------------------------obj func 2---------------------------------------------------------------------------------------------
    # f2 = (-1)* norm_x
    f2 = list(error_v) + list(error_theta)
    # f2_obj =[(-1)*x for x in f2]


# ==========================================constraints1========================================================
    upr_limit = np.array(constraints_limit2.upr_bnd)
   

    z_arr, zp_pu, zq_pu = generate_meas(vars)
    zp = [-100*x for x in zp_pu]
    zq = [-100*x for x in zq_pu]
    zp_arr = np.take(zp,attk_list)
    zq_arr = np.take(zq, attk_list)
    zpq_arr = np.array(list(zp_arr) + list(zq_arr))
    violation = np.maximum(0,( zpq_arr- upr_limit ))
    # return tuple(violation)
    constr1 = list(upr_limit - zpq_arr)
    # print("constr1",violation)
# ===============================================

# ==========================================constraints=2=======================================================
    lwr_limit = np.array(constraints_limit2.lwr_bnd)
   

   
    constr2 =  list(zpq_arr - lwr_limit)
# ===============================================
    constr = constr1 + constr2
    # return [f1, f2]
    # return  f2 , constr1, constr2
    return f2, constr

# =================================================constraint1============================================================================



for t in range(timestamps):
    
    problem = Problem(len(variable_range), 28, 20)  # 1 objectives

    problem.constraints[:] = [Constraint(">=0") for _ in range(20)]
    problem.types[:] = [Real(variable_range[i][0], variable_range[i][1]) for i in range(num_var)]  # Decision variables in specified ranges
    # print(len(problem.types))
    problem.directions[:] = Problem.MAXIMIZE
    #
    problem.function = objectives  # Set the objectives function

    algorithm = NSGAII(problem)
    #

    for i in range(10000):  # Perform 10000 iterations
        algorithm.step()
        num_iterations = algorithm.nfe
        print("Iteration:", i, "| Number of iterations:", num_iterations)

    soln_var = pd.DataFrame(columns = col_states)
    i=0
    for solution in algorithm.result:
        
        # print("soln ", solution.variables,"| Objective 1:", solution.objectives[0], "| Objective 2:", solution.objectives[1])
        soln_var.loc[i] = solution.variables
        i+=1
        # print("soln", solution.variables)-----------------
        constr1_values = solution.constraints[0:10]
        constr2_values = solution.constraints[10:]

        # Check if all constraints are greater than or equal to zero
        if all(value >= 0 for value in constr1_values) and all(value >= 0 for value in constr2_values):
            print("Constraints satisfied for this solution.")
        else:
            print("Constraints violated for this solution.")

    # soln_var.to_csv('./soln_variable_with_single_obj_try.csv')
    file_name = 'soln_variable_8bus'+str(t)
    soln_var.to_csv('./'+file_name + ".csv")
    print(soln_var.head())
