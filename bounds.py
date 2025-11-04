import pandapower as pp
import numpy as np
import pandas as pd
import pandapower.networks as nw
from numpy.linalg import norm
import pandapower.estimation as est
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
import simbench as sb
# 0.1
tau_freq = 0.89
lambda_x = 1
lambda_z =1
epochs =1
timestamps =1

# t=0
# 0.01
tau_loss = -7500

z_meas_ordered =(pd.read_csv('./z_meas_ordered.csv').drop('Unnamed: 0', axis ='columns'))
v_est = pd.read_csv('./estimated_v3.csv').drop('Unnamed: 0', axis='columns')
theta_est = pd.read_csv('./estimated_angle3.csv').drop('Unnamed: 0', axis='columns')

net= nw.case14()
# print(net.load)
pp.runpp(net)

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



# ---------------------load_bus_columns-----------------------------------------
col_load_v=[]
col_load_a=[ ]
for n in load_bus:
    col_load_v.append('v' + str(n))
    col_load_a.append('a'+str(n))

# ----------------------------------------------------------------------------------
# -----------------------------defining column names--------------------------------------------------------------
column_a_gen =[]
column_vm_gen =[]
for n in gen_bus:
    column_a_gen.append('a' + str(n))
    column_vm_gen.append('v' + str(n))
    se_a_gen = pd.DataFrame(columns=column_a_gen)
    se_v_gen = pd.DataFrame(columns=column_vm_gen)
# print(se_a_gen)


column_a_zer_inj =[]
column_vm_zer_inj =[]
for n in zero_inj:
    column_a_zer_inj.append('a' + str(n))
    column_vm_zer_inj.append('v' + str(n))
    se_a = pd.DataFrame(columns=column_a_zer_inj)
    se_v = pd.DataFrame(columns=column_vm_zer_inj)
# print(column_vm_zer_inj)



# -------------------------------------------------------------------------------

gen_bus_values = v_est.loc[0,column_vm_gen].values.tolist()
a = [x - 0.001 for x in gen_bus_values]

# --------------------------defining_upper bound--------------------------------------



v_est[v_est.columns] = np.repeat(1.5,14)

v_est_upr_bnd= np.array(v_est.iloc[0,:])
# print(v_est_upr_bnd)
theta_est[theta_est.columns] = np.concatenate(([0.001],np.repeat(0, 13)),axis =0)
theta_est_upr_bnd= np.array(theta_est.iloc[0,:])

# -----------------------defining lower bound-------------------------------------------------

v_est = pd.read_csv('./estimated_v3.csv').drop('Unnamed: 0', axis='columns')
theta_est = pd.read_csv('./estimated_angle3.csv').drop('Unnamed: 0', axis='columns')



v_est[v_est.columns] = np.repeat(0.75,14)
v_est_lwr_bnd= np.array(v_est.iloc[0,:])
theta_est[theta_est.columns] = np.concatenate(([0],np.repeat(-120, 13)),axis =0)
theta_est_lwr_bnd= np.array(theta_est.iloc[0,:])


x_upr_bnd = np.concatenate((v_est_upr_bnd, theta_est_upr_bnd), axis=0)
x_lwr_bnd = np.concatenate((v_est_lwr_bnd, theta_est_lwr_bnd), axis=0)

lwr_bnd =(x_lwr_bnd.flatten())
upr_bnd =(x_upr_bnd.flatten())


# -------------------------bounds are defined for 8 buses only----------------------------