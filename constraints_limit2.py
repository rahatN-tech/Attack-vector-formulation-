import pandapower as pp
import numpy as np
import pandas as pd
import pandapower.networks as nw
from numpy.linalg import norm
import pandapower.estimation as est
import simbench as sb
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
# import data
# import bounds

timestamps =4

net =nw.case14()
pp.runpp(net)

v_est = pd.read_csv('./estimated_v3.csv').drop('Unnamed: 0', axis='columns')
theta_est = pd.read_csv('./estimated_angle3.csv').drop('Unnamed: 0', axis='columns')


updated_attk_list = [9,10,11,12,13,8,5,6]


gen_bus =[]
load_bus =[]
zero_inj = []

for n in range(len(net.bus)):

   if (net.res_bus.loc[n, 'p_mw']) < 0:
       gen_bus.append(n)

   elif (net.res_bus.loc[n, 'p_mw'])> 0:
       load_bus.append(n)

   else:
      zero_inj.append(n)


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

# print(col_load_p)


col = col_p + col_q
col_load = col_load_p + col_load_q


z_meas_one =(pd.read_csv('./z_meas_ordered.csv').drop('Unnamed: 0', axis ='columns'))
# print(z_meas_one.iloc[0,3])
upr_bound_p =[]
lwr_bound_p =[]
upr_bound_q =[]
lwr_bound_q =[]


for t in range(timestamps):

    actual_p = []
    actual_q= []
    for i in range(len(net.bus)):
    # for i in updated_attk_list:
        temp_p = z_meas_one.iloc[t,i]
        temp_q = z_meas_one.iloc[t,(i+14)]
        actual_p.append(temp_p)
        actual_q.append(temp_q)

    # print("actual_p",actual_p)
    temp_p = np.take(actual_p, updated_attk_list)
   
    actual_val = actual_p + actual_q
    # print(len(actual_val))

    upr_bnd_p =[]
    lwr_bnd_p =[]
    upr_bnd_q =[]
    lwr_bnd_q =[]
    # for i in range(2*(len(net.bus))):
    # ==========================================================for p=========================================================================================
    for i in updated_attk_list:
        if ((actual_p[i])!=0.0):
           
       
            if ((actual_p[i])<0.0):
               
                temp_upr_p = actual_p[i]
                
                temp_lwr_p = 5*actual_p[i]
               
                
                upr_bnd_p.append(temp_upr_p)
                lwr_bnd_p.append(temp_lwr_p)
                

            else:
                
                temp_upr_p =5* actual_p[i]
                
                temp_lwr_p =0
              
                upr_bnd_p.append(temp_upr_p)
                lwr_bnd_p.append(temp_lwr_p)
               

                
        else:
        
            upr_bnd_p.append(0)
            lwr_bnd_p.append(0)
         
# =======================for q=========================================================
    for i in updated_attk_list:
        if ((actual_q[i])!=0.0):
           
       
            if ((actual_q[i])<0.0):
                # print(f"q is negative for {i}")
                temp_upr_q = 0
                
                temp_lwr_q = 5*actual_q[i] 
               
                
                upr_bnd_q.append(temp_upr_q)
                lwr_bnd_q.append(temp_lwr_q)
                

            else:
                # print(f"q is positive for {i}")
                temp_upr_q =  5*actual_q[i] 
               
                temp_lwr_q = 0
              
                upr_bnd_q.append(temp_upr_q)
                lwr_bnd_q.append(temp_lwr_q)
               

                
        else:
        # print('into else')
            upr_bnd_q.append(0)
            lwr_bnd_q.append(0)
           




# ======================================================================================================================================================
   


upr_bnd =upr_bnd_p + upr_bnd_q
lwr_bnd = lwr_bnd_p +lwr_bnd_q











