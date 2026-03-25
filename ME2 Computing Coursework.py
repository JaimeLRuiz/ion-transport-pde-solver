
#Hello. Welcome to the computing coursework done by Jaime and Enrico. 
#I hope you enjoy. I've tried to make it all as clear as possible.
#The code for the animations and plots has less comments as functionality and footprint
#was prioritized over clarity of the code. I hope you understand.



#In this section I am importing all the libraries I will need

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, LinearSegmentedColormap





#In this section I am setting the domain of solution and the discretised grid

dx = .5 #micrometers 
#dx = 0.25 for fine grid analysis
#dx = 1 for coarse grid analysis
dt = 10 #seconds
x = np.arange(0, 50+dx, dx)

t = np.arange(0, 20000, dt)
#initialize our solution array
C = np.ndarray((len(x), len(t)))





#In this section I am defining arrays or parameters I would need 

C_h = 1000 #mol/m^3 
C_l = 100 #mol/m^3 
C_ave_discharged = 170 #mol/m^3 
alpha = 2 #1/microm 
D = .0035 #microm^2/sec 
muE = .005
k= 0.0002 #1/sec.      





#In this section I am setting the boundary conditions/initial values

#initial condition (boundary conditions will be implemented in the numerical method)
for i in range(len(x)):
    C[i,0] = C_l + (C_h - C_l)*np.exp(-alpha*x[i])
    
    
    
    
    
#In this section I am implementing the numerical method

#EXPLICIT, FCTS

def ExplicitSolver(x, t, C):
    
    #loop through time
    for p in range(1, len(t)):
        #set boundary conditions
        C[0,p] = C_l + (C_h - C_l)*np.exp(-k*t[p-1])  #dirichlet BC (known concentration), anode
        C[-1,p] = C[-2,p-1] #neumann BC (0 flux), cathode
        
        #loop through space
        for i in range(1, len(x)-1): #1 and len(x)-1 because boundary conditions are set
            C[i,p]= dt*((D*(C[i+1,p-1]-2*C[i,p-1]+C[i-1,p-1])/(dx**2))-muE*(C[i+1,p-1]-C[i-1,p-1])/(2*dx))+C[i,p-1]
        
        #if the battery is discharged, exit the loop
        #'and p>200' to avoid breaking too soon (perhaps at the beginning the average is low)
        if ((np.sum(C[:,p]))/len(C[:,p])) < C_ave_discharged and p>200: 
            break
        
        #return the solution array and the last time step, useful for graphs later
    return C, p


#IMPLICIT, CRANK-NICOLSON

def MakeVectorB(x, p, C): #function which outputs B vector with x array, time position, and solution array as input 

    B = np.zeros(len(x))  
    
    #simplify code by combining constants into coefficients
    coef1 = D/(2*dx**2)-muE/(4*dx)
    coef2 = 1/dt - D/(dx**2)
    coef3 = D/(2*dx**2)+muE/(4*dx)
    
    #boundary condition at the anode
    B[0]= C_l + (C_h - C_l)*np.exp(-k*t[p])
    
    #main loop
    for i in range(1, len(B)-1):
        B[i] = coef1*C[i+1,p] + coef2*C[i,p] + coef3*C[i-1,p]
    
    #boundary condition at the cathode, zero flux
    B[-1] = 0
    
    return B

def MakeMatrixA(x): #function which outputs A matrix with x array as input 
    
    A = np.zeros((len(x), len(x)))
    
    #set boundary conditions
    A[0,0]= 1
    A[-1,-1]=+1/(dx)
    A[-1,-2]=-1/(dx)
                               
    #make rest of tridiagonal matrix
    for i in range(1, len(x)-1):
        A[i, i-1] = -D/(2*dx**2)-muE/(4*dx)
        A[i, i] = 1/dt + D/(dx**2)
        A[i, i+1] = -D/(2*dx**2)+muE/(4*dx)
    
    return A

def ImplicitSolver(x, t, C): #the meat of the problem
    
    #loop through time
    for p in range(len(t)-1):
        #make matrix
        A = MakeMatrixA(x) 
        #make vector. function uses p, so we can input p
        B = MakeVectorB(x, p, C)
        
        #now solve for the next set of concentration values at next timestep p+1!
        C[:,p+1]= np.dot(np.linalg.inv(A), B)
        
        #again, if battery fully charged, break
        if ((np.sum(C[:,p+1]))/len(C[:,p+1])) < C_ave_discharged and p>200: 
            break
    #return solution array and final timestep, useful for graphs
    return C, p





#In this section I am showing the results

#compute the solutions
C_imp_solution, final_timestep_i = ImplicitSolver(x, t, C)
C_exp_solution, final_timestep_e = ExplicitSolver(x, t, C)

#Figures 1 and 2 in the report. Remove ''' to run code
'''
#Figure 1
plt.figure(1)
plt.xlabel('Distance along the electrolyte (µm)')
plt.ylabel('Lithium ion concentration (mol/m^3)')
plt.title('Lithium ion concentration across the electrolyte')
plt.plot(x,C_imp_solution[:,0], color=(1,0,0))
plt.plot(x,C_imp_solution[:,int(len(t)/10)], color = 'blue')
plt.plot(x,C_imp_solution[:,200], color = 'green')
plt.plot(x,C_imp_solution[:,int(len(t)*2/3)], color = 'blue')
plt.plot(x,C_imp_solution[:,final_timestep_i], color = 'purple')

#Figure 2
plt.figure(2)
plt.xlabel('Time elapsed since discharge began (s)')
plt.ylabel('Lithium ion concentration (mol/m^3)')
plt.title('Lithium ion concentration over time')
t_plot = t[:final_timestep_i]
plt.plot(t_plot,C_imp_solution[0,:final_timestep_i], color = 'blue')
plt.plot(t_plot,C_imp_solution[int(len(x)/3),:final_timestep_i], color = 'green')
plt.plot(t_plot,C_imp_solution[int(len(x)*2/3),:final_timestep_i], color = 'blue')
plt.plot(t_plot,C_imp_solution[-1,:final_timestep_i], color = 'blue')
'''


#Figures 3 and 4 in the report. Remove ''' to run code
'''
#Figure 3(a) #make dx = 0.25, Figure 4(a) make dx = 1
plt.xlabel('Time elapsed since discharge began (s)')
plt.ylabel('Lithium ion concentration (mol/m^3)')
plt.title('Lithium ion concentration over time')
plt.ylim(0,1000)
t_plot = t[:final_timestep_e]
plt.plot(t_plot,C_exp_solution[int(len(x)/3),:final_timestep_e], color = 'green')
#Figure 3(b) #make dx = 0.25, Figure 4(b) make dx = 1
plt.xlabel('Distance along the electrolyte (µm)')
plt.ylabel('Lithium ion concentration (mol/m^3)')
plt.title('Lithium ion concentration across the electrolyte')
t_plot = t[:final_timestep_e]
plt.plot(x,C_exp_solution[:,200], color = 'green')
'''

#Heat map animation. Remove ''' to run code
'''
C_solution = C_imp_solution
length = x[-1]
y_repeat = 8  # Arbitrary, for visualization thickness
C_vis = np.tile(C_solution[:, -1], (y_repeat, 1))
fig, ax = plt.subplots(figsize=(8, 4))
yellow_rgb = (1, 1, 0)  # Yellow
purple_rgb = (0.5, 0, 0.5)  # Purple
dark_blue_rgb = (0, 0, 0.5)  # Dark blue
orange_rgb = (1, 0.5, 0)    # Orange
cmap_colors = [dark_blue_rgb, purple_rgb, orange_rgb, yellow_rgb]
cmap = LinearSegmentedColormap.from_list("custom_cmap", cmap_colors)
norm = Normalize(vmin=100, vmax=1000)
cax = ax.imshow(C_vis, cmap=cmap, interpolation='nearest', aspect='auto', extent=[0, length, 0, 1], norm=norm)
cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
cbar.set_ticks([100, 1000])
cbar.set_ticklabels(['Low (100 mol/m³)', 'High (1000 mol/m³)'])
ax.set_title('Lithium Ion Concentration in the Electrolyte')
ax.set_xticks([])
ax.set_yticks([])

time_text = ax.text(.5, -.1, '', transform=ax.transAxes, ha='center', fontsize=12)
# Function to update the time indicator text
def update_time_indicator(frame):
    time_text.set_text('t = {} s'.format(frame * dt))
    
def update(frame):
    update_time_indicator(frame)
    cax.set_data(C_solution[:, frame:frame+1])
    return cax, time_text

ax.set_title('Lithium Ion Concentration in the Electrolyte', fontsize=14, pad=20)
ax.title.set_position([0.5, 1.05])  # Adjust the position of the title

ax.xaxis.set_label_coords(0.5, -0.1) 
ax.yaxis.set_label_coords(-0.1, 0.5) 
ax.text(1.6, 1.05, 'Anode', ha='center', va='center', fontsize=12)
ax.text(2, -0.1, 'Cathode', ha='center', va='center', fontsize=12)
plt.subplots_adjust(top=0.85)

ani = FuncAnimation(fig, update, frames=range(0, len(t), 10), blit=True, interval=.001)
'''

#Implicit vs Explicit comparison animation. Remove ''' to run code
'''
fig, ax = plt.subplots()
ax.set_xlabel('Position in Electrolyte (μm)')
ax.set_ylabel('Concentration (mol/m³)')
ax.set_title('Concentration Profile Over Time')

line1, = ax.plot([], [], lw=6, label='Implicit method')
line2, = ax.plot([], [], lw=2, color='red', label='Explicit method')  # Change color as needed

def init():
    ax.set_xlim(0, x[-1])  # Correct for x-axis
    ax.set_ylim(0, 1000)  # Adjust for actual concentration range
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2,

time_text = ax.text(0.8, 0.75, '', transform=ax.transAxes, ha='center', fontsize=12)
def update_time_indicator(frame):
    time_text.set_text('t = {} s'.format(frame * dt))

def update(frame):
    line1.set_data(x, C_imp_solution[:, frame])
    line2.set_data(x, C_exp_solution[:, frame])
    update_time_indicator(frame) 
    return line1, line2, time_text

ani = FuncAnimation(fig, update, frames=range(0, len(t), 10), init_func=init, blit=True, interval=1)

plt.legend()
'''





print('CW done: I deserve a good mark')
 