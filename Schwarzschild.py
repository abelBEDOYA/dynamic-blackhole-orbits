# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 09:28:52 2021

@author: Abel Amado Gonzalez Bernad
"""
import numpy as np
import matplotlib.pyplot as plt
exec(open("metodos_numericos.py").read())
"""
----------------
"""
#CONDICIONES INICIALES

R =  7              # La distancia inicial al centro medida por el observador lejano
thetai = 0          #posicion angular
r0dot = 0           #Velocidad radial
theta0dot = 0.07    #Velocidad angular # Ha de cumplirse r0dot**2+R**2theta0dot**2<c**2
b =   90            # Tiempo propio final del astronauta # 19.10085
mu = 1              #Masa del agujero engro  #Mitad del radio de schwarzschild
"""
----------------
"""
# Constantes del problema:
c = 1


# Consideramos w=[t,r,theta]. 

# Planteamos el sistema de ODEs de segundo orden:
def wddot(sigma, w, wdot):
    t = w[0]
    r = w[1]
    theta = w[2]
    tdot = wdot[0]
    rdot = wdot[1]
    thetadot = wdot[2]
    Factor = -1 + 2*mu/r
    tddot = 2*mu/r**2*rdot*tdot/Factor
    rddot = -Factor*(-mu/r**2*c**2*tdot**2+mu/(2*mu-r)**2*rdot**2+r*thetadot**2)
    thetaddot = -2*rdot*thetadot/r
    wddot = [tddot, rddot, thetaddot]
    return wddot


# Condiciones iniciales para la caida radial:

#Posiciones
ti=0
w0 = array([ti, R, thetai])


#Velocidades
t0dot = sqrt((-c**2-1/(1-2*mu/R)*r0dot**2-R**2*theta0dot**2)/(-1+2*mu/R))/c
w0dot = array([t0dot, r0dot, theta0dot])

#El parametro es el tiempo propio
a = 0 # Tiempo inicial

N = 80000 # Numero de pasos 40000

ss, W, Wdot = segundo_orden(wddot, a, b, w0, w0dot, N, verbose = 0)

#Arrays con la evolucion
t = W[:,0]
r = W[:,1]
theta = W[:,2]

#A cartesianas para poderlo pintar
x=r*cos(theta)
y=r*sin(theta)





#Hay que jugar con el granulado del metodo numerico, el tiepo propio final (b)
#la velocidad a la que pinta el refresco de imagenes, spe.


spe=1000 #Velocidad en la que se pinta la trayectoria


#Los indices de interes para pintar con evolucion relativa al obervador externo
indices = np.array([])
parar = False
for ind in range(round(len(t)/spe)):
    
    if parar==True:
        break
    
    if ind==0:

        indices = np.append(indices, int(1))
        
    elif ind < 3:#round(len(t)/spe/5):
        
        indices = np.append(indices, int(indices[ind-1]+spe))

    elif indices[ind-1] < N-2:

        i1 = int(indices[ind-1])
        i2 = int(indices[ind-2])
        dss = ss[i1]-ss[i2]
        dt =t[i1]-t[i2]+0.001
        c_tiem = 3*(dss/dt)
        
        posible =  int(round(indices[ind-1]+spe*c_tiem))
        if posible < N-1:
            indices = np.append(indices, posible)#int(round(indices[ind-1]+spe*c_tiem)))
        else: 
            parar = True
    else:
        break        
       

C1=2*mu*cos(linspace(0,2*pi, 300))
C2=2*mu*sin(linspace(0,2*pi, 300))
for i in indices:
    
    
    plt.clf() #Refrescar
    #plt.title(r'$R_s$ = '+str(Rs)+', $\mu$ = '+str(mu), fontsize=25)
    plt.title('Simulacion con Rel.General relativa a observador lejano, osea, tú')
    #Agujero negro
    plot(C1,C2, 'black')
    plt.fill(C1,C2,'black')
    
    indice = max(0, i-1000)
    
    indice=0
    
    i = int(i)
    plot(x[indice:i], y[indice:i]) #Trayectoria
    plot(x[i], y[i], 'or', label='Reloj astronauta: {}, tu reloj: {}'.format(round(ss[i], 2), round(t[i],2)))
    
    dina = 2#Para hacerlo dinamico, muy dinamico o estático
    if dina ==0:
        MAX = 2*x[i]#10*2*mu
        
        plt.xlim(-MAX,MAX)
        plt.ylim(-MAX, MAX)
    elif dina ==1:
        med = 0.05*x[i]
        plt.xlim(-med - 3*med*(x[i]-2*mu) + 2*mu + (x[i]-2*mu)/2,med + 2*mu + (x[i]-2*mu)/2 + 3*med*(x[i]-2*mu))
        plt.ylim(-med - 3*med*(x[i]-2*mu) , med + 3*med*(x[i]-2*mu))
        
    else:
        Max= 10
        plt.xlim(-Max,Max)
        plt.ylim(-Max, Max)
    #plt.xlabel('x')
    #plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    plt.pause(0.01)





#%%
'''
plt.plot(t, r)
print(ss[len(ss)-1])
print(t[len(t)-1])

'''