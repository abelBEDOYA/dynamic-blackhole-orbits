# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:48:34 2021

@author: Carlos Beltran

Este fichero contiene comandos que permiten usar mas comodamente
Python, asi como algunos programas de la asignatura Metodos Numericos

Siempre comenzaremos cualquier programacion de Metodos cargando
este fichero con "from MyN_python_siempre_contigo.py import *"
"""
#%% Librerias generales:
import numpy as np
import matplotlib.pyplot as plt
#%% Funciones importadas desde numpy:
from numpy import sin
from numpy import cos
from numpy import tan
from numpy import exp
from numpy import log
from numpy import pi
from numpy import e
from numpy.random import rand
from numpy.random import randn
from numpy import array
from numpy.linalg import norm
from numpy import sign
from numpy import abs
from numpy import zeros
from numpy import ones
from numpy import linspace
from numpy import concatenate
from numpy import vstack
from numpy import hstack
from numpy import polyfit
from numpy import polyval
from numpy import interp
from numpy import sqrt
from numpy import arcsin
from numpy import arccos
from numpy import arctan
from numpy import sort
from numpy import argmin
from numpy import append
from numpy import polyder
from numpy import flip
#%% Funciones importadas desde pyplot:
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
from matplotlib.pyplot import xlim
from matplotlib.pyplot import ylim
from matplotlib.pyplot import pause
from matplotlib.pyplot import clf
from matplotlib.pyplot import xlabel
from matplotlib.pyplot import ylabel
from matplotlib.pyplot import title
from matplotlib.pyplot import text


























































#%% Funciones basicas de metodos numericos:   
# %% Metodo de biseccion

def biseccion(f, a, b, epsilon, verbose = 0):
    # f es la funcion que define f(x)=0. a y b son dos puntos tales que el signo de f(a) y el de f(b) son distintos. Finalmente, "e" es el "epsilon" que marca cuando consideramos que podemos parar
    fa = f(a)
    fb = f(b)
    if sign(fa) == sign(fb):
        print("Error: la funcion tiene igual signo en los extremos")
        return # La funcion no devuelve ningun resultado
    else:
        if verbose:
            print("a=Extremo_izdo\tc=Punto_Medio\tb=Extremo_dcho\t\tf(c)") # Escribiremos una tabla para ir mostrando los resultados
        c = (a+b)/2
        fc = f(c)
        if verbose:
            print(f"{a:.5f}\t\t\t{c:.5f}\t\t\t{b:.5f}\t\t\t{fc:.5f}")
        condicion_parada = (abs(fc) < epsilon) or (abs(b-a)<epsilon)
        while not(condicion_parada):
            if sign(fc) == sign(fa):
                a = c # El nuevo intervalo sera [c,b], ahora se llama [a,b]
                fa = f(a)
                c = (a+b)/2 # El nuevo punto medio
                fc = f(c)
            else:
                b = c # El nuevo intervalo sera [a,c], ahora se llama [a,b]
                fb = f(b)
                c = (a+b)/2 # El nuevo punto medio
                fc = f(c)
            condicion_parada = (abs(fc) < epsilon) or (abs(b-a)<epsilon)
            if verbose:
                print(f"{a:.5f}\t\t\t{c:.5f}\t\t\t{b:.5f}\t\t\t{fc:.5f}")
        if verbose:
            print(f"La solucion aproximada es c={c:.5f} que cumple f(c)={fc:.5f}")
    return c, fc

# %% Metodo de Newton en su version practica
def Newton(f, x, h, epsilon, verbose = 0):
    # f es la funcion que define f(X)=0. x es el valor inicial dado a la iteracion. h es el parametro usado para calcular la derivada de forma aproximada. "e" es el epsilon usado para decidir cuando detener la iteracion.
    fx = f(x)
    contador = 1 # Contaremos cuantas iteraciones hacemos
    condicion_parada = (abs(fx) < epsilon) or (contador>100) # A lo sumo 100 iteraciones
    if verbose:
        print("x=Punto_actual\t f(x)") # Escribiremos una tabla para ir mostrando los resultados
        print(f"{x:.5f}\t\t\t{fx:.5f}")
    while not(condicion_parada):
        denominador = f(x+h) - f(x-h) 
        if (denominador == 0) and (fx!=0): # Nota: "!=" significa "distinto de"
            print("Error: he encontrado una division por 0, intenta con otro punto inicial")
        else:
            x = x - 2*h*fx/denominador
            fx = f(x)
            contador = contador + 1
            if verbose:
                print(f"{x:.5f}\t\t\t{fx:.5f}")
        condicion_parada = (abs(fx) < epsilon) or (contador>100)
    if contador > 100:
        if verbose:
            print("No se ha alcanzado convergencia, intenta con otro punto inicial")
        return
    else:
        if verbose:
            print(f"La solucion aproximada es x={x:.5f} que cumple f(x)={fx:.5f}")
    return x,fx

    # %% Metodo del Gradiente para minimizar una funcion de n variables
def Minimiza(f, v0, h, epsilon, N, verbose = 0):
    # Dada una funcion de n variables, empezando por v0 y con paso controlado por h realiza el metodo del  gradiente para minimizar la funcion. El epsilon es un criterio sobre cuando detenerse. El N es el numero maximo de pasos. Solo recomendamos verbose=1 si n=2. h tiene que ser un numero mayor que y cercano a 0.
    n=len(v0)
    contador = 1
    def gradiente(f, w): # Calcula el gradiente de f en w = [x1,...,xn] de forma aproximada
        grad=0*w
        for j in range(len(w)):
            ej=0*w
            ej[j]=1
            parcialj=(f(w+h*ej)-f(w-h*ej))/2/h;
            grad[j]=parcialj
        return grad
    continua = 1
    if verbose: # Dibujaremos y acumularemos los valores intermedios
        plot(v0[0], v0[1], 'o')
        xlim(-2*abs(v0[0]), 2*abs(v0[1]))
        ylim(-2*abs(v0[1]), 2*abs(v0[1]))
        show()
        pause(0.1)
        vtotal=v0
    paso = .1
    while continua:
        direccion = gradiente(f, v0)
        normadireccion = norm(direccion)
        direccion = direccion/normadireccion
        haymejora = 0
        while (haymejora == 0) and (contador<=N) and (paso>epsilon):
            v0new = v0-paso*direccion
            contador = contador+1
            if f(v0new)<f(v0):
                haymejora = 1
                print('mejora')
                v0=v0new
                
                paso =paso*(1+h)
            else:
                paso = paso/(1+h)
            print(paso,contador,v0)
            if verbose:
                vtotal=vstack((vtotal, v0))
                clf()
                plot(vtotal[-1, 0], vtotal[-1, 1], 'xk')
                plot(vtotal[1:, 0], vtotal[1:, 1], 'k')
                xlim(v0[0]-1.1*paso, v0[0]+1.1*paso)
                ylim(v0[1]-1.1*paso, v0[1]+1.1*paso)
                cadena = 'Valor actual de f = '+str(f(v0))
                title(cadena)
                show()
                pause(0.01)
        if contador>N:
            continua = 0
    return v0, f(v0)
# %% Metodo de Euler
def Euler(f, a, b, r0, N, verbose = 0):
    # Calcularemos la solucion dr/dt=f(t,r) con valor inicial r(a)=r0, y con paso h=(b-a)/N, usando el metodo de Euler. Notese que r0 tiene que ser un vector, esto es, un np.array, en cuyo caso r(t) sera una funcion vectorial, de la que nos devolvera el metodo: r(0), r(h), r(2h),...r(Nh=b). Por lo tanto, lo que nos devuelve el metodo es una matriz "T" con N+1 filas y numero de columnas igual a la dimension de r0.
    T = zeros((N+1,r0.size)) #  Fijamos el tamagno de T y la iremos rellenando
    T[0,:] = r0 # la primera fila de T es los elementos de r0
    contador = 0 # Para llevar la cuenta de en que fila estamos
    h = (b-a)/N # El valor de h
    t0 = a + contador*h # El valor del tiempo en este mometo
    if verbose:
        print("----------------------------------------")
        print(f"t={a}")
        print(f"r0={r0}")
    while contador<N:
        r0 = r0 + h*f(a + contador*h, r0) # Formula del metodo de Euler (nnotese que estamos en el momento t = a + contador*h): actualizamos r0
        contador = contador+1
        t0 = t0+h # actualizamos el tiempo
        T[contador,:] = r0 # El valor actualizado
        if verbose:
            print("----------------------------------------")
            print(f"t={t0}")
            print(f"r0={r0}")
            print("----------------------------------------")
    tt = linspace(a, b, N+1) # Que nos devuelva tambien el vector de tiempos
    return tt, T

# %% MetodoRK4
def RK4(f, a, b, r0, N, verbose = 0):
    # Calcularemos la solucion dr/dt=f(t,r) con valor inicial r(a)=r0, y con paso h=(b-a)/N, usando RK4. Notese que r0 tiene que ser un vector, esto es, un np.array, en cuyo caso r(t) sera una funcion vectorial, de la que nos devolvera el metodo: r(0), r(h), r(2h),...r(Nh=b). Por lo tanto, lo que nos devuelve el metodo es una matriz "T" con N+1 filas y numero de columnas igual a la dimension de r0.
    T = zeros((N+1,r0.size)) #  Fijamos el tamagno de T y la iremos rellenando
    T[0,:] = r0 # la primera fila de T es los elementos de r0
    contador = 0 # Para llevar la cuenta de en que fila estamos
    h = (b-a)/N # El valor de h
    t0 = a + contador*h # El valor del tiempo en este mometo
    if verbose:
        print("----------------------------------------")
        print(f"t={a:.5f}")
        print(f"r0={r0}")
    while contador<N:
        # Lo que viene ahora es lo unico que cambia con respecto a Euler
        k_1 = h*f(t0,r0)
        k_2 = h*f(t0+h/2, r0+k_1/2)
        k_3 = h*f(t0+h/2, r0+k_2/2)
        k_4 = h*f(t0+h, r0+k_3)
        r0 = r0+1/6*(k_1+2*k_2+2*k_3+k_4)
        # Ya esta: lo demas es todo igual!
        contador = contador+1
        t0 = t0+h # actualizamos el tiempo
        T[contador,:] = r0 # El valor actualizado
        if verbose:
            print("----------------------------------------")
            print(f"t={t0}")
            print(f"r0={r0}")
            print("----------------------------------------")
    tt = linspace(a, b, N+1) # Que nos devuelva tambien el vector de tiempos
    return tt, T

# %% Problemas de orden 2
def segundo_orden(f, a, b, r0, rdot0, N, verbose = 0):
    # Dada una funcion f(t,r,rdot) que nos da el valor de d^2 r/dt^2, y dado un intervalo temporal [a,b], calcula una aproximacion de la funcion r(t) que satisface la ecuacion con valores iniciales r(t=0)=r0, dr/dt(t=0)=rdot0, mediante el procedimiento de reducirlo a un problema de primer orden y resolver con el metodo de RK4. Nos devolvera tt (los tiempos donde se aproxima la solucion, correspondientes a "a", "a+h", "a+2h", ..., "a+Nh=b$ donde h=(b-a)N), ademas de el valor de r en cada uno de esos tiempos y  tambien el valor de dr/dt en cada uno de esos tiempos, porque ya que los generamos en el proceso pues los acumulamos.
    # Construimos el problema de primer orden (y doble de dimension) correspondiente a f: consideraremos una nueva funcion inventada w(t)=(r,dr/dt). Con ello,
    p = r0.size*2 # luego el tamagno de r es p/2, y el de dr/dt tambien
    def f_aux(t, w):
        r = w[0:p//2] # Extraemos las primeras p/2 coordenadas: tenemos r
        rdot = w[p//2:] # O sea desde d/2 hasta el final: tenemos dr/dt
        # La derivada de r es dr/dt:
        deri_r = rdot
        # La derivada de dr/dt viene dada por la funcion f original:
        deri_rdot = f(t, r, rdot)
        # Finalmente, acumulamos deri_r y deri_rdot en un solo array y devolvemos:
        wdot = concatenate((deri_r,deri_rdot))
        
        return wdot
    # Ahora resolvemos este problema de primer orden. Primero definimos el valor inicial para este nuevo problema:
    w0 = concatenate((r0, rdot0))
    tt, T = RK4(f_aux, a, b, w0, N, verbose)
    # Con ello, T sera una matriz de N+1 filas y p columnas. Las primeras p/2 corresponden a r(tt) y las ultimas p/2 corresponden a dr/dt(tt):
    R = T[:, 0:p//2]
    Rdot = T[:, p//2:]
    return tt, R, Rdot



#%% Programa para generar los nodos de interpolacion o aproximacion optimos.

def Siguientes_nodos(a, b, nodos_previos, n):
    # Genera n nuevos nodos optimos en el intervalo [a,b] si queremos conservar los de la lista nodos_previos.
    # "nodos_previos" es un array de nodos.
    if nodos_previos.size == 0:
        nodos_previos = array([(a+b)/2]) # Si no parto de ningun nodo, empiezo por el punto medio
        n = n-1
    nodos_final = nodos_previos
    for i in range(n):
        # Definimos la funcion a minimizar:
        def minimizame(x):
            Producto = 1
            for j in range(nodos_final.size):
                Producto = Producto * abs(x-nodos_final[j])
            return -Producto
        candidatos = linspace(a, b, 2000) # Probaremos la funcion a minimizar en muchos puntos
        valores = minimizame(candidatos) # Evaluamos la funcion a minimizar en todos ellos
        posicion = (argmin(valores)) # Nos quedamos con el indice del que alcanzo menor valor
        nuevo_nodo = candidatos[posicion] # Este es por lo tanto el nodo que buscamos
        nodos_final = (append(nodos_final, nuevo_nodo))
    print(f'Los nuevos nodos son {sort(nodos_final[-n:])}')
    return (nodos_final)
        
        

    
    
    
    
    
  #%% Ver la representación en IEEE64 de un número:
def ieee64(a):
    if a<0:
        signo='1'
    else:
        signo='0'
    b = np.binary_repr(np.float64(abs(a)).view(np.int64))
    l = len(b)
    if l<63:
        c = ''
        for k in range(63-l):
            c = c+'0'
        b = c+b   
        
    b = signo+b
    print(f'La representacion en IEEE64 de {np.float64(a)} es {b}')
    return b
# Ejemplo de uso:
#b=ieee64((2-2**-52)*2**1023)
#b=ieee64(-np.inf)
#b=ieee64(-np.nan)      

