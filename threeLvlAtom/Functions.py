# Basado en http://scipy-cookbook.readthedocs.io/items/CoupledSpringMassSystem.html

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def blochEqs(w, t, p):
    """Voy a hacer una prueba para el átomo de 3 niveles:
    s22=sigma22
    s33=sigma33
    sr21=Re(sigma21)
    si21=Im(sigma21)
    sr31=Re(sigma31)
    si31=Im(sigma31)
    sr32=Re(sigma32)
    si32=Im(sigma32)
    g=gamma
    d=delta
    or=Re(Omega)
    oi=Im(Omega)
    """
    s22, s33, sr21, sr31, sr32, si21, si31, si32 = w
    g21, g31, g32, d21, d31, d32, or21, or31, or32, oi21, oi31, oi32 = p

    # Create f = (s22', s33', sr21', sr31', sr32', si21', si31', si32'):
    f = [g32*s33-g21*s22 + 2*oi21*sr21-2*or21*si21 - 2*oi32*sr32+2*or32*si32 ,
        -(g32+g31)*s33 + 2*oi31*sr31-2*or31*si31 + 2*oi32*sr32-2*or32*si32 ,
###########################################################################################################
        -d21*si21 - (1/2)*(g21)*sr21 - oi21*(2*s22+s33) - oi31*sr32+or31*si32 - oi32*sr31+or32*si31 + oi21,
        -d31*si31 - (1/2)*(g31+g32)*sr31 - oi31*(2*s33+s22) - oi21*sr32-or21*si32 + oi32*sr21+or32*si21 + oi31,
        -d32*si32 - (1/2)*(g21+g31+g32)*sr32 - oi32*(s33-s22) + oi21*sr31-or21*si31 + oi31*sr21-or31*si21 ,
###########################################################################################################
        d21*sr21 - (1/2)*(g21)*si21 + or21*(2*s22+s33) + or31*sr32+oi31*si32 - or32*sr31-oi32*si31 - or21,
        d31*sr31 - (1/2)*(g31+g32)*si31 + or31*(2*s33+s22) + or21*sr32-oi21*si32 - or32*sr21+oi32*si21 - or31,
        d32*sr32 - (1/2)*(g21+g31+g32)*si32 + or32*(s33-s22) + or21*sr31+oi21*si31 - or31*sr21-oi31*si21 ]
    return f




# La siguiente función toma como argumentos:
#           tiempo (arreglo1D), condiciones iniciales (arreglo1D) y parámetros ec. Bloch (arreglo1D)
# La función resuelve la parte dinámica y estacionaria y guarda las soluciones en:
#           'threeLvlAtom_Dynam.dat' y 'threeLvlAtom_Stat.dat'
# Además guarda los parámetros utilizados en 'threeLvlAtom_Param.dat'
# La salida de la función es una tupla con 8 valores, correspondientes a la solución estacionaria
def solver(t,w0,p):
    #s22, s33, sr21, sr31, sr32, si21, si31, si32 = w0
    #g21, g31, g32, d21, d31, d32, or21, or31, or32, oi21, oi31, oi32 = p

    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-6

    # Call the ODE solver.
    wsol = odeint(blochEqs, w0, t, args=(p,),
                  atol=abserr, rtol=relerr)

    # Save the solutions
    with open('threeLvlAtom_Dynam.dat', 'w') as f:
        # Print & save the solution.
        for t1, w1 in zip(t, wsol):
            print(t1, w1[0], w1[1], w1[2], w1[3], w1[4], w1[5], w1[6], w1[7], file= f)
        f.close

    ##### Ahora la parte de la solución estacionaria (ver notas 4/sept/16):
    #a = np.array([  [-g21,    g32,    2*oi21,      0,  -2*oi32,  -2*or21,      0,  2*or32],
    #                [0, -(g32+g31),   0,      2*oi31,   2*oi32,   0,     -2*or31,  -2*or32],
    #                [-2*oi21, -oi21, -(1/2)*(g21),       -oi32     ,           -oi31      ,  -d21,  or32,  or31],
    #                [-oi31, -2*oi31,      oi32,    -(1/2)*(g31+g32),           -oi21      ,  or32,  -d31,  -or21],
    #                [oi32,    -oi32,      oi31,           oi21     ,  -(1/2)*(g21+g31+g32),  -or31,  -or21,  -d32],
    #                [2*or21, or21,     d21, -or32,  or31,  -(1/2)*(g21),      -oi32      ,         oi31         ],
    #                [or31, 2*or31,   -or32,   d31,  or21,      oi32    , -(1/2)*(g31+g32),        -oi21         ],
    #                [-or32, or32,    -or31,  or21,   d32,     -oi31    ,       oi21      ,  -(1/2)*(g21+g31+g32)]])
    a = np.array([  [-p[0],    p[2],    2*p[9],      0,  -2*p[11],  -2*p[6],      0,  2*p[8]],
                [0, -(p[2]+p[1]),   0,      2*p[10],   2*p[11],   0,     -2*p[7],  -2*p[8]],
                [-2*p[9], -p[9], -(1/2)*(p[0]),       -p[11]     ,           -p[10]      ,  -p[3],  p[8],  p[7]],
                [-p[10], -2*p[10],      p[11],    -(1/2)*(p[1]+p[2]),           -p[9]      ,  p[8],  -p[4],  -p[6]],
                [p[11],    -p[11],      p[10],           p[9]     ,  -(1/2)*(p[0]+p[1]+p[2]),  -p[7],  -p[6],  -p[5]],
                [2*p[6], p[6],     p[3], -p[8],  p[7],  -(1/2)*(p[0]),      -p[11]      ,         p[10]         ],
                [p[7], 2*p[7],   -p[8],   p[4],  p[6],      p[11]    , -(1/2)*(p[1]+p[2]),        -p[9]         ],
                [-p[8], p[8],    -p[7],  p[6],   p[5],     -p[10]    ,       p[9]      ,  -(1/2)*(p[0]+p[1]+p[2])]])
    b = np.array([0,0,-p[9],-p[10],0,p[6],p[7],0])
    x = np.linalg.solve(a, b)

    # Guarda la solución estacionaria
    with open('threeLvlAtom_Stat.dat', 'w') as f:
        print(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7], file= f)
        f.close

    # Guarda los parámetros elegidos
    with open('threeLvlAtom_Param.dat', 'w') as f:
        print(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], file= f)
        f.close

    # La función regresa la solución estacionaria
    return (x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7])




##### Sólo solución estacionaria #####
# La siguiente función toma como argumentos:
#           parámetros ec. Bloch (arreglo1D) y un booleano (True,False) que define si se guarda o no la solución
# La función resuelve únicamente la parte estacionaria, guarda las soluciones y los parámetros
#           en 'threeLvlAtom_Stat.dat' y en 'threeLvlAtom_Param.dat'
# La salida de la función es una tupla con 8 valores, correspondientes a la solución estacionaria
def solverEst(p,guarda):
    #Solución estacionaria (ver notas 4/sept/16):
    #a = np.array([  [-g21,    g32,    2*oi21,      0,  -2*oi32,  -2*or21,      0,  2*or32],
    #                [0, -(g32+g31),   0,      2*oi31,   2*oi32,   0,     -2*or31,  -2*or32],
    #                [-2*oi21, -oi21, -(1/2)*(g21),       -oi32     ,           -oi31      ,  -d21,  or32,  or31],
    #                [-oi31, -2*oi31,      oi32,    -(1/2)*(g31+g32),           -oi21      ,  or32,  -d31,  -or21],
    #                [oi32,    -oi32,      oi31,           oi21     ,  -(1/2)*(g21+g31+g32),  -or31,  -or21,  -d32],
    #                [2*or21, or21,     d21, -or32,  or31,  -(1/2)*(g21),      -oi32      ,         oi31         ],
    #                [or31, 2*or31,   -or32,   d31,  or21,      oi32    , -(1/2)*(g31+g32),        -oi21         ],
    #                [-or32, or32,    -or31,  or21,   d32,     -oi31    ,       oi21      ,  -(1/2)*(g21+g31+g32)]])
    a = np.array([  [-p[0],    p[2],    2*p[9],      0,  -2*p[11],  -2*p[6],      0,  2*p[8]],
                [0, -(p[2]+p[1]),   0,      2*p[10],   2*p[11],   0,     -2*p[7],  -2*p[8]],
                [-2*p[9], -p[9], -(1/2)*(p[0]),       -p[11]     ,           -p[10]      ,  -p[3],  p[8],  p[7]],
                [-p[10], -2*p[10],      p[11],    -(1/2)*(p[1]+p[2]),           -p[9]      ,  p[8],  -p[4],  -p[6]],
                [p[11],    -p[11],      p[10],           p[9]     ,  -(1/2)*(p[0]+p[1]+p[2]),  -p[7],  -p[6],  -p[5]],
                [2*p[6], p[6],     p[3], -p[8],  p[7],  -(1/2)*(p[0]),      -p[11]      ,         p[10]         ],
                [p[7], 2*p[7],   -p[8],   p[4],  p[6],      p[11]    , -(1/2)*(p[1]+p[2]),        -p[9]         ],
                [-p[8], p[8],    -p[7],  p[6],   p[5],     -p[10]    ,       p[9]      ,  -(1/2)*(p[0]+p[1]+p[2])]])
    b = np.array([0,0,-p[9],-p[10],0,p[6],p[7],0])
    x = np.linalg.solve(a, b)

    if guarda:
        with open('threeLvlAtom_Stat.dat', 'w') as f:
            print(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7], file= f)
            f.close
        # Para guardar los parámetros elegidos
        with open('threeLvlAtom_Param.dat', 'w') as f:
            print(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], file= f)
            f.close
    # Para referencia:    x = [s22, s33, sr21, sr31, sr32, si21, si31, si32]
    return np.array([x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]])








# Hago una función para calcular y graficar soluciones estacionarias cuando se varía la desintonía d32
def grafiStat(g21, g31, g32, d21, d31, d32, or21, or31, or32, oi21, oi31, oi32):
    solText=['s22','s33','sr21','sr31','sr32','si21','si31','si32'] #para títulos de gráficas
    solu= np.zeros([len(d32),8])
    for i in range(len(d32)):
        p = [g21, g31, g32, d21, d31[i], d32[i], or21, or31, or32, oi21, oi31, oi32]
        solu[i]=solverEst(p,False)
    #
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=[12,4])
    ax1.plot(d32,solu[:,0],lw=2)
    ax1.set_title(solText[0])
    ax1.set_xlabel(r'$\delta_{32}$ ¿unidades?')
    ax1.grid()
    #
    ax2.plot(d32,solu[:,1],lw=2)
    ax2.set_title(solText[1])
    ax2.set_xlabel(r'$\delta_{32}$ ¿unidades?')
    ax2.grid()
    plt.tight_layout()
    #plt.show()


    for i in range(3):
        ejeY=True
        if i==0:
            ejeY=False
        f, (ax1, ax2) = plt.subplots(1, 2,figsize=[12,4],sharey=ejeY)
        ax1.plot(d32,solu[:,i+2],lw=2)
        ax1.set_title(solText[i+2])
        ax1.set_xlabel(r'$\delta_{32}$ ¿unidades?')
        ax1.grid()
        #
        ax2.plot(d32,solu[:,i+5],lw=2)
        ax2.set_title(solText[i+5])
        ax2.set_xlabel(r'$\delta_{32}$ ¿unidades?')
        ax2.grid()
        plt.tight_layout()
        #plt.show()
    plt.show()
