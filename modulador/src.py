###


from os import listdir # función para enlistar el contenido de un directorio
from sys import path # función para enlistar los directorios de donde se extraen módulos
from subprocess import call #función para correr comandos de Ubuntu
from subprocess import Popen #función para correr eog sin esperar a cerrar ventana para continuar
import numpy as np
from time import sleep
from time import strftime #función para obtener fecha y/o hora local
from PIL import Image

import modulador.dospi


# para obtener directorio:
for paths in path:
    if 'MisPys' in paths: # en la carpeta MisPys/ es donde está este módulo
        direct=paths
        break
direct=direct+'/modulador/'


print("Bienvenido al módulo que controla el SLM. \n Para empezar utiliza la función inicia. \n ¡¡Recuerda correr la función finaliza al terminar tu sesión!!")
nlong = int(input('¿Qué long. de onda desea utilizar (1), (2), (3) ó (4)? \n 1 -> ninguna \n 2 -> 911 nm \n 3 -> 780 nm \n 4 -> 633 nm \n'))
CONST_2PI=modulador.dospi.DOSPI[nlong-1]
if nlong==1:
    PATRON_CORRECTOR=np.zeros(shape=(600,800),dtype=np.uint8)
else:
    for archvs in listdir(direct+'PatronCorrector/'):
        if ('n%d' %nlong) in archvs:
            direct1=direct+'PatronCorrector/'+archvs
            break
    PATRON_CORRECTOR=np.array(   np.concatenate( (np.array(Image.open(direct1)),np.zeros(shape=(600,8))) ,axis=1 )   ,dtype=np.uint8)

#La función grayImage toma como argumento una matriz de enteros de nVer × nHor(como la que genera blazeMat) y la convierte en una imagen.
def grayImageCorr(matInt:"ndarray"):
    #matInt=(modul.PATRON_CORRECTOR+matInt) # %CONST_2PI # % es para obtener módulo
    return Image.fromarray(PATRON_CORRECTOR+matInt,'L')  # El tipo 'L' es para "8-bit pixels, black and white"
#
def grayImage(matInt:"ndarray"):
    return Image.fromarray(matInt,'L')  # El tipo 'L' es para "8-bit pixels, black and white"

dir1=direct+'PrepMonit1'
dir2=direct+'PrepMonit2'
dir3=direct+'PrepMonit3'
dir4=direct+'imagen.png'
def inicia():
    resp1 = input('¿El SLM está conectado como único monitor a otra computadora (controlada por ssh)? (s/n)')
    if resp1=='n':
        resp2 = input('¿El SLM está conectado a esta computadora como segundo monitor? (s/n)')
        if resp2=='s':
            #Para preparar el 2do monitor (colocación, resolución y orientación)
            call(['bash',dir1])
            sleep(3)
            #Para abrir EOG en 2do monitor Fullscreen
            call(['bash',dir2])
            sleep(2)
        elif resp2=='n':
            #Para preparar la resolución del monitor
            call(['bash',dir1])
            print('Debes editar programa. Contacta a Santiago si tienes dudas: shgaeo@yahoo.com.mx')
    elif resp1=='s':
        print('Asegurate de haber corrido (antes de abrir pyhton) el comando $$$ export DISPLAY=:0')
    ##### Esto es para abrir Eye of Gnome
    Popen(['eog','--fullscreen',dir4,'&'])
    ##### Esto es para abrir Eye of Gnome
def finaliza():
    # #Images.imwrite(grayImage(ones(Int64,600,800)),dir4)
    # save(dir4,grayImage(np.zeros(shape=(600,800),dtype=np.uint8)))
    tempImg=grayImage(np.zeros(shape=(600,800),dtype=np.uint8))
    tempImg.save(dir4)
    resp2 = input('¿El SLM está conectado a esta computadora como segundo monitor? (s/n)')
    if resp2=='s':
        call(['bash',dir3])
def monitor2(imagen:"Image"):
    if imagen.size == (800,600):
        imagen.save(dir4)
    else:
        print('ERROR: Deben ser imágenes de 800x600')
        return
    sleep(1.1) # esto es para que le dé tiempo de guardar y cambiar la imagen en eog Viewer
    return None

# Guarda imágen en negro para proyectar en modulador al iniciar.
grayImage(np.zeros(shape=(600,800),dtype=np.uint8)).save(dir4)

#La función blazeMat da un arreglo de cols×rengs enteros que representa una rejilla blaze.
#El primer argumento es el periodo (en pixeles, i.e. debe ser un entero)
#El segundo argumento es el entero que representa la fase 2π.
def blazeMat(periodo,dosPi=CONST_2PI,cols=800,rengs=600):
    reng = np.array(  (dosPi)*np.mod(  np.linspace(0,cols-1,cols, dtype=int),  periodo)/(periodo-1)  ,dtype=np.uint8)
    return np.tile(reng,(rengs,1))

#La función escalon da un arreglo de cols×rengs enteros que representa una rejilla binaria.
#El primer argumento es el periodo (en pixeles, i.e. debe ser un entero)
#El segundo(tercer) argumento es el valor que toma la cresta(valle) de la rejilla binaria.
def escalon(periodo,dosPi=CONST_2PI,fondo=0,cols=800,rengs=600):
    if (fondo<0 or fondo>255 or dosPi<0 or dosPi>255 or dosPi<fondo):
        print('ERROR: debe ocurrir que 0 <= fondo <= dosPi <= 255')
        return
    else:
        if np.mod(periodo,2) != 0: #si periodo no es par
            print('WARNING: esta función está optimizada cuando "periodo" es un número par')
        reng= np.array(  fondo+(dosPi-fondo)*np.round( np.mod( np.linspace(0,cols-1,cols) , periodo)/(periodo-1) )  ,dtype=np.uint8)
        return np.tile(reng,(rengs,1))

#La función thetaMat da un arreglo de 800x600 cuyas entradas representan los ángulos (van de -π a π)
#El argumento th son los grados por los cuales se puede rotar el holograma
#Recuerda que la convención para matrices es invertir eje Y, por eso valores negativos quedan arriba
def thetaMat(th=0):
    x = np.tile(  np.linspace(-400,400-1,800,dtype=np.int)  ,  (600,1)  )
    y = np.transpose(np.tile(  np.linspace(300,-300+1,600,dtype=np.int)  ,  (800,1)  ))
    xp = np.cos(th*np.pi/180)*x-np.sin(th*np.pi/180)*y # rotacion
    yp = np.sin(th*np.pi/180)*x+np.cos(th*np.pi/180)*y
    return np.arctan2(y,x)

#La función thetaMatInt toma la matriz de la función thetaMat y la convierte en enteros de 8-bits
#Esta es la función que se utiliza para darle vórtice al haz para (junto con axicón) generar el Bessel
def thetaMatInt(n,dosPi=CONST_2PI,th=0):
    return np.array(  np.mod( n*(dosPi)*( 0.5 + thetaMat(th)/(2*np.pi) ) , dosPi+1)  ,dtype=np.uint8)

### Lo siguiente está comentado porque sólo funciona con webcams (USB) que no permiten controlar casi nada.
# #La siguiente función es para capturar imágenes, para configuración ver ./webcamConfig
# dir5=direct+'webcamConfig'
# directCapt=direct+'capturas/'
# directCal=direct+'calib/'
# def capturaImg(nombre=strftime("%Y%m%d-%H%M%S")):
#     dirTemp=directCapt+nombre+'.jpeg'
#     call(['fswebcam','-c',dir5,'--save',dirTemp])
# def capturaImgCal(n):
#     dirTemp=directCal+n+'.jpeg'
#     call(['fswebcam','-c',dir5,'--save',dirTemp])

#La siguiente función es para calibrar el SLM
#Lo que hace es esperar a que tomes una foto (hasta ahora esto debe hacerse manualmente y de manera independiente)
#para cada nivel de la función escalón
def calibrar(inicio=0,fin=255):
    print('Una vex finalizado el análisis debes incluir el resultado en "./dospi.py"')
    print('Tras lo cual debes importar nuevamente el módulo para que se tomen en cuenta los cambios')
    for i in range(inicio,fin+1):
        monitor2(grayImage(escalon(10,i))) #Este periodo permite ver los órdenes 0,1,2 en un CCD de webcam común
        resp = input('Capturar foto para i='+str(i)+'. Enter -> siguiente (abort -> interrumpir)')
        if resp=='abort':
            print('Proceso interrumpido.')
            break
    return
