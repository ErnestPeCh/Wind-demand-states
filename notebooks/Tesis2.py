#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import numpy as np


# In[50]:


ojuelos = pd.read_csv("..Data/OJUELOSLIMPIO_.csv")
ojuelos["date"] = pd.to_datetime(ojuelos["date"])
ojuelos.index = ojuelos["date"]
del ojuelos["date"]
ojuelos["Disponibilidad"] = 4

ventosa = pd.read_csv("..Data/VENTOSALIMPIO_.csv")
ventosa["date"] = pd.to_datetime(ventosa["date"])
ventosa.index = ventosa["date"]
del ventosa["date"]
ventosa["Disponibilidad"] = 2

merida = pd.read_csv("..Data/MERIDALIMPIO_.csv")
merida["date"] = pd.to_datetime(merida["date"])
merida.index = merida["date"]
del merida["date"]
merida["Disponibilidad"] = 1

sanfernando = pd.read_csv("..Data/SANFERNANDOLIMPIO_.csv")
sanfernando["date"] = pd.to_datetime(sanfernando["date"])
sanfernando.index = sanfernando["date"]
del sanfernando["date"]
sanfernando["Disponibilidad"] = 5

tepexi = pd.read_csv("..Data/TEPEXILIMPIO_.csv")
tepexi["date"] = pd.to_datetime(tepexi["date"])
tepexi.index = tepexi["date"]
del tepexi["date"]
tepexi["Disponibilidad"] = 3

rumorosa = pd.read_csv("..Data/RUMOROSALIMPIO_.csv")
rumorosa["date"] = pd.to_datetime(rumorosa["date"])
rumorosa.index = rumorosa["date"]
del rumorosa["date"]
rumorosa["Disponibilidad"] = 7

cuauhtemoc = pd.read_csv("..Data/CUAHUTEMOCLIMPIO_.csv")
cuauhtemoc["date"] = pd.to_datetime(cuauhtemoc["date"])
cuauhtemoc.index = cuauhtemoc["date"]
del cuauhtemoc["date"]
cuauhtemoc["Disponibilidad"] = 6


# In[51]:


plt.figure(figsize=(25,5))
rumorosa.Disponibilidad.plot(marker="|",markersize=10,linestyle="None",alpha=1,figsize=(10,5),grid=True)
ventosa.Disponibilidad.plot(marker="|",markersize=10,linestyle="None",alpha=1,figsize=(10,5),grid=True)
tepexi.Disponibilidad.plot(marker="|",markersize=10,linestyle="None",alpha=1,figsize=(10,5),grid=True)
ojuelos.Disponibilidad.plot(marker="|",markersize=10,linestyle="None",alpha=1,figsize=(10,5),grid=True)
sanfernando.Disponibilidad.plot(marker="|",markersize=10,linestyle="None",alpha=1,figsize=(10,5),grid=True)
cuauhtemoc.Disponibilidad.plot(marker="|",markersize=10,linestyle="None",alpha=1,figsize=(10,5),grid=True)
merida.Disponibilidad.plot(marker="|",markersize=10,linestyle="None",alpha=1,figsize=(10,5),grid=True)
plt.legend(["La Rumorosa","La Ventosa","Tepexi","Ojuelos","San Fernando","Cuauhtémoc","Mérida"],bbox_to_anchor=(1.01,0.7))
plt.xticks(rotation=52.5)
plt.xlabel("Fecha")
plt.savefig("Disponibilidad")


# In[52]:


#ESTOS SON LOS ARCHIVOS GENERADOS DEL NOTEBOOK TESIS 1 (DE MATHEMATICA)

#OJUELOS
ojuelos1 = pd.read_csv("Estados estaciones nuevo/ojuelosestado1.csv")
ojuelos1["date"] = pd.to_datetime(ojuelos1["date"])
ojuelos1.index = ojuelos1["date"]
del ojuelos1["date"]

ojuelos3 = pd.read_csv("Estados estaciones nuevo/ojuelosestado3.csv")
ojuelos3["date"] = pd.to_datetime(ojuelos3["date"])
ojuelos3.index = ojuelos3["date"]
del ojuelos3["date"]

ojuelos2 = pd.read_csv("Estados estaciones nuevo/ojuelosestado2.csv")
ojuelos2["date"] = pd.to_datetime(ojuelos2["date"])
ojuelos2.index = ojuelos2["date"]
del ojuelos2["date"]

ojuelos4 = pd.read_csv("Estados estaciones nuevo/ojuelosestado4.csv")
ojuelos4["date"] = pd.to_datetime(ojuelos4["date"])
ojuelos4.index = ojuelos4["date"]
del ojuelos4["date"]


#MERIDA
merida1 = pd.read_csv("Estados estaciones nuevo/meridaestado1.csv")
merida1["date"] = pd.to_datetime(merida1["date"])
merida1.index = merida1["date"]
del merida1["date"]

merida3 = pd.read_csv("Estados estaciones nuevo/meridaestado3.csv")
merida3["date"] = pd.to_datetime(merida3["date"])
merida3.index = merida3["date"]
del merida3["date"]

merida2 = pd.read_csv("Estados estaciones nuevo/meridaestado2.csv")
merida2["date"] = pd.to_datetime(merida2["date"])
merida2.index = merida2["date"]
del merida2["date"]

merida4 = pd.read_csv("Estados estaciones nuevo/meridaestado4.csv")
merida4["date"] = pd.to_datetime(merida4["date"])
merida4.index = merida4["date"]
del merida4["date"]


#VENTOSA
ventosa1 = pd.read_csv("Estados estaciones nuevo/ventosaestado.csv")
ventosa1["date"] = pd.to_datetime(ventosa1["date"])
ventosa1.index = ventosa1["date"]
del ventosa1["date"]


#SAN FERNANDO
sanfer1 = pd.read_csv("Estados estaciones nuevo/sanferestado1.csv")
sanfer1["date"] = pd.to_datetime(sanfer1["date"])
sanfer1.index = sanfer1["date"]
del sanfer1["date"]

sanfer3 = pd.read_csv("Estados estaciones nuevo/sanferestado3.csv")
sanfer3["date"] = pd.to_datetime(sanfer3["date"])
sanfer3.index = sanfer3["date"]
del sanfer3["date"]

sanfer2 = pd.read_csv("Estados estaciones nuevo/sanferestado2.csv")
sanfer2["date"] = pd.to_datetime(sanfer2["date"])
sanfer2.index = sanfer2["date"]
del sanfer2["date"]

sanfer4 = pd.read_csv("Estados estaciones nuevo/sanferestado4.csv")
sanfer4["date"] = pd.to_datetime(sanfer4["date"])
sanfer4.index = sanfer4["date"]
del sanfer4["date"]


#TEPEXI
tepexi1 = pd.read_csv("Estados estaciones nuevo/tepexiestado1.csv")
tepexi1["date"] = pd.to_datetime(tepexi1["date"])
tepexi1.index = tepexi1["date"]
del tepexi1["date"]

tepexi3 = pd.read_csv("Estados estaciones nuevo/tepexiestado3.csv")
tepexi3["date"] = pd.to_datetime(tepexi3["date"])
tepexi3.index = tepexi3["date"]
del tepexi3["date"]

tepexi2 = pd.read_csv("Estados estaciones nuevo/tepexiestado2.csv")
tepexi2["date"] = pd.to_datetime(tepexi2["date"])
tepexi2.index = tepexi2["date"]
del tepexi2["date"]

tepexi4 = pd.read_csv("Estados estaciones nuevo/tepexiestado4.csv")
tepexi4["date"] = pd.to_datetime(tepexi4["date"])
tepexi4.index = tepexi4["date"]
del tepexi4["date"]


#CUAHUTEMOC
cuau4 = pd.read_csv("Estados estaciones nuevo/cuauestado4.csv")
cuau4["date"] = pd.to_datetime(cuau4["date"])
cuau4.index = cuau4["date"]
del cuau4["date"]

cuau2 = pd.read_csv("Estados estaciones nuevo/cuauaestado2.csv")
cuau2["date"] = pd.to_datetime(cuau2["date"])
cuau2.index = cuau2["date"]
del cuau2["date"]

cuau1 = pd.read_csv("Estados estaciones nuevo/cuauestado1.csv")
cuau1["date"] = pd.to_datetime(cuau1["date"])
cuau1.index = cuau1["date"]
del cuau1["date"]

cuau3 = pd.read_csv("Estados estaciones nuevo/cuauestado3.csv")
cuau3["date"] = pd.to_datetime(cuau3["date"])
cuau3.index = cuau3["date"]
del cuau3["date"]


#RUMOROSA
rumorosa4 = pd.read_csv("Estados estaciones nuevo/rumorosaestado4.csv")
rumorosa4["date"] = pd.to_datetime(rumorosa4["date"])
rumorosa4.index = rumorosa4["date"]
del rumorosa4["date"]

rumorosa3 = pd.read_csv("Estados estaciones nuevo/rumorosaestado3.csv")
rumorosa3["date"] = pd.to_datetime(rumorosa3["date"])
rumorosa3.index = rumorosa3["date"]
del rumorosa3["date"]

rumorosa1 = pd.read_csv("Estados estaciones nuevo/rumorosaestado1.csv")
rumorosa1["date"] = pd.to_datetime(rumorosa1["date"])
rumorosa1.index = rumorosa1["date"]
del rumorosa1["date"]

rumorosa2 = pd.read_csv("Estados estaciones nuevo/rumorosaestado2.csv")
rumorosa2["date"] = pd.to_datetime(rumorosa2["date"])
rumorosa2.index = rumorosa2["date"]
del rumorosa2["date"]


# In[75]:


ojuelos4


# In[53]:


#SE SELECCIONAN LOS VALORES IGUALES A 1
ojuelos2 = ojuelos2[ojuelos2["0"] == 1]
ojuelos4 = ojuelos4[ojuelos4["0"] == 1]
ojuelos1 = ojuelos1[ojuelos1["1"] == 1]
ojuelos3 = ojuelos3[ojuelos3["0"] == 1]

merida1 = merida1[merida1["1"] == 1]
merida3 = merida3[merida3["0"] == 1]
merida2 = merida2[merida2["0"] == 1]
merida4 = merida4[merida4["0"] == 1]

ventosa1 = ventosa1[ventosa1["1"] == 1]

sanfer1 = sanfer1[sanfer1["1"] == 1]
sanfer3 = sanfer3[sanfer3["0"] == 1]
sanfer2 = sanfer2[sanfer2["0"] == 1]
sanfer4 = sanfer4[sanfer4["0"] == 1]

tepexi1 = tepexi1[tepexi1["1"] == 1]
tepexi3 = tepexi3[tepexi3["0"] == 1]
tepexi2 = tepexi2[tepexi2["0"] == 1]
tepexi4 = tepexi4[tepexi4["0"] == 1]

cuau2 = cuau2[cuau2["0"] == 1]
cuau4 = cuau4[cuau4["0"] == 1]
cuau1 = cuau1[cuau1["1"] == 1]
cuau3 = cuau3[cuau3["0"] == 1]

rumorosa3 = rumorosa3[rumorosa3["0"] == 1]
rumorosa4 = rumorosa4[rumorosa4["0"] == 1]
rumorosa1 = rumorosa1[rumorosa1["1"] == 1]
rumorosa2 = rumorosa2[rumorosa2["0"] == 1]


# In[54]:


#SE CONCATENAN LOS ARCHIVOS. ESTOS DATAFRAMES CONTIENEN LA VELOCIDAD Y LA DIRECCIÓN PARA CADA ESTADO DE VIENTO.

ojuelos2 = pd.merge(ojuelos, ojuelos2, on='date', how='inner')
ojuelos4 = pd.merge(ojuelos, ojuelos4, on='date', how='inner')
ojuelos1 = pd.merge(ojuelos, ojuelos1, on='date', how='inner')
ojuelos3 = pd.merge(ojuelos, ojuelos3, on='date', how='inner')

merida1 = pd.merge(merida, merida1, on='date', how='inner')
merida3 = pd.merge(merida, merida3, on='date', how='inner')
merida2 = pd.merge(merida, merida2, on='date', how='inner')
merida4 = pd.merge(merida, merida4, on='date', how='inner')

sanfer1 = pd.merge(sanfernando, sanfer1, on='date', how='inner')
sanfer3 = pd.merge(sanfernando, sanfer3, on='date', how='inner')
sanfer2 = pd.merge(sanfernando, sanfer2, on='date', how='inner')
sanfer4 = pd.merge(sanfernando, sanfer4, on='date', how='inner')

tepexi1 = pd.merge(tepexi, tepexi1, on='date', how='inner')
tepexi3 = pd.merge(tepexi, tepexi3, on='date', how='inner')
tepexi2 = pd.merge(tepexi, tepexi2, on='date', how='inner')
tepexi4 = pd.merge(tepexi, tepexi4, on='date', how='inner')

cuau2 = pd.merge(cuauhtemoc, cuau2, on='date', how='inner')
cuau4 = pd.merge(cuauhtemoc, cuau4, on='date', how='inner')
cuau1 = pd.merge(cuauhtemoc, cuau1, on='date', how='inner')
cuau3 = pd.merge(cuauhtemoc, cuau3, on='date', how='inner')

rumorosa3 = pd.merge(rumorosa, rumorosa3, on='date', how='inner')
rumorosa4 = pd.merge(rumorosa, rumorosa4, on='date', how='inner')
rumorosa1 = pd.merge(rumorosa, rumorosa1, on='date', how='inner')
rumorosa2 = pd.merge(rumorosa, rumorosa2, on='date', how='inner')

ventosa1 = pd.merge(ventosa, ventosa1, on='date', how='inner')


# In[55]:


#VELOCIDAD MEDIA.
def velocidadmedia(estadoviento):
    return estadoviento["WS_80mA_mean"].mean()


# In[56]:


#VELOCIDAD MEDIA
velocidadmedia(cuau4)


# In[57]:


#TIEMPO DE PERMANENCIA
def tiempototal(estadoviento):
    return (len(estadoviento)/52560)*365


# In[58]:


#TIEMPO PERMANENCIA
tiempototal(sanfer1)


# In[59]:


#VARIACION PROMEDIO DEL ÁNGULO
def angulovariation(estadoviento):
    loop = (range(len(estadoviento)-1))
    angulo = []
    for i in (loop):
        valor1 = estadoviento["WD_78m_mean"][i+1] 
        valor2 = estadoviento["WD_78m_mean"][i]
        if abs(valor1 - valor2) > 180:
            valor = 360-(abs(valor1-valor2))
            if valor < 10:
                angulo.append(valor)
        else:
            valor = abs(valor1-valor2)
            if valor < 10:
                angulo.append(valor)
    return sum(angulo)/len(angulo)


# In[44]:


#VARIACION PROMEDIO ÁNGULO
angulovariation(cuau4)


# In[78]:


#DENSIDAD DE POTENCIA
def densidadpotencia(estadoviento):
    return (0.5*1.225*(estadoviento["WS_80mA_mean"]**3)).sum()/len(estadoviento)


# In[46]:


#DENSIDAD DE POTENCIA


# In[61]:


#FACTOR DE PLANTA PROMEDIO
from scipy.interpolate import interp1d
    
curva1 = pd.read_csv("Aerogeneradores/VestasV90.txt",delimiter='\t',header=None) #AEROGENERADOR DE 2 MW
curva2 = pd.read_excel("Aerogeneradores/VestasV80.xlsx",header=None) #AEROGENERADOR DE 3 MW

# funcion interpolante para la curva de potencia
f1 = interp1d(curva1[0],curva1[1])
f2 = interp1d(curva2[0],curva2[1])

def potencia1(vel):
    if ((vel>=curva1[0].min()) & (vel<=curva1[0].max())):
        return f1(vel)
    return 0

def potencia2(vel):
    if ((vel>=curva2[0].min()) & (vel<=curva2[0].max())):
        return f2(vel)
    return 0

def factorplanta1(estadoviento):
    return (estadoviento["WS_80mA_mean"].apply(potencia1)*(1/6)).sum()/(2*len(estadoviento)*(1/6))

def factorplanta2(estadoviento):
    return (estadoviento["WS_80mA_mean"].apply(potencia2)*(1/6)).sum()/(3*len(estadoviento)*(1/6))

def factorplantapromedio(estadoviento):
    return ((estadoviento["WS_80mA_mean"].apply(potencia1)*(1/6)).sum()/(2*len(estadoviento)*(1/6)) + 
           (estadoviento["WS_80mA_mean"].apply(potencia2)*(1/6)).sum()/(3*len(estadoviento)*(1/6)))/2


# In[67]:


#FACTOR DE PLANTA PROMEDIO
factorplanta1(ojuelos3)


# In[110]:


estadoviento = merida4


print(f"Densidad de potencia: {densidadpotencia(estadoviento)} \n Variación ángulo: {angulovariation(estadoviento)} \n Tiempo permanencia: {tiempototal(estadoviento)} \n Velocidad media: {velocidadmedia(estadoviento)} \n Factor de planta 1 es:{factorplanta1(estadoviento)} \n Factor 2: {factorplanta2(estadoviento)}")

