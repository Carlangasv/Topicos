def union_diccionarios(d1,d2):
    '''retorna un diccionario que contiene las claves de d1 y d2.
    El valor de cada clave que está en d2 es el valor de d2, de otra forma
    es el valor de d1'''
    d = dict(d1) #copia d1
    d.update(d2)
    return d