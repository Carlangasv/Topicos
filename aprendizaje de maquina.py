import math, random
import csv
from Busqueda import Visualizable
import numpy as np

booleana = [False, True]


class Data(Visualizable):
    '''Un conjunto de datos consiste en una lista de datos de entrenamiento (train) y una lista de validación (test)'''
    seed = None  # la idea es que se haga None si queremos un conjunto diferente cada vez

    def __init__(self, train, test=None, prob_test=0.30, indice_target=0, encabezado=None):
        '''Train es una lista de tuplas representando las observaciones o ejemplos de entrenamiento.
        Test es una lista de tuplas representando las observaciones o ejemplos de validación. Si test = None
        se crea un conjunto de validacióln seleccionando muestras con probabilidad prob_test.
        indice_target es el índice de la característica objetivo. Si  es mayor que el número de propiedades, quiere
        decir que no hay característica objetivo.
        encabezado es una lista de los nombres de las características'''

        if test is None:
            train, test = particionamiento_data(train, prob_test, seed=self.seed)
        self.train = train
        self.test = test
        self.visualizar(2, "Tuplas leidas. \nConjunto de entrenamiento ", len(train), " ejemplos. Número de columnas: ",
                        {len(e) for e in train}, "\nConjunto de validación ", len(test),
                        " ejemplos. Número de columnas: ",
                        {len(e) for e in test})
        self.prob_test = prob_test
        self.numero_propiedades = len(self.train[0])
        if indice_target < 0:
            indice_target = self.numero_propiedades + indice_target
        self.indice_target = indice_target
        self.encabezado = encabezado
        self.crear_caracteristicas()
        self.visualizar(2, "Hay ", len(self.caracteristicas_entrada), " caracteristicas de entrada")

    def crear_caracteristicas(self):
        '''genera las características de entrada y la característica objetivo. Aquí se asume que todas las características
        tienen rango {0,1}. Si tienen rangos diferentes se debe sobre-escribir'''
        self.caracteristicas_entrada = []
        for i in range(self.numero_propiedades):
            def caracteristica(e, index=i):
                return e[index]

            if self.encabezado:
                caracteristica.__doc__ = self.encabezado[i]
            else:
                caracteristica.__doc__ = "e[" + str(i) + "]"
            caracteristica.rango = [0, 1]
            if i == self.indice_target:
                self.target = caracteristica
            else:
                self.caracteristicas_entrada.append(caracteristica)

    criterios_evaluacion = ["suma_cuadrados", "suma_absoluta", "logloss"]

    def evaluar_dataset(self, data, predictor, criterio_evaluacion):
        '''evalúa el predictor sobre los datos de acuerdo a algún criterio de evaluación. predictor es una función
        que toma un ejemplo y retorna una predicción sobre las características.'''
        assert criterio_evaluacion in self.criterios_evaluacion, str(criterio_evaluacion)

        if data:
            try:
                error = sum(error_ejemplo(predictor(ejemplo), self.target(ejemplo), criterio_evaluacion) for ejemplo in
                            data) / len(data)
            except ValueError:
                return float("inf")
            return error


def particionamiento_data(data, prob_test=0.30, seed=None):
    '''Particiona los datos en conjintos de entrenamiento y validacion, donde prob_test es la probabilidad de que un
        ejemplo pertenezca al grupo de validacion. Una alternativa es usar random.sample() para garantizar que tenga
        una proporcion de prob_test en los datos de validacion'''
    train = []
    test = []
    if seed:
        random.seed(seed)
    #perm = np.random.permutation(len(data))
    #test = list(data[i] for i in perm[:int(prob_test * len(data))])
    #train = list(data[i] for i in perm[int(prob_test * len(data)):])
    for ejemplo in data:
        if random.random() < prob_test:
            test.append(ejemplo)
        else:
            train.append(ejemplo)

    return train, test


datos = Data([(0.2, 0.3, 1), (0.4, 0.7, 2), (0.2, 0.4, 0.6), (0.2, 0.4, 3)], indice_target=-1)

#print(datos.__dict__)


def error_ejemplo(prediccion, real, criterio_evaluacion):
    '''retorna el error de la prediccion actual dado el valor real de acuerdo a criterio_evaluacion'''

    if criterio_evaluacion == "suma_cuadrados":
        return (prediccion - real) ** 2
    elif criterio_evaluacion == "suma_absoluta":
        return abs(prediccion - real)
    elif criterio_evaluacion == "logloss":
        assert real in [0, 1], "real = " + str(real)
        if real == 0:
            return -math.log2(1 - prediccion)
        else:
            return -math.log2(prediccion)
    else:
        raise RuntimeError(str(criterio_evaluacion), " no es un criterio de evaluación")


class Data_archivo(Data):

    def __init__(self, archivo, separador=',', num_train=None, prob_test=0.3, tiene_encabezado=False,
                 indice_target=0, caracteristicas_booleanas=True, categoricas=[], incluir=None):
        '''crea un dataset de un archivo.
        separador es el caracter que separa los atributos.
        num_train es un número n que especifica si las primeras n tuplas son entrenamiento o no.
        tiene_encabezado indica si la primera línea del archivo es un encabezado.
        caracteristicas_booleanas indica si queremos crear caracteristicas booleanas. Si es falso se usan las características
        originales.
        categoricas es una lista de características que deben ser tratadas como categóricas.
        incluir es unal ista de índices de columnas para incluir.'''

        self.caracteristicas_booleanas = caracteristicas_booleanas
        with open(archivo, 'r', newline='') as archivo_csv:
            data_all = (linea.strip().split(separador) for linea in archivo_csv)
            if incluir is not None:
                data_all = ([v for (i, v) in enumerate(linea) if i in incluir] for linea in data_all)
            if tiene_encabezado:
                encabezado = next(data_all)
            else:
                encabezado = None
            data_tuplas = (hacer_numeros(d) for d in data_all if len(d) > 1)  # queda pendiente
            if num_train is not None:
                train = []
                for i in range(num_train):
                    train.append(next(data_tuplas))
                test = list(data_tuplas)
                Data.__init__(self, train, test=test, indice_target=indice_target, encabezado=encabezado)
            else:
                Data.__init__(self, data_tuplas, prob_test=prob_test, indice_target=indice_target,
                              encabezado=encabezado)

    def __str__(self):
        return ("Data: " + str(len(self.train)) + " ejemplos de entrenamiento, " + str(self.test) + " ejemplos de test")

    '''para la creación de características Booleanas consideraremos tres casos:

    1. Cuando el rango solamente tiene dos valores. En ese caso uno se asume como Verdadero.
    2. Cuando todos los valores son numéricos y están ordenados. Se construyen las características Booleanas por intervalos,
    i.e., la característica es e[ind] < corte. Se elije corte si sobrepasar max_cortes.
    3. Cuando los valores no son todos numéricos, se asumen no ordenados y se crea una función indicadora para cada valor.
    '''

    def crear_caracteristica(self, max_cortes=8):
        '''crea características Booleanas a partir de las características de entrada.'''

        rangos = [set() for i in range(self.num_propiedades)]
        for ejemplo in self.train:
            for ind, val in enumerate(ejemplo):
                rangos[ind].add(val)
        if self.indice_target <= self.num_propiedades:
            def target(e, indice=self.indice_target):
                return e[indice]

            if self.encabezado:
                target.__doc__ = self.encabezado[self.indice_target]
            else:
                target.__doc__ = "e[" + str(self.indice_target) + "]"
            target.rango = rangos[self.indice_target]
        if self.caracteristicas_booleanas:
            self.caracteristicas_entrada = []
            for ind, rango in enumerate(rangos):
                if len(rango) == 2:  # dos valores, uno de los dos se asume como verdadero
                    valor_verdad = list(rango)[1]  # asigna uno como verdadero

                    def caracteristica(e, i=ind, tv=valor_verdad):
                        return e[i] == tv

                    if self.encabezado:
                        caracteristica.__doc__ = self.encabezado[ind] + " == " + str(valor_verdad)
                    else:
                        caracteristica.__doc__ = "e[" + str(ind) + "] == " + str(valor_verdad)
                    caracteristica.rango = booleana
                    self.caracteristicas_entrada.append(caracteristica)
                elif all(isinstance(val, (int, float)) for val in
                         rango):  # todos los valores en el rango son numéricos y ordenados
                    rango_ordenado = sorted(rango)
                    numero_cortes = min(max_cortes, len(rango))
                    posiciones_cortes = [len(rango) * i // numero_cortes for i in range(1, numero_cortes)]
                    for corte in posiciones_cortes:
                        corte_en = rango_ordenado[corte]

                        def caracteristica(e, ind_=ind, corte_en=corte_en):
                            return e[ind_] < corte_en

                        if self.encabezado:
                            caracteristica.__doc__ = self.encabezado[ind] + " < " + str(corte_en)
                        else:
                            caracteristica.__doc__ = "e[" + str(ind) + "] < " + str(corte_en)
                        caracteristica.rango = booleana
                        self.caracteristicas_entrada.append(caracteristica)
                else:  # se crea una variable indicadora para cada valor
                    for val in rango:
                        def caracteristica(e, ind_=ind, val_=val):
                            return e[ind_] == val_

                        if self.encabezado:
                            caracteristica.__doc__ = self.encabezado[ind] + " == " + str(val)
                        else:
                            caracteristica.__doc__ = "e[" + str[ind] + "] == " + str(val)
        else:  # si caracteristicas_booleanas == False
            self.caracteristicas_entrada = []
            for i in range(self.num_propiedades):
                def caracteristica(e, index=i):
                    return e[index]

                if self.encabezado:
                    caracteristica.__doc__ = self.encabezado[i]
                else:
                    caracteristica.__doc__ = "e[" + str(i) + "]"
                if i == self.indice_target:
                    self.target = caracteristica
                else:
                    self.caracteristicas_entrada.append(caracteristica)


def hacer_numeros(str_list):
    '''hace los elementos de una lista  de strings numéricos si es posible. De otra forma se remueven los espacios iniciales
    y finales'''
    res = []
    for e in str_list:
        try:
            res.append(int(e))
        except ValueError:
            try:
                res.append(float(e))
            except ValueError:
                res.append(e.strip())
    return res
data = Data_archivo('archivo_prueba.csv', indice_target = -1, tiene_encabezado= True)

#print(data.__dict__)


class Aprendiz(Visualizable):

    def __init__(self, dataset):
        raise NotImplementedError("Aprendiz.__init__")

    def aprender(self):
        '''retorna  un predictor, una función de una tupla o un valor para el target'''
        raise NotImplementedError("aprender")


import math, random

selecciones = ["mediana", "media"]


def prediccion_puntual(target, datos_entrenamiento, seleccion="media"):
    '''hace una predicción puntual para un conjunto de entrenamiento. target indica la caracteristica objetivo.
    datos_entrenamiento indica los datos para usar en el entrenamiento, casi siempre es un subconjunto de train.
    selección especifica qué estádistica se usará como evaluación.'''

    assert len(datos_entrenamiento) > 0, "Datos de entrenamiento insuficientes"
    if seleccion == 'mediana':
        conteo, total = conteo_target(target, datos_entrenamiento)
        mitad = total / 2
        acumulador = 0
        for val, num in sorted(conteo.items()):
            acumulador += num
            if acumulador > mitad:
                break
    elif seleccion == "media":
        val = media((target(e) for e in datos_entrenamiento))
    #     elif seleccion == "moda":
    #         raise NotImplementedError("moda")
    else:
        raise RuntimeError("Selección no válida")
    fun = lambda x: val
    fun.__doc__ = str(val)
    return fun


def conteo_target(target, sub_data):
    '''retorna un diccionario valor:conteo del número de veces que target tiene su valor en sub_data, y el número de ejemplos'''
    conteo = {val: 0 for val in target.rango}
    total = 0
    for instancia in sub_data:
        total += 1
        conteo[target(instancia)] += 1
    return conteo, total


def media(enum, conteo=0, sum=0):
    '''retorna la media de la enumeración enum.'''
    for e in enum:
        conteo += 1
        sum += e
    return sum / conteo

class Data_random(Data):

    def __init__(self, prob, train_size, test_size=100):
        train = [[1] if random.random() < prob else [0] for i in range(train_size)]
        test = [[1] if random.random() < prob else [0] for i in range(test_size)]
        Data.__init__(self, train, test, indice_target=0)


def test_puntual():
    num_muestras = 10
    test_size = 100
    for train_size in [1,2,3,1000]:
        error_total ={(select, crit):0 for select in selecciones for crit in Data.criterios_evaluacion}
        for muestra in range(num_muestras): #promedio sobre el número de muestras
            p = random.random()
            data = Data_random(p, train_size, test_size)
            for select in selecciones:
                prediccion = prediccion_puntual(data.target, data.train, seleccion = select)
                for ecrit in Data.criterios_evaluacion:
                    test_error = data.evaluar_dataset(data.test, prediccion, ecrit)
                    error_total[(select, ecrit)] += test_error
        print("Para un conjunto de entrenamiento de tamaño ", train_size, ":")
        for ecrit in Data.criterios_evaluacion:
            print(" Evaluando de acuerdo con ", ecrit, ":")
            for select in selecciones:
                print("     El error promedio de ", select, " es ", error_total[(select, ecrit)]/num_muestras)


#test_puntual()


''' El algoritmo de árboles de desición hace divisiones binarias, y asume que todas las características de entrada son 
binarias de los ejemplos. Deja de hacer las divisiones cuando no hay más características de entrada (no podemos hacer
más nodos), cuando el número de ejemplos es menor que el especificado o cuando todos los ejemplos de acuerdo con la
característica objetivo'''

class Aprendiz_DT(Aprendiz):

    def __init__(self, dataset, para_optimizar = "suma_cuadrados", seleccion_hojas = "media", train = None,
                 numero_min_ejemplos = 10):
        self.dataset = dataset
        self.target = dataset.target
        self.para_optimizar = para_optimizar
        self.seleccion_hojas = seleccion_hojas
        self.numero_min_ejemplos = numero_min_ejemplos
        if train is None: #Se una para validación cruzada
            self.train = self.dataset.train
        else:
            self.train = train
    def aprender(self):
        return self.aprender_arbol(self.dataset.caracteristicas_entrada, self.train)

    def aprender_arbol(self, caracteristicas_entrada, data_subset):
        '''retorna un arbol de desición para caracteristicas_entrada siendo un conjunto de posibles condiciones y
        data_subset un conjunto para construir el árbol'''
        if(caracteristicas_entrada and len(data_subset) >= self.numero_min_ejemplos):
            primer_valor_target = self.target(data_subset[0])
            unanimidad = all(self.target(inst) == primer_valor_target for inst in data_subset)
            if not unanimidad:
                dividir, particionamiento = self.seleccion_dividir(caracteristicas_entrada, data_subset)
                if dividir: #Si la división fue exitosa
                    ejemplos_falsos, ejemplos_verdaderos = particionamiento
                    caracteristicas_remanentes = [fe for fe in caracteristicas_entrada if fe != dividir]
                    arbol_verdadero = self.aprender_arbol(caracteristicas_remanentes, ejemplos_verdaderos)
                    arbol_falso = self.aprender_arbol(caracteristicas_remanentes, ejemplos_falsos)
                    fun = lambda e: arbol_verdadero(e) if dividir(e) else arbol_falso(e)
                    fun.__doc__ = ("if " + dividir.__doc__ + " then (" + arbol_verdadero.__doc__
                                   + ") else (" + arbol_falso.__doc__ + ")")
                    return fun
        return prediccion_puntual(self.target, data_subset, seleccion = self.seleccion_hojas)

    def seleccion_dividir(self, caracteristicas_entrada, data_subset):
        ''' encuentra le mejor condición para hacer la nueva división'''

        mejor_caracteristica = None
        mejor_error = error_entrenamiento(self.dataset, data_subset, self.para_optimizar)
        mejor_particion = None
        for caracteristica in caracteristicas_entrada:
            ejemplos_falsos, ejemplos_verdaderos = particion(data_subset, caracteristica)
            if ejemplos_verdaderos and ejemplos_falsos:
                error = (error_entrenamiento(self.dataset, ejemplos_falsos, self.para_optimizar) +
                         error_entrenamiento(self.dataset, ejemplos_verdaderos, self.para_optimizar))
                if error <= mejor_error:
                    mejor_caracteristica = caracteristica
                    mejor_error = error
                    mejor_particion = ejemplos_falsos, ejemplos_verdaderos
        return mejor_caracteristica, mejor_particion

def particion(data_subset, caracteristica):
    ejemplos_verdaderos = []
    ejemplos_falsos = []
    for ejemplo in data_subset:
        if caracteristica(ejemplo):
            ejemplos_verdaderos.append(ejemplo)
        else:
            ejemplos_falsos.append(ejemplo)
    return ejemplos_falsos, ejemplos_verdaderos

def error_entrenamiento(dataset, data_subset, para_optimizar):
    '''retorna el error de entrenamiento para dataset con la métrica para optimizar'''
    diccionario_seleccion = {'suma_cuadrados': 'media', 'suma_absoluta' : 'mediana', 'logloss' : 'media'}
    seleccion = diccionario_seleccion[para_optimizar]
    predictor = prediccion_puntual(dataset.target, data_subset, seleccion = seleccion)
    error = sum(error_ejemplo(predictor(ejemplo), dataset.target(ejemplo), para_optimizar) for ejemplo in data_subset)
    return error

def test(data):
    for criterio in Data.criterios_evaluacion:
        for hoja in selecciones:
            arbol = Aprendiz_DT(data, para_optimizar = criterio, seleccion_hojas = hoja).aprender()
            print("Para ", criterio, " usando ", hoja, " para la etiqueta, el árbol construido es: ", arbol.__doc__)
            if data.test:
                for ecriterio in Data.criterios_evaluacion:
                    error_test = data.evaluar_dataset(data.test, arbol, ecriterio)
                    print("El error promedio para ", ecriterio, " usando ", hoja, " para la etiqueta,es: ", error_test)

#test(data = Data_archivo('mail_data.csv', indice_target=-1,tiene_encabezado=True))

# Usando SKLEARN

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report # Métricas de desempeño
from sklearn.model_selection import train_test_split # División entrenamiento/validación
from sklearn.tree import DecisionTreeClassifier # arboles de desición
import numpy as np

# Cargamos los datos

input_file = 'datos_arboles.txt'
datos = np.loadtxt(input_file, delimiter = ',')
X, y = datos[:,:-1], datos[:,-1]

# Solo para visualizar los datos
clase0 = np.array(X[y==0])
clase1 = np.array(X[y==1])
plt.figure()
plt.scatter(clase0[:,0], clase0[:,1], facecolors = 'black', edgecolor='black',linewidth = 1, marker = '*')
plt.scatter(clase1[:,0], clase1[:,1], facecolors = 'white', edgecolor= 'black',linewidth = 1, marker = 'o')
plt.title("Datos de entrada")
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

parametros = {'random_state' : 0, 'max_depth' : 4}
clasificador = DecisionTreeClassifier(**parametros)
clasificador.fit(X_train, y_train)

DecisionTreeClassifier(max_depth=4, random_state=0)
y_pred = clasificador.predict(X_test)
acc =  100.0 * (y_test == y_pred).sum()/X_test.shape[0]
print("El acierto de clasificación es del ", acc,"%")

nombre_clases = ['Clase 0', 'Clase 1']
print('\n' + '#'*40)
print('\n desempeño de clasificador sobre el conjunto de validación \n')
print(classification_report(y_test, y_pred, target_names = nombre_clases))
print('\n' + '#'*40)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize= (10,8))
sns.heatmap(cm, annot = True, fmt = 'd')
#plt.show()

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

clase0 = np.array(X[y==0])
clase1 = np.array(X[y==1])
clase2 = np.array(X[y==2])
plt.figure()
plt.scatter(clase0[:,0], clase0[:,1], facecolors = 'yellow', edgecolor='black',linewidth = 1, marker = '*')
plt.scatter(clase1[:,0], clase1[:,1], facecolors = 'red', edgecolor= 'black',linewidth = 1, marker = 'o')
plt.scatter(clase2[:,0], clase2[:,1], facecolors = 'blue', edgecolor= 'black',linewidth = 1, marker = 's')
plt.title("Caracteristica 0 vs caracteristica 1 ")
plt.show()


X = iris.data
y = iris.target

clase0 = np.array(X[y==0])
clase1 = np.array(X[y==1])
clase2 = np.array(X[y==2])
plt.figure()
plt.scatter(clase0[:,2], clase0[:,3], facecolors = 'yellow', edgecolor='black',linewidth = 1, marker = '*')
plt.scatter(clase1[:,2], clase1[:,3], facecolors = 'red', edgecolor= 'black',linewidth = 1, marker = 'o')
plt.scatter(clase2[:,2], clase2[:,3], facecolors = 'blue', edgecolor= 'black',linewidth = 1, marker = 's')
plt.title("Caracteristica 2 vs caracteristica 3 ")
plt.show()

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_p = sc.transform(X_train)
X_test_p = sc.transform(X_test)

clasificador = LogisticRegression(C = 1000.0, random_state = 0)
clasificador.fit(X_train_p, y_train)
y_pred = clasificador.predict(X_test_p)
nombre_clases = ['Clase 0', 'Clase 1', 'Clase 2']
print('\n'+ '#'*60)
print('\n desempeño del clasificador sobre el conjunto de validación \n')
print(classification_report(y_test, y_pred, target_names = nombre_clases))
print('\n' + '#'*60)