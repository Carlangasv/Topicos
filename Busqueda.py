class Problema_busqueda():
    '''-Nodo inicio
     - función de vecinos
     - meta (Funcion booleana)
     - heurística'''

    def nodo_inicio(self):
        'Retorna el nodo inicio'
        raise NotImplementedError('nodo_inicio')

    def es_meta(self, nodo):
        'Retorne verdadero si nodo es meta'
        raise NotImplementedError('es_meta')

    def heuristica(self, n):
        'Retorna la heuristica para el nodo n'
        return 0

    def vecinos(self, nodo):
        'Retorna la lista de los arcos de los vecinos del nodo'
        raise NotImplementedError('vecinos')


class Arco():
    '''- nodo entrante
    -nodo saliente
    -costo (no negativo)'''

    def __init__(self, nodo_saliente, nodo_entrante, costo=1, accion=None):
        assert costo >= 0, (' El costo no puede ser negativo para' +
                            str(nodo_saliente) + ' -> ' + str(nodo_entrante))
        self.nodo_saliente = nodo_saliente
        self.nodo_entrante = nodo_entrante
        self.accion = accion
        self.costo = costo

    def __repr__(self):
        if self.accion:
            return str(self.nodo_saliente) + ' --' + str(self.accion) + ',' + str(self.costo) + '-> ' + str(
                self.nodo_entrante)
        else:
            return str(self.nodo_saliente) + ' --' + str(self.costo) + '-> ' + str(self.nodo_entrante)


class Busqueda_grafo(Problema_busqueda):
    '''- lista o conjunto de nodos
    - lista de arcos
    - nodo inicio
    - lista o conjunto de nodos meta
    - un diccionario que mapee cada nodo en un valor heurístico'''

    def __init__(self, nodos, arcos, inicio=None, metas=set(), hmap={}):
        self.veci = {}
        self.nodos = nodos
        for nodo in nodos:
            self.veci[nodo] = []
        self.arcos = arcos
        for arc in arcos:
            self.veci[arc.nodo_saliente].append(arc)
        self.inicio = inicio
        self.metas = metas
        self.hmap = hmap

    def nodo_inicio(self):
        'Retorna el nodo inicio'
        return self.inicio

    def es_meta(self, nodo):
        'Retorna verdadero si el nodo es una meta'
        return nodo in self.metas

    def heuristica(self, nodo):
        'Retorbna un valor para la heuristica del nodo'
        if nodo in self.hmap:
            return self.hmap[nodo]
        else:
            return 0

    def vecinos(self, nodo):
        return self.veci[nodo]

    def __repr__(self):
        salida = ""
        costo_total = 0
        for arc in self.arcos:
            salida += str(arc)
        return salida


def nodos_vecinos(self, nodo):
    '''Retorna un iterador sobre los vecinos del nodo'''
    return (camino.nodo_entrante for camino in self.veci[nodo])  # El nodo entrante es vecino del nodo saliente


class Camino():
    def __init__(self, inicial, arco=None):
        '''inicial puede ser un nodo'''

        self.inicial = inicial
        self.arco = arco
        if arco is None:
            self.costo = 0
        else:
            self.costo = inicial.costo + arco.costo

    def fin(self):
        '''Retorna el nodo final del camino'''
        if self.arco is None:
            return self.inicial
        else:
            return self.arco.nodo_entrante  # Retorna el nodo entrante del ultimo arco

    def nodo(self):
        '''Retorna los nodos del camino de atras hacia adelante'''
        actual = self
        while actual.arco is not None:
            yield actual.arco.nodo_entrante
            actual = actual.inicial
        yield actual.inicial

    def nodos_iniciales(self):
        '''Retorna todos los nodos antes del nodo final'''
        if self.arco is not None:
            for nd in self.inicial.nodos(): yield nd

    def __repr__(self):
        if self.arco is None:
            return str(self.inicial)
        elif self.arco.accion:
            return str(self.inicial) + "\n --" + str(self.arco.accion) + " -->" + str(self.arco.nodo_entrante)
        else:
            return str(self.inicial) + "\n ---->" + str(self.arco.nodo_entrante)


problema1 = Busqueda_grafo({'a', 'b', 'c', 'd', 'g'},
                           [Arco('a', 'b', 1),
                            Arco('a', 'c', 3),
                            Arco('b', 'c', 1),
                            Arco('b', 'd', 3),
                            Arco('c', 'd', 1),
                            Arco('c', 'g', 3),
                            Arco('d', 'g', 1)],
                           inicio='a', metas={'g'})
##print(problema1)

hmap = {'mail': 26, 'ts': 23, 'o103': 21, 'o109': 24, 'o111': 27, 'o119': 11,
        'o123': 4, 'o125': 6, 'r123': 0, 'b1': 13, 'b2': 15, 'b3': 17, 'b4': 18,
        'c1': 6, 'c2': 10, 'c3': 12, 'storage': 12}


class Visualizable():
    '''Controla la cantidad de detalles por nivel max_nivel_visual'''

    max_nivel_visual = 1

    def visualizar(self, nivel, *args, **nargs):
        if nivel <= self.max_nivel_visual:
            print(*args, **nargs)


class Buscador(Visualizable):
    def __init__(self, problema):
        self.problema = problema
        self.inicializar_frontera()
        self.num_expansion = 0
        self.agregar_a_frontera(Camino(problema.nodo_inicio()))
        super().__init__()

    def inicializar_frontera(self):
        self.frontera = []

    def frontera_vacia(self):
        return self.frontera == []

    def agregar_a_frontera(self, camino):
        self.frontera.append(camino)

    def buscar(self):
        '''Retorna el siguiente camino para el problema del nido inicio al nodo meta'''
        while not self.frontera_vacia():
            camino = self.frontera.pop()
            self.visualizar(2, "Expandiendo ", camino)
            self.num_expansion += 1
            if self.problema.es_meta(camino.fin()):
                self.visualizar(1, self.num_expansion, "Caminos expandidos ", len(self.frontera),
                                "Caminos en la frontera")
                return camino
            else:
                vecis = self.problema.vecinos(camino.fin())
                self.visualizar(2, "Los vecinos son: ", vecis)
                for arco in reversed(list(vecis)):
                    self.agregar_a_frontera(Camino(camino, arco))
                self.visualizar(2, "frontera: ", self.frontera)
            self.visualizar(2, "No hay mas soluciones. Se expandieron un total de ",
                            self.num_expansion, " caminos")

    def buscar_amplitud(self):
        while not self.frontera_vacia():
            camino = self.frontera.pop(0)
            self.visualizar(2, "Expandiendo: ", camino)
            self.num_expansion += 1
            if self.problema.es_meta(camino.fin()):
                self.visualizar(1, self.num_expansion, "Caminos expandidos: ", len(self.frontera),
                                " Caminos en la frontera")
                return camino
            else:
                vecis = self.problema.vecinos(camino.fin())
                self.visualizar(2, "Los vecinos son: ", vecis)
                for arco in list(vecis):
                    self.agregar_a_frontera(Camino(camino, arco))
                self.visualizar(2, "frontera: ", self.frontera)
            self.visualizar(2, "No hay mas soluciones. Se expandieron un total de: ", self.num_expansion, " caminos")




class BuscadorPI(Visualizable):

    def __init__(self, problema):
        self.problema = problema
        super().__init__()
        self.profundidad = False
        self.limite = 0
        self.camino = Camino(problema.nodo_inicio())
        self.meta_intermedia = False

    def buscar_PI(self, camino, b):
        if b > 0:
            for arco in self.problema.vecinos(camino.fin()):
                res = self.buscar_PI(Camino(camino, arco), b - 1)
                if res != None:
                    return res
        elif b <= 0 and self.problema.es_meta(camino.fin()):
            self.profundidad = True
            self.camino = camino
            return self.camino

        elif b <= 0 and self.problema.vecinos(camino.fin()) != None:
            self.visualizar(1, "camino ", camino)
            self.profundidad = True

    def ciclo(self):
        while not self.profundidad:
            self.profundidad = False
            res = self.buscar_PI(self.camino, self.limite)
            if res != None:
                return res
            else:
                self.limite += 1
                self.profundidad = False
                self.ciclo()
        return self.camino


problemaExamen2 = Busqueda_grafo({'3,3,1', '3,2,0', '3,1,0', '2,2,0', '3,2,1', '3,0,0', '3,1,1', '1,1,0', '2,2,1',
                                 '0,2,0', '0,3,1', '0,1,0', '0,2,1', '0,0,0'},
                                [Arco('3,3,1', '3,2,0', 1, "ir"),
                                 Arco('3,3,1', '3,1,0', 1, "ir"),
                                 Arco('3,3,1', '2,2,0', 1, "ir"),
                                 Arco('3,1,0', '3,2,1', 1, "volver"),
                                 Arco('2,2,0', '3,2,1', 1, "volver"),
                                 Arco('3,2,1', '3,0,0', 1, "ir"),
                                 Arco('3,0,0', '3,1,1', 1, "volver"),
                                 Arco('3,1,1', '1,1,0', 1, "ir"),
                                 Arco('1,1,0', '2,2,1', 1, "volver"),
                                 Arco('2,2,1', '0,2,0', 1, "ir"),
                                 Arco('0,2,0', '0,3,1', 1, "volver"),
                                 Arco('0,3,1', '0,1,0', 1, "ir"),
                                 Arco('0,1,0', '0,2,1', 1, "volver"),
                                 Arco('0,2,1', '0,0,0', 1, "ir"),
                                 ],
                                inicio='3,3,1', metas={('0,0,0')})

#print(Buscador(problemaExamen2).buscar())


problema_entregas_aciclico = Busqueda_grafo(
        {'mail', 'ts', 'o103', 'b3', 'b1', 'c2', 'c1', 'c3', 'b2', 'b4', 'o109',
         'o111', 'o119', 'storage', 'o123', 'r123', 'o125'},
        [Arco('ts', 'mail', 6), Arco('o103', 'ts', 8), Arco('o103', 'b3', 4),
         Arco('b3', 'b1', 4), Arco('b1', 'c2', 3), Arco('c2', 'c1', 4),
         Arco('c1', 'c3', 8), Arco('c2', 'c3', 6), Arco('b1', 'b2', 6),
         Arco('b2', 'b4', 3), Arco('b3', 'b4', 7), Arco('b4', 'o109', 7),
         Arco('o103', 'o109', 12), Arco('o109', 'o111', 4), Arco('o109', 'o119', 16),
         Arco('o119', 'storage', 7), Arco('o119', 'o123', 9), Arco('o123', 'r123', 4),
         Arco('o123', 'o125', 4)], inicio='o103', metas={'r123'}, hmap={
            'mail': 26, 'ts': 23, 'o103': 21, 'o109': 24, 'o111': 27, 'o119': 11,
            'o123': 4, 'o125': 6, 'r123': 0, 'b1': 13, 'b2': 15, 'b3': 17,
            'b4': 18, 'c1': 6, 'c2': 10, 'c3': 12, 'storage': 12
        })

problema_ciudades = Busqueda_grafo({
    'Vigo', 'Coruña', 'Oviedo', 'Bilbao', 'Zaragoza', 'Barcelona', 'Gerona', 'Valladolid', 'Madrid', 'Valencia',
    'Badajoz', 'Albacete', 'Murcia', 'Sevilla', 'Cadiz', 'Granada', 'Jaen'},
   [
        Arco('Coruña', 'Vigo', 171),
        Arco('Valladolid', 'Coruña', 455),
        Arco('Valladolid', 'Vigo', 356),
        Arco('Bilbao', 'Valladolid', 280),
        Arco('Madrid', 'Valladolid', 193),
        Arco('Badajoz', 'Madrid', 403),
        Arco('Bilbao', 'Madrid', 395),
        Arco('Albacete', 'Madrid', 251),
        Arco('Jaen', 'Madrid', 335),
        Arco('Zaragoza', 'Madrid', 325),
        Arco('Bilbao', 'Oviedo', 304),
        Arco('Zaragoza', 'Bilbao', 324),
        Arco('Barcelona', 'Zaragoza', 296),
        Arco('Gerona', 'Barcelona', 100),
        Arco('Valencia', 'Barcelona', 349),
        Arco('Albacete', 'Valencia', 191),
        Arco('Murcia', 'Valencia', 241),
        Arco('Albacete', 'Murcia', 150),
        Arco('Granada', 'Murcia', 278),
        Arco('Sevilla', 'Granada', 256),
        Arco('Jaen', 'Granada', 99),
        Arco('Sevilla', 'Jaen', 242),
        Arco('Cadiz', 'Sevilla', 125),
        Arco('Vigo', 'Coruña', 171),
        Arco('Coruña', 'Valladolid', 455),
        Arco('Vigo', 'Valladolid', 356),
        Arco('Valladolid', 'Bilbao', 280),
        Arco('Valladolid', 'Madrid', 193),
        Arco('Madrid', 'Badajoz', 403),
        Arco('Madrid', 'Bilbao', 395),
        Arco('Madrid', 'Albacete', 251),
        Arco('Madrid', 'Jaen', 335),
        Arco('Madrid', 'Zaragoza', 325),
        Arco('Oviedo', 'Bilbao', 304),
        Arco('Bilbao', 'Zaragoza', 324),
        Arco('Zaragoza', 'Barcelona', 296),
        Arco('Barcelona', 'Gerona', 100),
        Arco('Barcelona', 'Valencia', 349),
        Arco('Valencia', 'Albacete', 191),
        Arco('Valencia', 'Murcia', 241),
        Arco('Murcia', 'Albacete', 150),
        Arco('Murcia', 'Granada', 278),
        Arco('Granada', 'Sevilla', 256),
        Arco('Granada', 'Jaen', 99),
        Arco('Jaen', 'Sevilla', 242),
        Arco('Sevilla', 'Cadiz', 125)

        ],
        inicio='Coruña', metas={'Cadiz'})

#print(BuscadorPI(problema_ciudades).ciclo())

##print(BuscadorPI(problema_ciudades).ciclo())



problema_granja= Busqueda_grafo({'lobo, cabra, col, izquierda', 'lobo, col, derecha', 'lobo, col, izquierda',
                                 'col, derecha', 'col, cabra, izquierda', 'cabra, derecha', 'ninguno, derecha',
                                 'cabra, izquierda'},
                                [
                                    Arco('lobo, cabra, col, izquierda', 'lobo, col, derecha'),
                                    Arco('lobo, col, derecha', 'lobo, col, izquierda'),
                                    Arco('lobo, col, izquierda', 'col, derecha'),
                                    Arco('col, derecha', 'col, cabra, izquierda'),
                                    Arco('col, cabra, izquierda', 'cabra, derecha'),
                                    Arco('cabra, derecha', 'cabra, izquierda'),
                                    Arco('cabra, izquierda' , 'ninguno, derecha')
                                ], inicio = 'lobo, cabra, col, izquierda', metas={'ninguno, derecha'})

#print(Buscador(problema_granja).buscar_amplitud())

def busqueda_int(Busqueda,intermedio):
    problem1 = Busqueda_grafo(Busqueda.nodos, Busqueda.arcos, Busqueda.inicio, intermedio)
    problem2 = Busqueda_grafo(Busqueda.nodos, Busqueda.arcos, intermedio, Busqueda.metas)
    print(BuscadorPI(problem1).ciclo())
    print(BuscadorPI(problem2).ciclo())

#Frontera como clase prioritaria
import heapq
class FronteraCP():
    '''Es una frontera que consiste en una cola prioritaria de tripletas (Valor, Indice, Camino), donde valor es lo que se
    desea minizar'''

    def __init__(self):
        self.frontera_index = 0
        self.fronteracp = []

    def vacia(self):
        '''Retorna verdadero si la frontera es vacia'''
        return self.fronteracp == []

    def agregar(self, camino, valor):
        self.frontera_index += 1
        heapq.heappush(self.fronteracp, (valor, self.frontera_index, camino))

    def pop(self):
        '''Retorna y remueve el camino de la frotntera con valor minimo'''
        (_,_,camino) = heapq.heappop(self.fronteracp)
        return camino

    def conteo(self, val):
        '''Retorna el numero de elementos de la frontera con valor = val'''
        return sum(1 for i in self.fronteracp if i[0] == val)

    def __repr__(self):
        return str([n, c, str(p)] for (n, c, p) in self.fronteracp)

    def __len__(self):
        return len(self.fronteracp)

    def __iter__(self):
        '''itera a traves de los caminos en la frontera'''

        for (_,_,camino) in self.fronteracp:
            yield camino

class AEstrella(Buscador):

    def __init__(self, problema):
        super().__init__(problema)

    def inicializar_frontera(self):
        self.frontera = FronteraCP()


    def frontera_vacia(self):
        return self.frontera.vacia()

    def agregar_a_frontera(self, camino):
        valor = camino.costo + self.problema.heuristica(camino.fin())
        self.frontera.agregar(camino, valor)

#print(AEstrella(problema_entregas_aciclico).buscar())

