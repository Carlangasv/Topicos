class Restriccion():
    '''Una restricción consiste:
    -alcance: tupla de var
    -condición: función que debe ser aplicada a la tupla de valores
    Debe ser booleana'''

    def __init__(self, alcance, condicion):
        self.alcance = alcance
        self.condicion = condicion

    def __repr__(self):
        return str(self.alcance) + self.condicion.__name__  # + str(self.condicion)

    def holds(self, asingacion):
        '''retorna el valor de la restricción evaluada en la asignación'''

        return self.condicion(*tuple(asingacion[v] for v in self.alcance))


from operator import lt, ne, eq, gt


def noes(valor):
    noes_valor = lambda x: x != valor
    noes_valor.__name__ = str(valor) + "!="
    return noes_valor


def es(valor):
    es_valor = lambda x: x == valor
    es_valor.__name__ = str(valor) + "=="
    return es_valor


def adya(valor):
    es_adya = lambda x: x + 1 == valor
    es_adya.__name__ = str(valor) + " adyacente"
    return es_adya


def eqChar(pos1, pos2):
    value = lambda x, y: False if len(x) <= pos1 or len(y) <= pos2 else x[pos1] == y[pos2]
    value.__name__ = str(pos1) + "==" + str(pos2)
    return value


def eqLen(size):
    value = lambda x: len(x) == size
    value.__name__ = str(size) + "=="
    return value


class CSP:
    '''Un CSP requiere:
    - dominios : un diccionario que mapea las variables al conjunto de posibles valores.
    dominio[var] es el dominio de la variable var.
     - restricciones: conjunto o lista de objetos de clase Restricción'''

    def __init__(self, dominios, restricciones):
        self.variables = set(dominios)
        self.dominios = dominios
        self.restricciones = restricciones
        self.variables_a_restricciones = {var: set() for var in self.variables}
        for con in restricciones:
            for var in con.alcance:
                self.variables_a_restricciones[var].add(con)

    def __str__(self):
        '''representación del CSP'''
        return str(self.dominios)

    def __repr__(self):
        '''representación más detallada del CSP'''
        return "CSP( " + str(self.dominios) + " , " + str([str(c) for c in self.restricciones]) + " )"

    def consistencia(self, asignacion):
        '''asignación es variable: valor diccionario
        retorna Verdadero si todas las restricciones que pueden ser evaluadas, son evaluadas
        verdaderas dada una asignación'''
        return all(con.holds(asignacion) for con in self.restricciones if all(v in asignacion for v in con.alcance))


'''Tenemos un CSP con variables X, Y, Z, cada una con dominio {1,2,3}. Las restricciones son X<Y, Y>Z'''

csp0 = CSP({'X': {1, 2, 3}, 'Y': {1, 2, 3}, 'Z': {1, 2, 3}},
           [Restriccion(('X', 'Y'), lt),
            Restriccion(('Y', 'Z'), gt)])

'''Tenemos un CSP con variables A,B,C, cada una con dominio {1,2,3,4}. Las restricciones son:
A<B, B !=2, B<C'''

csp1 = CSP({'A': {1, 2, 3, 4}, 'B': {1, 2, 3, 4}, 'C': {1, 2, 3, 4}},
           [Restriccion(('A', 'B'), lt),
            Restriccion(('B',), noes(2)),
            Restriccion(('B', 'C'), lt)])

csp2 = CSP({'A': {1, 2, 3, 4}, 'B': {1, 2, 3, 4}, 'C': {1, 2, 3, 4}, 'D': {1, 2, 3, 4}, 'E': {1, 2, 3, 4}},
           [
               Restriccion(('B',), noes(3)),
               Restriccion(('C',), noes(2)),
               Restriccion(('A', 'B'), ne),
               Restriccion(('B', 'C'), ne),
               Restriccion(('C', 'D'), lt),
               Restriccion(('A', 'D'), eq),
               Restriccion(('E', 'A'), lt),
               Restriccion(('E', 'B'), lt),
               Restriccion(('E', 'C'), lt),
               Restriccion(('E', 'D'), lt),
               Restriccion(('B', 'D'), ne)
           ])

'''Tenemos un CSP con 5 variables, de la 'A' a la 'E'. Cada una con dominio {1,2,3,4,5}. Las restricciones son:
A y B deben ser adyacentes, B y C deben ser adyacentes, C y D deben ser adyantes, D y E deben ser adyacentes,
A diferente de C, B diferente de D y C diferente de E.
'''

csp3 = CSP(
    {'A': {1, 2, 3, 4, 5}, 'B': {1, 2, 3, 4, 5}, 'C': {1, 2, 3, 4, 5}, 'D': {1, 2, 3, 4, 5}, 'E': {1, 2, 3, 4, 5}},
    [
        Restriccion(('A',), adya('B')),
        Restriccion(('B',), adya('C')),
        Restriccion(('C',), adya('D')),
        Restriccion(('D',), adya('E')),
        Restriccion(('A', 'C'), ne),
        Restriccion(('B', 'D'), ne),
        Restriccion(('C', 'E'), ne)
    ])

csp4 = CSP({'Bell': {'8-10s', '10-12s', '14-16s', '16-18s'},
            'Maria clara': {'8-10s', '10-12s', '8-10d', '10-12d'},
            'Luisa': {'8-10s', '10-12s', '14-16s', '16-18s', '8-10d', '10-12d', '14-16d', '16-18d'},

            'Gloria': {'10-12s', '14-16s', '10-12d', '14-16d'},
            'Jesus': {'8-10s', '10-12s', '8-10d', '10-12d'},
            'Jaime': {'8-10s', '10-12s', '14-16s', '16-18s'},
            'Liliana': {'8-10s', '10-12s', '14-16s', '16-18s', '8-10d', '10-12d', '14-16d', '16-18d'},
            'Lina': {'8-10s', '10-12s', '8-10d', '10-12d'}},
           [
               Restriccion(('Bell', 'Maria clara'), ne),
               Restriccion(('Bell', 'Luisa'), ne),
               Restriccion(('Bell', 'Gloria'), ne),
               Restriccion(('Bell', 'Jesus'), ne),
               Restriccion(('Bell', 'Jaime'), ne),
               Restriccion(('Bell', 'Liliana'), ne),
               Restriccion(('Bell', 'Lina'), ne),
               Restriccion(('Maria clara', 'Luisa'), ne),
               Restriccion(('Maria clara', 'Gloria'), ne),
               Restriccion(('Maria clara', 'Jesus'), ne),
               Restriccion(('Maria clara', 'Jaime'), ne),
               Restriccion(('Maria clara', 'Liliana'), ne),
               Restriccion(('Maria clara', 'Lina'), ne),
               Restriccion(('Luisa', 'Gloria'), ne),
               Restriccion(('Luisa', 'Jesus'), ne),
               Restriccion(('Luisa', 'Jaime'), ne),
               Restriccion(('Luisa', 'Liliana'), ne),
               Restriccion(('Luisa', 'Lina'), ne),
               Restriccion(('Gloria', 'Jesus'), ne),
               Restriccion(('Gloria', 'Jaime'), ne),
               Restriccion(('Gloria', 'Liliana'), ne),
               Restriccion(('Gloria', 'Lina'), ne),
               Restriccion(('Jesus', 'Jaime'), ne),
               Restriccion(('Jesus', 'Liliana'), ne),
               Restriccion(('Jesus', 'Lina'), ne),
               Restriccion(('Jaime', 'Liliana'), ne),
               Restriccion(('Jaime', 'Lina'), ne),
               Restriccion(('Liliana', 'Lina'), ne),

           ])


# print(csp4)

def test(solucionador_csp, csp, soluciones=[{'Lina': '8-10s', 'Maria clara': '10-12s',
                                             'Bell': '14-16s', 'Jaime': '16-18s', 'Liliana': '8-10d', 'Jesus': '10-12s',
                                             'Bell': '14-16d', 'Luisa': '16-18d'}]):
    print('Evaluando el csp con ', solucionador_csp.__doc__)
    sol = solucionador_csp(csp)
    print("Se encontró una solución: ", sol)
    assert sol in soluciones, "La solución no es correcta para " + str(csp)
    print('Pasó el test unitario')


from Busqueda import Arco, Problema_busqueda
from utilidades import union_diccionarios


class Busqueda_CSP(Problema_busqueda):
    '''Un nodo es variable: valor diccionario'''

    def __init__(self, csp, orden_variable=None):
        self.csp = csp
        if orden_variable:
            assert set(orden_variable) == set(csp.variables)
            assert len(orden_variable) == len(csp.variables)
            self.variables = orden_variable
        else:
            self.variables = list(csp.variables)

    def es_meta(self, nodo):
        return len(nodo) == len(self.csp.variables)

    def nodo_inicio(self):
        return {}

    def vecinos(self, nodo):
        ''''este método usa el hecho de que la longitud del nodo, que es el número de variables
        asignadas, es el índice de la siguiente variable para hacer la división del grafo.
        Note que no se necesita revisar si hay más variables para hacer la división,
        dado que todos los nodos son consistentes por construcción y que no se tienen
        más variables  si hay una solución'''

        var = self.variables[len(nodo)]  # la siguiente variable
        res = []
        for val in self.csp.dominios[var]:
            nuevo = union_diccionarios(nodo, {var: val})
            if self.csp.consistencia(nuevo):
                res.append(Arco(nodo, nuevo))
        return res


from Busqueda import Buscador, BuscadorPI

# print(Buscador(Busqueda_CSP(csp4)).buscar())

from Busqueda import Visualizable


class Solucionador_consistencia(Visualizable):
    '''Soluciona un CSP con consistencia de arco y división de dominio'''

    def __init__(self, csp, **kwargs):
        self.csp = csp
        super().__init__(**kwargs)

    def hacer_arcos_consistentes(self, dominios_originales=None, para_revisar=None):
        '''Hace que el CSP sea de arcos consistentes. dominios_originales son los dominios originales del CSP
        para_revisar es un conjunto de pares (variable, restriccion)'''

        if dominios_originales is None:
            dominios_originales = self.csp.dominios
        if para_revisar is None:
            para_revisar = {(variable, restriccion) for restriccion in self.csp.restricciones for variable in
                            restriccion.alcance}
        else:
            para_revisar = para_revisar.copy()
        dominios = dominios_originales.copy()
        self.visualizar(2, 'Realizando AC con dominios: ', dominios)
        while para_revisar:
            variable, restriccion = self.seleccionar_arco(para_revisar)
            self.visualizar(2, 'Procesando arco ( ', variable, ',', restriccion, ')')
            otras_variables = [otras for otras in restriccion.alcance if otras != variable]
            nuevo_dominio = {valor for valor in dominios[variable] if
                             self.se_mantiene(dominios, restriccion, {variable: valor},
                                              otras_variables)}
            if nuevo_dominio != dominios[variable]:
                self.visualizar(2, 'El arco : (', variable, ',', restriccion, ') es inconsistente')
                self.visualizar(2, 'Hay poda de dominio ', 'dominio(', variable, ')=', nuevo_dominio, ' debido a ',
                                restriccion)
                dominios[variable] = nuevo_dominio
                agregar_para_revisar = self.nuevo_para_revisar(variable, restriccion) - para_revisar
                para_revisar |= agregar_para_revisar  # unión de conjuntos
                self.visualizar(2, ' agregando ', agregar_para_revisar if agregar_para_revisar else "nada",
                                ' para revisar')
            self.visualizar(2, "El arco (", variable, ',', restriccion, ") ahora es consistente")
        self.visualizar(2, 'Consistencia de arcos terminada. Dominios reducidos ', dominios)
        return dominios

    def nuevo_para_revisar(self, variable, restriccion):
        '''retorna nuevos elementos para agregar a para_revisar después de asignar variable en restriccion'''
        return {(nvariable, nrestriccion) for nrestriccion in self.csp.variables_a_restricciones[variable]
                if nrestriccion != restriccion
                for nvariable in nrestriccion.alcance
                if nvariable != variable}

    def seleccionar_arco(self, para_revisar):
        '''selecciona el arco del conjunto para_revisar, donde un arco es un par (variable, restriccion).
        El arco seleccionado debe ser removido del conjunto para_revisar'''
        return para_revisar.pop()

    def se_mantiene(self, dominios, restriccion, asignacion, otras_variables, ind=0):
        '''retorna Verdadero si restriccion se mantiene para una asignacion que se extiene con las variables en otras_variables[ind:]
        asignacion es un diccionario.'''
        if ind == len(otras_variables):
            return restriccion.holds(asignacion)
        else:
            variable = otras_variables[ind]
            for valor in dominios[variable]:
                asignacion[variable] = valor
                if self.se_mantiene(dominios, restriccion, asignacion, otras_variables, ind + 1):
                    return True
            return False

    def una_solucion(self, dominios=None, para_revisar=None):

        '''retorna una solución al CSP actual o retorna Falso si no hay soluciones. para_revisar es un conjunto de arcos que se deben revisar'''
        if dominios is None:
            dominios = self.csp.dominios
        nuevos_dominios = self.hacer_arcos_consistentes(dominios, para_revisar)
        if any(len(nuevos_dominios[variable]) == 0 for variable in
               dominios):  # alguna variable no tiene elementos en el dominio
            return False
        elif all(len(nuevos_dominios[variable]) == 1 for variable in dominios):  # ya se tiene una solución
            self.visualizar(2, "solución: ",
                            {variable: seleccionar(nuevos_dominios[variable]) for variable in nuevos_dominios})
            return {variable: seleccionar(nuevos_dominios[variable]) for variable in dominios}
        else:
            variable = self.seleccionar_variable(x for x in self.csp.variables if len(nuevos_dominios[x]) > 1)
            if variable:
                dominio1, dominio2 = particion_dominio(nuevos_dominios[variable])
                self.visualizar(2, "...particionando ", variable, " en ", dominio1, " y ", dominio2)
                nuevos_dominios1 = copiar_con_asignacion(nuevos_dominios, variable, dominio1)
                nuevos_dominios2 = copiar_con_asignacion(nuevos_dominios, variable, dominio2)
                para_revisar = self.nuevo_para_revisar(variable, None)
                self.visualizar(2, ' agregando ', para_revisar if para_revisar else " nada ", " para revisar ")
                return self.una_solucion(nuevos_dominios1, para_revisar) or self.una_solucion(nuevos_dominios2, para_revisar)


    def varias_soluciones(self, dominios=None, para_revisar=None):
        soluciones = []
        '''retorna una solución al CSP actual o retorna Falso si no hay soluciones. para_revisar es un conjunto de arcos que se deben revisar'''
        if dominios is None:
            dominios = self.csp.dominios
        nuevos_dominios = self.hacer_arcos_consistentes(dominios, para_revisar)
        if any(len(nuevos_dominios[variable]) == 0 for variable in
               dominios):  # alguna variable no tiene elementos en el dominio
            return False
        elif all(len(nuevos_dominios[variable]) == 1 for variable in dominios):  # ya se tiene una solución
            self.visualizar(2, "solución: ",
                            {variable: seleccionar(nuevos_dominios[variable]) for variable in nuevos_dominios})
            return {variable: seleccionar(nuevos_dominios[variable]) for variable in dominios}
        else:
            variable = self.seleccionar_variable(x for x in self.csp.variables if len(nuevos_dominios[x]) > 1)
            if variable:
                dominio1, dominio2 = particion_dominio(nuevos_dominios[variable])
                self.visualizar(2, "...particionando ", variable, " en ", dominio1, " y ", dominio2)
                nuevos_dominios1 = copiar_con_asignacion(nuevos_dominios, variable, dominio1)
                nuevos_dominios2 = copiar_con_asignacion(nuevos_dominios, variable, dominio2)
                para_revisar = self.nuevo_para_revisar(variable, None)
                self.visualizar(2, ' agregando ', para_revisar if para_revisar else " nada ", " para revisar ")
                soluciones.append(self.varias_soluciones(nuevos_dominios1, para_revisar))
                soluciones.append(self.varias_soluciones(nuevos_dominios2, para_revisar))
                return soluciones

    def seleccionar_variable(self, siguiente_variable):
        '''retorna la siguiente variable para hacer la división'''
        return seleccionar(siguiente_variable)


def particion_dominio(dominio):
    '''partición del dominio en dos'''
    dividir = len(dominio) // 2
    dominio1 = set(list(dominio)[:dividir])
    dominio2 = dominio - dominio1
    return dominio1, dominio2


def copiar_con_asignacion(dominios, variable=None, nuevo_dominio={True, False}):
    '''crea una copia de los dominios con asignación variable = nuevo_dominio. Si variable = None quiere decir que solo es una copia'''
    nuevosdominios = dominios.copy()
    if variable is not None:
        nuevosdominios[variable] = nuevo_dominio
    return nuevosdominios


def seleccionar(iterable):
    '''selecciona un elemento de iterable, retorna None si no existe dicho elemento.'''
    for e in iterable:
        return e


csp_crucigrama = CSP({'1': {'profe', 'iterativa', 'itxrativa', 'busqueda', 'sancocho'},
                      '2': {'porfavor', 'voraz', 'veras', 'hidra', 'bendecido'},
                      '3d': {'denos', 'anchura', 'sabor', 'sufrimiento'},
                      '3a': {'una', 'aestrella', 'sandia', 'osu!'},
                      '4': {'bonificacion', 'profundidad', 'kiwis', 'liga'},
                      '5': {'gracias', 'arco', 'palabra', 'leyendas'},
                      '6': {'equipos', 'camino', 'guaracha', 'borondo'},
                      '7': {'bendecido', 'nodo', 'macaco', 'cactus'},
                      },

                     [
                         Restriccion(('1',), eqLen(9)),
                         Restriccion(('2',), eqLen(5)),
                         Restriccion(('3a',), eqLen(9)),
                         Restriccion(('3d',), eqLen(7)),
                         Restriccion(('4',), eqLen(11)),
                         Restriccion(('5',), eqLen(4)),
                         Restriccion(('6',), eqLen(6)),
                         Restriccion(('7',), eqLen(4)),

                         Restriccion(('1', '3a'), eqChar(1, 3)),
                         Restriccion(('1', '4'), eqChar(3, 1)),
                         Restriccion(('1', '6'), eqChar(8, 1)),
                         Restriccion(('2', '4'), eqChar(3, 9)),
                         Restriccion(('3a', '3d'), eqChar(0, 0)),
                         Restriccion(('3d', '3a'), eqChar(0, 0)),
                         Restriccion(('3a', '1'), eqChar(3, 1)),
                         Restriccion(('4', '1'), eqChar(1, 3)),
                         Restriccion(('4', '2'), eqChar(9, 3)),
                         Restriccion(('5', '6'), eqChar(3, 5)),
                         Restriccion(('6', '5'), eqChar(5, 3)),
                         Restriccion(('6', '7'), eqChar(4, 0)),
                         Restriccion(('7', '6'), eqChar(0, 4))
                     ])

#print(Solucionador_consistencia(csp_entregas).una_solucion())

csp_ejemplo = CSP({'A': {1, 2, 3, 4}, 'B': {1, 2, 3, 4}, 'C': {1, 2, 3, 4}},
                  [Restriccion(('A', 'B'), lt),
                   Restriccion(('B', 'C'), lt),
                   Restriccion(('B',), noes(2))])

#print(BuscadorPI(Busqueda_CSP(csp_crucigrama)).ciclo())
print(Solucionador_consistencia(csp_crucigrama).varias_soluciones())


