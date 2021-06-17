class Clausula():
    def __init__(self, cabeza, cuerpo=[]):
        self.cabeza = cabeza
        self.cuerpo = cuerpo

    def __str__(self):
        if self.cuerpo:
            return self.cabeza + " <-" + "^".join(self.cuerpo) + "."
        else:
            return self.cabeza + "."


class Observable():
    def __init__(self, atomo):
        self.atomo = atomo

    def __str__(self):
        return "pregunta " + self.atomo + "."


def si(resp):
    '''Retorna verdadero si la respuestaes si'''
    return resp.lower() in ['si', 'Si', 's', 'yes', 'Yes', 'y']


from Busqueda import Visualizable


class KB(Visualizable):
    '''La KB es un conjunto de clausulas. Tambien se crea un diccionario para dar acceos más rapido a las clausulas
        con un atomo en la cabeza'''

    def __init__(self, declaraciones=[]):
        self.declaraciones = declaraciones
        self.clausulas = [c for c in declaraciones if isinstance(c, Clausula)]
        self.observables = [c.atomo for c in declaraciones if isinstance(c, Observable)]
        self.atomo_a_clausula = {}
        for c in self.clausulas:
            if c.cabeza in self.atomo_a_clausula:
                self.atomo_a_clausula[c.cabeza].append(c)
            else:
                self.atomo_a_clausula[c.cabeza] = [c]

    def clausulas_de_atomo(self, a):
        '''Retorna un conjunto de clausulas con un atomo "a" como cabeza'''
        if a in self.atomo_a_clausula:
            return self.atomo_a_clausula[a]
        else:
            return set()

    def __str__(self):
        return '\n'.join([str(c) for c in self.declaraciones])


covid_KB = KB([Clausula('no_contagio', ['en_casa']),
               Clausula('en_casa'),
               Clausula('mas_trabajo', ['cuarentena', 'en_casa']),
               Observable('fiebre'),
               Clausula('escalofrio', ['fiebre']),
               Clausula('cuarentena'),
               Clausula('no_contagio', ['cuarentena'])])

red_electrica_KB = KB([Clausula('l1', ['f_w0']),
                       Clausula('f_w0', ['u_s2', 'f_w1']),
                       Clausula('f_w0', ['d_s2', 'f_w2']),
                       Observable('u_s2'),
                       Observable('d_s2'),
                       Clausula('f_w1', ['u_s1', 'f_w3']),
                       Clausula('f_w2', ['d_s1', 'f_w3']),
                       Observable('u_s1'),
                       Observable('d_s1'),
                       Clausula('f_w3', ['c_cb1', 'f_op']),
                       Observable('c_cb1'),
                       Observable('f_op'),
                       Clausula('f_w4', ['c_s3', 'f_w3']),
                       Observable('c_s3'),
                       Clausula('f_p1', ['f_w3'])
                       ])
plomeria_KB = KB([Clausula('presurizacion_p1'),
                  Clausula('presurizacion_p2', ['on_t1', 'presurizacion_p1']),
                  Clausula('flujo_ducha', ['on_t2', 'presurizacion_p2']),
                  Clausula('mojado_bañera', ['flujo_ducha']),
                  Clausula('flujo_d2', ['mojado_bañera', 'sin_tapon_bañera']),
                  Observable('on_t1'),
                  Observable('on_t2'),
                  Observable('sin_tapon_bañera'),
                  Clausula('flujo_d1', ['flujo_d2']),
                  Clausula('presurizacion_p3', ['on_t1', 'presurizacion_p1']),
                  Observable('on_t3'),
                  Clausula('flujo_lavamanos', ['on_t3', 'presurizacion_p3']),
                  Clausula('mojado_lavamanos', ['flujo_lavamanos']),
                  Clausula('flujo_d3', ['mojado_lavamanos', 'sin_tapon_lavamanos']),
                  Clausula('flujo_d1', ['flujo_d3']),
                  Clausula('sin_tapon_lavamanos'),
                  Clausula('piso_mojado', ['con_tapon_bañera', 'flujo_ducha', 'bañera_llena', 'no_flujo_d2']),
                  Clausula('piso_mojado', ['con_tapon_lavamanos', 'flujo_lavamanos', 'lavamanos_lleno', 'no_flujo_d3']),
                  Observable('con_tapon_bañera'),
                  Observable('con_tapon_lavamanos'),
                  Observable('bañera_llena'),
                  Observable('lavamanos_lleno'),
                  Clausula('no_flujo_d2', ['con_tapon_bañera', 'tapon_bañera_funcional']),
                  Clausula('no_flujo_d3', ['con_tapon_lavamanos', 'tapon_lavamanos_funcional']),
                  Observable('tapon_bañera_funcional'),
                  Observable('tapon_lavamanos_funcional')])


def punto_fijo(kb):
    '''Retorna el punto fijo de la base de conocimiento kb, esto es el conjunto minimo de consecuencias'''
    fp = pregunta_observable(kb)
    adicionado = True
    while adicionado:
        adicionado = False  # Se vuelve verdadero cuando se agrega un atomo al fp durante esta iteracion
        for c in kb.clausulas:
            if c.cabeza not in fp and all(b in fp for b in c.cuerpo):
                fp.add(c.cabeza)
                adicionado = True
                kb.visualizar(1, c.cabeza, " adicionado al fp por la clausula ", c)

    return fp


def pregunta_observable(kb):
    return {a for a in kb.observables if si(input(a + " es verdadero? "))}


def arriba_abajo(kb, cuerpo_respuesta, indentacion=""):
    '''retorna Verdadero si kb |- cuerpo_respuesta.
    cuerpo_respuesta con los átomos a probar'''
    kb.visualizar(1, indentacion, 'si <-', ' ^ '.join(cuerpo_respuesta))
    if cuerpo_respuesta:
        seleccion = cuerpo_respuesta[0]  # selecciona el primer átomo de cuerpo_respuesta
        if seleccion in kb.observables:
            return (si(input(seleccion + " es verdadero? ")) and arriba_abajo(kb, cuerpo_respuesta[1:],
                                                                              indentacion + " "))
        else:
            return any(arriba_abajo(kb, cl.cuerpo + cuerpo_respuesta[1:], indentacion + " ") for cl in
                       kb.clausulas_de_atomo(seleccion))
    else:
        return True  # cuando el cuerpo_respuesta queda vacío


ejemplo = KB([Clausula('a', ['b', 'c']),
              Clausula('b', ['g', 'e']),
              Clausula('b', ['d', 'e']),
              Clausula('c', ['e']),
              Clausula('d'),
              Clausula('e'),
              Clausula('f', ['a', 'g'])])

quiz3 = KB(
    [Clausula('e1_empuje', ['v13_abierto', 'v13_ok', 'v13_presurizada', 'v14_abierto', 'v14_ok', 'v14_presurizada']),
     Clausula('v13_presurizada', ['v5_presurizada', 'v5_abierto', 'v5_ok']),
     Clausula('v13_presurizada', ['v6_presurizada', 'v6_abierto', 'v6_ok']),
     Clausula('v14_presurizada', ['v7_presurizada', 'v7_abierto', 'v7_ok']),
     Clausula('v14_presurizada', ['v8_presurizada', 'v8_abierto', 'v8_ok']),
     Clausula('v5_presurizada', ['v1_presurizada', 'v1_abierto', 'v1_ok']),
     Clausula('v6_presurizada', ['v1_presurizada', 'v1_abierto', 'v1_ok']),
     Clausula('v7_presurizada', ['v2_presurizada', 'v2_abierto', 'v2_ok']),
     Clausula('v8_presurizada', ['v2_presurizada', 'v2_abierto', 'v2_ok']),
     Clausula('v12_presurizada', ['v4_presurizada', 'v4_abierto', 'v4_ok']),
     Clausula('v1_presurizada', ['t1_presurizada']),
     Clausula('v2_presurizada', ['t2_presurizada']),
     Clausula('v13_abierto'),
     Clausula('v13_ok'),
     Clausula('v14_abierto'),
     Clausula('v14_ok'),
     Clausula('v5_abierto'),
     Clausula('v5_ok'),
     Clausula('v6_abierto'),
     Clausula('v6_ok'),
     Clausula('v7_abierto'),
     Clausula('v7_ok'),
     Clausula('v8_abierto'),
     Clausula('v8_ok'),
     Clausula('t1_presurizada'),
     Clausula('v2_abierto'),
     Clausula('v2_ok'),
     Clausula('v1_abierto'),
     Clausula('v1_ok'),
     Clausula('e2_empuje', ['v15_abierto', 'v15_ok', 'v15_presurizada', 'v16_abierto', 'v16_ok', 'v16_presurizada']),
     Clausula('v15_presurizada', ['v9_presurizada', 'v9_abierto', 'v9_ok']),
     Clausula('v15_presurizada', ['v10_presurizada', 'v10_abierto', 'v10_ok']),
     Clausula('v16_presurizada', ['v11_presurizada', 'v11_abierto', 'v11_ok']),
     Clausula('v16_presurizada', ['v12_presurizada', 'v12_abierto', 'v12_ok']),
     Clausula('v9_presurizada', ['v3_presurizada', 'v3_abierto', 'v3_ok']),
     Clausula('v10_presurizada', ['v3_presurizada', 'v3_abierto', 'v3_ok']),
     Clausula('v11_presurizada', ['v4_presurizada', 'v4_abierto', 'v4_ok']),
     Clausula('v3_presurizada', ['t1_presurizada']),
     Clausula('v4_presurizada', ['t2_presurizada']),
     Clausula('v15_abierto'),
     Clausula('v15_ok'),
     Clausula('v16_abierto'),
     Clausula('v16_ok'),
     Clausula('v9_abierto'),
     Clausula('v9_ok'),
     Clausula('v10_abierto'),
     Clausula('v10_ok'),
     Clausula('v11_abierto'),
     Clausula('v11_ok'),
     Clausula('v12_abierto'),
     Clausula('v12_ok'),
     Clausula('v3_abierto'),
     Clausula('v3_ok'),
     Clausula('v4_abierto'),
     Clausula('v4_ok'),
     Clausula('t2_presurizada')

     ])
print(arriba_abajo(quiz3, ['e1_empuje','e2_empuje']))
