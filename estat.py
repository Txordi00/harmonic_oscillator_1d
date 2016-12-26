# coding=utf-8
import math
import cmath
import numpy as np
from scipy import constants
import matplotlib.pylab as plt
import matplotlib.animation as animation
from matplotlib.legend_handler import HandlerLine2D

''' Classe Estat. Aquí emmagatzeno tota la informació rellevant del estat. Gran part de la funcionalitat es subordinarà
 a altres classes/funcions. '''
class Estat:
    def __init__(self, m=constants.m_e, k=1, x0=0, p0=0, traslacio=False, kick=False):
        self.m = m
        self.x0 = x0
        self.p0 = p0
        self.k = k
        coeffs = np.array([1,1])
        self.coeffs = coeffs / np.linalg.norm(coeffs)
        self.ona = FuncioOna(coeffs=self.coeffs,m=self.m,k=self.k)

''' Classe Funció d'ona. Aquí avaluem la funció d'ona i, per si es útil consultar-ho, emmagatzenem els darrers x i t, els coeficients
 i les constants a0 i omega. '''
class FuncioOna:
    ''' Inicialització de la classe. Guardem les constants i els coeficients cn. '''
    def __init__(self,coeffs, m=1, k=1):
        self.coeffs = coeffs
        self.omega = math.sqrt(k/m)
        self.a0 = math.sqrt(constants.hbar/(m*self.omega))

    ''' Funcionalitat Important. Avaluem la ona als diferents punts. '''
    def eval(self,x,t):
        self.x = x
        self.t = t
        N = np.size(coeffs)

        ''' Calculem els coeficients que multiplicaràn als polinomis d'Hermite i després els multipliquem per ells i ho sumem
        amb la funció np.polynomial.hermite.hermval. '''
        hermite_coeffs = [ coeffs[n]*1/( math.pi**(1/4) * math.sqrt(2**n*math.factorial(n)*self.a0) ) * math.exp(-(self.x/self.a0)**2 / 2) # f(n)*g(x)
                           * cmath.exp(-1j*self.omega*(float(n)+1/2)*self.t) for n in range(N)] # h(t)
        hermite_coeffs = np.array(hermite_coeffs)
        return np.polynomial.hermite.hermval(x=(self.x/self.a0), c=hermite_coeffs)

    ''' Dibuixem la ona en un rang de X i de T. Aprofitem que només hem de reescriure les línies i no els eixos i tota la resta de la figura
    per fer-ho de forma molt eficient amb blit. Igualment, '''
    def plot(self, x0=-5e-9, xf=5e-9, t0=0, tf=1, nx=100, nt=1000):
        X = np.linspace(x0, xf, nx)
        Y = np.array([self.eval(x=x, t=t0) for x in X])
        T = np.linspace(t0, tf, nt)

        fig, ax = plt.subplots()
        linia_real = ax.plot(X, Y.real, 'b', label='Real', animated=True)[0]
        linia_imag = ax.plot(X, Y.imag, 'r', label='Imaginari', animated=True)[0]
        ax.set_xlim(np.min(X), np.max(X))
        ax.set_ylim(-100000, 100000)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        lines = [linia_real, linia_imag]

        def animate(t):
            Y = np.array([self.eval(x=x, t=t) for x in X])
            lines[0].set_ydata(Y.real)
            lines[1].set_ydata(Y.imag)
            return lines

        ani = animation.FuncAnimation(fig, animate, T,
                                      interval=100, blit=True, repeat=False)

        plt.show()

            # for t in T:
        #     Y = np.array([self.eval(x=x, t=t) for x in X])
        #     linia_real, = plt.plot(X, Y.real, 'b', label='Real')
        #     linia_imag, = plt.plot(X, Y.imag, 'r', label='Imaginari')
        #     plt.legend(handler_map={linia_real: HandlerLine2D()})
        #     plt.xlim(np.min(X), np.max(X))
        #     plt.ylim(-100000, 100000)
        #     plt.show()
        #     plt.pause(0.01)
        #     plt.clf()


if __name__ == '__main__':
    coeffs = np.array([1,1])
    coeffs = coeffs / np.linalg.norm(coeffs)
    # ona = FuncioOna(coeffs=coeffs, m=constants.m_e)
    estat = Estat(m=constants.m_e,k=1)
    estat.ona.plot(x0=-5e-9,xf=5e-9,t0=0,tf=1,nx=100,nt=1000)


    print('Sortida')