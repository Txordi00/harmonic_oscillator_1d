# coding=utf-8
import math
import cmath
import numpy as np
from scipy import constants
from scipy import misc
from scipy import integrate
import matplotlib.pylab as plt
import matplotlib.animation as animation
from matplotlib.legend_handler import HandlerLine2D

INF = 1e6

'''Energía per cada n'''
def E_n(n, omega):
    return constants.hbar * omega * (float(n) + 1/2)

'''Part temporal de la solució de l'equació d'Schrodinger depenent del temps.'''
def exp_t(E_n,t):
    return cmath.exp(-1j*E_n*t/constants.hbar)

'''Solucions estàcionaries de l'equació d'Schrodinger.'''
def phi_n(x, n, a0):
    coeff = np.zeros((n+1,))
    coeff[n] = 1
    fn = 1/( math.pi**(1/4) * math.sqrt(2**n*math.factorial(n)*a0) )
    xbar = x/a0
    fx = math.exp(-xbar**2/2) * np.polynomial.hermite.hermval(x=xbar, c=coeff)
    return fn * fx

''' Classe Estat. Aquí emmagatzemo tota la informació rellevant del estat. Gran part de la funcionalitat es subordinarà
 a altres classes/funcions. '''
class Estat:
    def __init__(self, coeffs, m=constants.m_e, k=1):
        self.m = m
        self.k = k
        self.coeffs = coeffs
        '''Dintre d'estat guardem la funció d'ona.'''
        self.ona = FuncioOna(coeffs=self.coeffs,m=self.m,k=self.k)

    '''Fer Kick implica donar-li un impuls p0 al moment emmagatzemat a la funció d'ona. Posteriorment recalculem la descomposició
    de la ona en les funcions pròpies.'''
    def kick(self,p0):
        self.ona.p0 = self.ona.p0 + p0
        self.ona.update_coeffs()
        self.coeffs = self.ona.coeffs

    '''Trasladar implica aplicar un desplaçament a la x0 emmagatzemada a la funció d'ona. Posteriorment recalculem la descomposició
    de la ona en les funcions pròpies.'''
    def traslacio(self,x0):
        self.ona.x0 = self.ona.x0 + x0
        self.ona.update_coeffs()
        self.coeffs = self.ona.coeffs

''' Classe Funció d'ona. Aquí avaluem la funció d'ona i, per si es útil consultar-ho, emmagatzenem els darrers x i t, els coeficients
 i les constants a0 i omega. '''
class FuncioOna:
    ''' Inicialització de la classe. Guardem les constants i els coeficients cn. '''
    def __init__(self,coeffs, m=1, k=1, x0=0, p0=0):
        self.coeffs = coeffs
        self.x0 = x0
        self.p0 = p0
        self.omega = math.sqrt(k/m)
        self.a0 = math.sqrt(constants.hbar/(m*self.omega))

    ''' Funcionalitat Important. Avaluem la ona als diferents punts (sense fer ni kick ni translació).'''
    def eval0(self,x,t):
        N = np.size(self.coeffs)

        return np.sum([self.coeffs[n]*phi_n(x=x,n=n,a0=self.a0) * exp_t(E_n=E_n(n,self.omega),t=t) for n in range(N)])

    '''Apliquem el Kick.'''
    def eval1(self,x,t):
        return cmath.exp(-1j*self.p0*x/constants.hbar) * self.eval0(x,t)

    '''Apliquem la Traslació.'''
    def eval(self,x,t):
        '''De forma pura hauriem de aproximar l'operador de traslació tal que així:'''
        # '''Defineixo aquesta funció auxiliar perquè només volem calcular la parcial respecte de x.'''
        # def fx(x):
        #     return self.eval1(x,t)
        # '''Aproximació Taylor de grau 1 de U(x0)*f(x,t) = exp(-i*x0*p/hbar)*f(x,t). Això sol serveix per x molt petites,
        #   en un cas real hauriem de aproximar amb més graus.'''
        # return fx(x) - self.x0*misc.derivative(fx,x,dx=1e-18)
        '''Però, com que sabem que U(x0)*f(x,t) = f(x-x0,t):'''
        return self.eval1(float(x)-self.x0,t)

    '''Actualitzem els coeficients calculant la integral de phi_n'(x)*ona(x,0).'''
    def update_coeffs(self):
        '''Definim els integrands'''
        integrand_real = lambda x, n: (np.conj(phi_n(x=x,n=n,a0=self.a0)) * self.eval(x=x,t=0)).real
        integrand_imag = lambda x, n: (np.conj(phi_n(x=x,n=n,a0=self.a0)) * self.eval(x=x,t=0)).imag

        '''Calculem els coeficients cn fins que la suma de les probabilitats excedeix 0.99'''
        accum_prob = 0
        n=0
        coeffs_aux = []
        while (accum_prob<0.99):
            # cn = integrate.quadrature(func=integrand_real,a=-INF,b=+INF,args=(n,),vec_func=False,tol=1e-18,rtol=1e-18)[0] + 1j*integrate.quadrature(func=integrand_imag,a=-INF,b=+INF,args=(n,),vec_func=False,tol=1e-18,rtol=1e-18)[0]
            cn = integrate.quad(func=integrand_real, a=-np.inf, b=+np.inf, args=(n,))[0] + 1j * integrate.quad(func=integrand_imag, a=-np.inf, b=+np.inf, args=(n,))[0]
            coeffs_aux = coeffs_aux + [cn]
            accum_prob = accum_prob + abs(cn)**2
            n = n + 1
            print(str(n) + ' coeficients amb precissió ' + str(accum_prob))
        self.coeffs = np.array(coeffs_aux)


    ''' Dibuixem la ona en un rang de X i de T. Aprofitem que només hem de reescriure les línies i no els eixos i tota la resta de la figura
    per fer-ho de forma molt eficient amb blit.'''
    def plot(self, x0=-5e-9, xf=5e-9, t0=0, tf=1, nx=100, nt=1000):
        '''Valors de les X i T i valors inicials de la funció d'ona Y.'''
        X = np.linspace(x0, xf, nx)
        Y = np.array([self.eval(x=x, t=t0) for x in X])
        T = np.linspace(t0, tf, nt)

        '''Definició/Inicialització de tots els plots.'''
        fig, ax = plt.subplots()
        linia_real = ax.plot(X, Y.real, 'b', label='Real', animated=True)[0]
        linia_imag = ax.plot(X, Y.imag, 'r', label='Imaginari', animated=True)[0]
        ax.set_xlim(np.min(X), np.max(X))
        ylim = 1.25*np.max([abs(np.min(Y)), abs(np.max(Y))])
        ax.set_ylim(-ylim, ylim)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        lines = [linia_real, linia_imag]

        '''Funció auxiliar per computar la animació. Avalua per cada temps la funció d'ona Y als X donats.'''
        def animate(t):
            Y = np.array([self.eval(x=x, t=t) for x in X])
            lines[0].set_ydata(Y.real)
            lines[1].set_ydata(Y.imag)
            return lines

        '''Animació eficient. '''
        ani = animation.FuncAnimation(fig, animate, T,
                                      interval=100, blit=True, repeat=False)
        plt.show()

'''Main per fer petites proves'''
if __name__ == '__main__':
    coeffs = np.array([1,0])
    coeffs = coeffs / np.linalg.norm(coeffs)
    estat = Estat(coeffs=coeffs, m=constants.m_e, k=1)

    estat.ona.plot(x0=-5e-9,xf=5e-9,t0=0,tf=1,nx=1000,nt=20)

    estat.kick(p0=1e-10)
    estat.ona.plot(x0=-5e-9, xf=5e-9, t0=0, tf=1, nx=1000, nt=20)

    estat.kick(p0=-1e-10)
    estat.ona.plot(x0=-5e-9, xf=5e-9, t0=0, tf=1, nx=1000, nt=20)

    estat.traslacio(x0=1e-9)
    estat.ona.plot(x0=-5e-9, xf=5e-9, t0=0, tf=1, nx=1000, nt=20)

    print('Sortida Correcta')
