# coding=utf-8
import math
import cmath
import numpy as np
##TODO: Arreglar import scipy as sp
# from scipy import constants
from scipy import integrate
from scipy.misc import derivative
import matplotlib.pylab as plt
import matplotlib.animation as animation

HBAR = 1
M = 1
K = 1

def adjust_err_bar(errobj, x, y, x_error, y_error):
    ln, (errx_top, errx_bot, erry_top, erry_bot), (barsx, barsy) = errobj

    if(not isinstance(x, np.ndarray)):
        x_base = np.array([x])
    else:
        x_base = x
    if(not isinstance(y, np.ndarray)):
        y_base = np.array([y])
    else:
        y_base = y

    ln.set_data(x_base, y_base)
    xerr_top = x_base + x_error
    xerr_bot = x_base - x_error
    yerr_top = y_base + y_error
    yerr_bot = y_base - y_error

    errx_top.set_xdata(xerr_top)
    errx_bot.set_xdata(xerr_bot)
    errx_top.set_ydata(y_base)
    errx_bot.set_ydata(y_base)

    erry_top.set_xdata(x_base)
    erry_bot.set_xdata(x_base)
    erry_top.set_ydata(yerr_top)
    erry_bot.set_ydata(yerr_bot)

    new_segments_x = [np.array([[xt, y], [xb,y]]) for xt, xb, y in zip(xerr_top, xerr_bot, y_base)]
    new_segments_y = [np.array([[x, yt], [x,yb]]) for x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
    # print(new_segments_x)
    barsx.set_segments(new_segments_x)
    barsy.set_segments(new_segments_y)

'''Energía per cada n'''
def E_n(n, omega):
    return HBAR * omega * (float(n) + 1/2)

'''Part temporal de la solució de l'equació d'Schrodinger depenent del temps.'''
def exp_t(E_n,t):
    return cmath.exp(-1j*E_n*t/HBAR)

'''Solucions estàcionaries de l'equació d'Schrodinger.'''
def phi_n(x, n, a0):
    coeff = np.zeros((n+1,))
    coeff[n] = 1
    fn = 1/( math.pi**(1./4.) * math.sqrt(2**n*math.factorial(n)*a0) )
    xbar = x/a0
    fx = math.exp(-xbar**2/2) * np.polynomial.hermite.hermval(x=xbar, c=coeff)
    return fn * fx

''' Classe Estat. Aquí emmagatzemo tota la informació rellevant del estat. Gran part de la funcionalitat es subordinarà
 a altres classes/funcions. '''
class Estat:
    def __init__(self, coeffs, m=M, k=K):
        self.m = m
        self.k = k
        self.coeffs = coeffs / np.linalg.norm(coeffs)
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
    def __init__(self,coeffs, m=M, k=K, x0=0, p0=0):
        self.coeffs = coeffs
        self.x0 = x0
        self.p0 = p0
        self.m = m
        self.k = k
        self.omega = math.sqrt(k/m)
        self.a0 = math.sqrt(HBAR/(m*self.omega))

    ''' Funcionalitat Important. Avaluem la ona als diferents punts (sense fer ni kick ni translació).'''
    def eval0(self,x,t):
        N = np.size(self.coeffs)

        return np.sum(np.array([self.coeffs[n]*phi_n(x=x,n=n,a0=self.a0) * exp_t(E_n=E_n(n,self.omega),t=t) for n in range(N)]))

    '''Apliquem el Kick.'''
    def eval1(self,x,t):
        return cmath.exp(-1j*self.p0*x/HBAR) * self.eval0(x,t)

    '''Apliquem la Traslació.'''
    def eval(self,x,t):
        '''De forma pura hauriem d'aproximar l'operador de traslació tal que així:'''
        # '''Defineixo aquesta funció auxiliar perquè només volem calcular la parcial respecte de x.'''
        # def fx(x):
        #     return self.eval1(x,t)
        # '''Aproximació Taylor de grau 1 de U(x0)*f(x,t) = exp(-i*x0*p/hbar)*f(x,t). Això sol serveix per x molt petites,
        #   en un cas real hauriem de aproximar amb més graus.'''
        # return fx(x) - self.x0*sp.misc.derivative(fx,x,dx=1e-18)
        '''Però, com que sabem que U(x0)*f(x,t) = f(x-x0,t):'''
        return self.eval1(x-self.x0,t)

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
            # cn = integrate.romberg(function=integrand_real,a=-INF,b=+INF,args=(n,),vec_func=False,divmax=10) + 1j*integrate.romberg(function=integrand_imag,a=-INF,b=+INF,args=(n,),vec_func=False,divmax=5)
            cn = integrate.quad(func=integrand_real, a=-np.inf, b=+np.inf, args=(n,))[0] + 1j * integrate.quad(func=integrand_imag, a=-np.inf, b=+np.inf, args=(n,))[0]
            coeffs_aux = coeffs_aux + [cn]
            accum_prob = accum_prob + abs(cn)**2
            n = n + 1
            # print('coeficient ' + str(n) + ', probabilitat acumulada: ' + str(accum_prob))
        print(str(n) + ' coeficients amb precissió ' + str(accum_prob))
        self.coeffs = np.array(coeffs_aux)


    def expected_value(self, operator, t):
        func_prob = lambda x, t: abs(self.eval(x,t))**2
        func_x = lambda x,t: func_prob(x,t)*x
        func_x2 = lambda x,t: func_prob(x,t)*x**2
        func_p = lambda x,t: np.conj(self.eval(x,t)) * HBAR/1j * derivative(func=self.eval,x0=x,dx=1e-2,n=1,args=(t,))
        func_p2 = lambda x,t: np.conj(self.eval(x,t)) * (-HBAR**2) * derivative(func=self.eval,x0=x,dx=1e-2,n=2,args=(t,))

        if(operator.lower()=='x'):
            return integrate.quad(func=func_x, a=-np.inf, b=+np.inf, args=(t,))[0]
        elif(operator.lower()=='x2'):
            return integrate.quad(func=func_x2, a=-np.inf, b=+np.inf, args=(t,))[0]
        elif (operator.lower() == 'p'):
            return integrate.quad(func=func_p, a=-np.inf, b=+np.inf, args=(t,))[0]
        elif (operator.lower() == 'p2'):
            return integrate.quad(func=func_p2, a=-np.inf, b=+np.inf, args=(t,))[0]



    ''' Dibuixem la ona en un rang de X i de T. Aprofitem que només hem de reescriure les línies i no els eixos i tota la resta de la figura
    per fer-ho de forma eficient amb blit.'''
    def plot(self, x0=-10, xf=10, t0=0, tf=10, nx=480, nt=100):
        '''Valors de les X i T i valors inicials de la funció d'ona Y.'''
        X = np.linspace(x0, xf, nx)
        T = np.linspace(t0, tf, nt)
        V = 1./2.*self.k*X**2
        print('Calculant els valors de la funció de ona...')
        Y = [np.array([self.eval(x=x, t=t) for x in X]) for t in T]
        EXP_VAL = np.array([self.expected_value(operator='x',t=t) for t in T])
        STD = np.sqrt(np.array([self.expected_value(operator='x2',t=t) for t in T]) - EXP_VAL**2)

        '''Definició/Inicialització de tots els plots.'''
        fig, ax = plt.subplots()
        ax.set_xlim(np.min(X), np.max(X))
        ylim = 1.25*np.max([abs(np.min(Y[0])), abs(np.max(Y[0]))])
        ax.set_ylim(-ylim, ylim)

        linia_real = ax.plot(X, Y[0].real, 'b', label='Re[Ona]', animated=True)[0]
        linia_imag = ax.plot(X, Y[0].imag, 'r', label='Im[Ona]', animated=True)[0]
        linia_prob = ax.plot(X, abs(Y[0])**2, 'y', label='Prob', animated=True)[0]
        linia_exp = ax.plot([EXP_VAL[0],EXP_VAL[0]], [-ylim,ylim], 'k', label='Exp_val', animated=True)[0]
        # linia_std, bottoms_tops, _ = ax.errorbar(EXP_VAL[0], ylim/4., xerr=STD[0], yerr=0, label='Std', animated=True)
        errobj = ax.errorbar(EXP_VAL[0], ylim / 4., xerr=STD[0], yerr=0, label='Std', animated=True)
        linia_pot = ax.plot(X, V-ylim, 'g', label='Potencial', animated=False)[0]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        lines = [linia_real, linia_imag, linia_prob, linia_exp, linia_pot]

        '''Funció auxiliar per computar la animació. Avalua per cada temps la funció d'ona Y als X donats.'''
        def animate(n):
            # Y = np.array([self.eval(x=x, t=t) for x in X])
            lines[0].set_ydata(Y[n].real)
            lines[1].set_ydata(Y[n].imag)
            lines[2].set_ydata(abs(Y[n])**2)
            lines[3].set_xdata([EXP_VAL[n],EXP_VAL[n]])
            adjust_err_bar(errobj=errobj,x=EXP_VAL[n],y=ylim/4.,x_error=STD[n],y_error=0)
            # bottoms_tops[0].set_xdata(EXP_VAL[n]-STD[n])
            # bottoms_tops[1].set_xdata(EXP_VAL[n] + STD[n])
            # lines[4].set_xdata(EXP_VAL[n])
            return lines

        '''Animació eficient.'''
        ani = animation.FuncAnimation(fig, animate, range(len(T)),
                                      interval=100, blit=True, repeat=False)
        plt.show()

def prova(a):
    a = a + 1
    print(a)

'''Main per fer petites proves'''
if __name__ == '__main__':
    x0 = -10
    xf = 10
    t0 = 0
    tf = 10
    nx = 480
    nt = 100
    a = 0
    print(a)
    prova(a)
    print(a)

    coeffs = np.array([1])
    estat = Estat(coeffs=coeffs, m=M, k=1)

    # int_phi = lambda x, n, a0: abs(phi_n(x=x,n=n,a0=a0))**2
    # int_ona = lambda x: abs(estat.ona.eval(x=x,t=0))**2
    # I_phi = integrate.quad(func=int_phi, a=-np.inf, b=+np.inf, args=(0,1))[0]
    # print('Àrea phi: ' + str(abs(I_phi)**2))
    # I_ona = integrate.quad(func=int_ona, a=-np.inf, b=+np.inf)[0]
    # print('Àrea ona: ' + str(abs(I_ona)**2))

    estat.ona.plot(x0=x0,xf=xf,t0=t0,tf=tf,nx=nx,nt=nt)
    print('Valor esperat: ' + str(estat.ona.expected_value(operator='x',t=0)))


    # estat.kick(p0=8)
    # estat.ona.plot(x0=x0,xf=xf,t0=t0,tf=tf,nx=nx,nt=nt)
    #
    # estat.kick(p0=-8)
    # estat.ona.plot(x0=x0,xf=xf,t0=tf+t0,tf=tf+tf,nx=nx,nt=nt)
    #
    estat.traslacio(x0=1)
    estat.ona.plot(x0=x0,xf=xf,t0=t0,tf=tf,nx=nx,nt=nt)

    print('Sortida Correcta')
