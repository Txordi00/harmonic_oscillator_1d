# coding=utf-8
import math
import cmath
import numpy as np
# from scipy import constants
from scipy import integrate
from scipy.misc import derivative
import matplotlib.pylab as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
import sys
from altres_funcions import *

'''Constants globals.'''
HBAR = 1
M = 1
K = 1
'''Per fer-ho anar tot més ràpit, es recomanable baixar la tolerància si no volem massa precissió'''
TOL = 1e-4

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

'''Funció secció 4 dels estats de mínima indeterminació.'''
def psi_squeezed(x, p0=0, x0=0, m=M, k=K, xi=0):
    omega = math.sqrt(k/m)
    delta_x = math.sqrt(HBAR/(2.*m*omega))*math.exp(-xi)
    return cmath.exp(1j*p0*x/HBAR) * math.exp(-(x-x0)**2/(4.*delta_x**2))

''' Classe Estat. Aquí emmagatzemo tota la informació rellevant del estat. Gran part de la funcionalitat es subordinarà
    a la classe FuncioOna. Podem definir un estat mitjançant els coeficients de les funcions pròpies 'coeffs' o
     mitjançant una funció qualsevol 'fx' (com fem a la secció 4). La funció fx rep el valor x com a primer argument i
     NO s'ha d'especificar a fx_args. La resta d'arguments de la funció (si n'hi han) vindràn en ordre a fx_args.'''
class Estat:
    def __init__(self, coeffs=None, m=M, k=K, fx = None, fx_args=()):
        assert (coeffs is not None or fx is not None)
        self.m = m
        self.k = k
        self.coeffs = coeffs
        if(coeffs is not None):
            self.coeffs = coeffs / np.linalg.norm(coeffs)
        '''Dintre d'estat guardem la funció d'ona.'''
        self.ona = FuncioOna(coeffs=self.coeffs,m=self.m,k=self.k,fx=fx,fx_args=fx_args)
        if(fx is not None):
            self.ona.update_coeffs()
            self.ona.fx = None
            self.coeffs = self.ona.coeffs


    '''Fer Kick implica donar-li un impuls p0 al moment emmagatzemat a la funció d'ona. Posteriorment recalculem la
    descomposició de la ona en les funcions pròpies.'''
    def kick(self,p0):
        self.ona.p0 = p0
        self.ona.update_coeffs()
        self.ona.p0 = 0
        self.coeffs = self.ona.coeffs

    '''Trasladar implica aplicar un desplaçament a la x0 emmagatzemada a la funció d'ona. Posteriorment recalculem
    la descomposició de la ona en les funcions pròpies.'''
    def traslacio(self,x0):
        self.ona.x0 = x0
        self.ona.update_coeffs()
        self.ona.x0 = 0
        self.coeffs = self.ona.coeffs

    '''Traslació i kick simultanis.'''
    def traslacio_kick(self,x0,p0):
        self.ona.x0 = x0
        self.ona.p0 = p0
        self.ona.update_coeffs()
        self.ona.x0 = 0
        self.ona.p0 = 0
        self.coeffs = self.ona.coeffs

    '''Valor esperat'''
    def valor_esperat(self, t, operator):
        return self.ona.expected_value(operator,t)

    '''Desviació estàndard'''
    def std(self, t, operator):
        assert(operator.lower() in ['x','p'])
        return math.sqrt(self.ona.expected_value(operator+'2',t) - self.ona.expected_value(operator,t)**2)

''' Classe Funció d'ona. Aquí avaluem la funció d'ona i els seus valors esperats i fem els plots. La classe Estat
s'encarrega de gestionar els passos generals a seguir i aquí es fan els càlculs. De la mateixa manera que hem dit a
Estat, podem partir dels coeficients de les funcions pròpies o d'una funció qualsevol de x 'fx'.'''
class FuncioOna:
    ''' Inicialització de la classe. Guardem les constants i els coeficients cn. '''
    def __init__(self, coeffs=None, m=M, k=K, fx=None, fx_args=()):
        assert(coeffs is not None or fx is not None)
        self.coeffs = coeffs
        self.fx = fx
        self.fx_args = fx_args
        '''Si partim d'una funció fx, la necessitem normalitzada.'''
        if(self.fx is not None):
            abs_fx_2 = lambda x: abs(self.fx(x, *self.fx_args))**2
            self.normaliztion_factor = math.sqrt(1./integrate.quad(func=abs_fx_2,a=-np.inf,b=np.inf)[0])
            print('[Normalització] Constant de normalització C: ' + str(self.normaliztion_factor))
        self.x0 = 0
        self.p0 = 0
        self.m = m
        self.k = k
        self.omega = math.sqrt(k/m)
        self.a0 = math.sqrt(HBAR/(m*self.omega))

    ''' Funcionalitat Important. Avaluem la ona als diferents punts (sense fer ni kick ni translació).'''
    def eval0(self,x,t):
        N = np.size(self.coeffs)

        return np.sum(np.array([self.coeffs[n]*phi_n(x=x,n=n,a0=self.a0) * exp_t(E_n=E_n(n,self.omega),t=t)
                                for n in range(N)]))

    '''Apliquem el Kick.'''
    def eval1(self,x,t):
        return cmath.exp(-1j*self.p0*x/HBAR) * self.eval0(x,t)

    '''Apliquem la Traslació'''
    def eval2(self,x,t):
        '''De forma pura hauriem d'aproximar l'operador de traslació tal que així:'''
        # '''Defineixo aquesta funció auxiliar perquè només volem calcular la parcial respecte de x.'''
        # def fx(x):
        #     return self.eval1(x,t)
        # '''Aproximació Taylor de grau 1 de U(x0)*f(x,t) = exp(-i*x0*p/hbar)*f(x,t). Això sol serveix per x molt petites,
        #   en un cas real hauriem de aproximar amb més graus.'''
        # return fx(x) - self.x0*sp.misc.derivative(fx,x,dx=1e-18)
        '''Però, com que sabem que U(x0)*f(x,t) = f(x-x0,t):'''
        return self.eval1(x - self.x0, t)

    '''Avaluem la funció. En cas que haguem definit la funció amb coeficients, retornem eval2 que és la funció amb el Kick
       i la Traslació (si n'hi han). En cas contrari, si partim d'una funció, com a la secció 4, tornem el valor de la
       funció normalitzada amb els seus arguments.'''
    def eval(self,x,t):
        if(self.fx is None):
            return self.eval2(x,t)
        else:
            return self.normaliztion_factor * self.fx(x, *self.fx_args)

    '''Actualitzem els coeficients calculant la integral de phi_n'(x)*ona(x,0).'''
    def update_coeffs(self):
        '''Definim els integrands'''
        integrand_real = lambda x, n: (np.conj(phi_n(x=x,n=n,a0=self.a0)) * self.eval(x=x,t=0)).real
        integrand_imag = lambda x, n: (np.conj(phi_n(x=x,n=n,a0=self.a0)) * self.eval(x=x,t=0)).imag

        '''Calculem els coeficients cn fins que la suma de les probabilitats excedeix 0.99'''
        accum_prob = 0
        n=0
        coeffs_aux = []
        while (accum_prob<(1.-TOL)):
            cn = integrate.quad(func=integrand_real, a=-np.inf, b=+np.inf, args=(n,))[0] + 1j * \
                                            integrate.quad(func=integrand_imag, a=-np.inf, b=+np.inf, args=(n,))[0]
            coeffs_aux = coeffs_aux + [cn]
            accum_prob = accum_prob + abs(cn)**2
            n = n + 1
        print(str(n) + ' coeficients amb precisió ' + str(accum_prob))
        self.coeffs = np.array(coeffs_aux)

    '''Funcio per calcular els valors esperats. Per seleccionar el valor què vols s'ha de escollir un operador x, x2, p, p2.'''
    def expected_value(self, operator, t):
        '''Definim les funcions per integrar'''
        func_prob = lambda x, t: abs(self.eval(x,t))**2
        func_x = lambda x,t: func_prob(x,t)*x
        func_x2 = lambda x,t: func_prob(x,t)*x**2
        func_p = lambda x,t: (np.conj(self.eval(x,t)) * HBAR/1j * derivative(func=self.eval,x0=x,dx=1e-2,n=1,
                                                                             args=(t,))).real
        func_p2 = lambda x,t: (np.conj(self.eval(x,t)) * (-HBAR**2) * derivative(func=self.eval,x0=x,dx=1e-2,n=2,
                                                                                 args=(t,))).real

        '''Calculem els valors esperats segons l'operador.'''
        if(operator.lower()=='x'):
            return integrate.quad(func=func_x, a=-np.inf, b=+np.inf, args=(t,))[0]
        elif(operator.lower()=='x2'):
            return integrate.quad(func=func_x2, a=-np.inf, b=+np.inf, args=(t,))[0]
        elif (operator.lower() == 'p'):
            return integrate.quad(func=func_p, a=-np.inf, b=+np.inf, args=(t,))[0]
        elif (operator.lower() == 'p2'):
            return integrate.quad(func=func_p2, a=-np.inf, b=+np.inf, args=(t,))[0]
        else:
            sys.exit('El valor de operador ha de ser un string d\'aquests: \'x\', \'x2\', \'p\', \'p2\'')



    ''' Dibuixem la ona en un rang de X i de T. Aprofitem que només hem de reescriure les línies i no els eixos i tota
     la resta de la figura per fer-ho de forma eficient amb blit.'''
    def plot(self, x0=-10, xf=10, t0=0, tf=10, nx=480, nt=100):
        '''Valors de les X i T i valors inicials de la funció d'ona Y.'''
        X = np.linspace(x0, xf, nx)
        T = np.linspace(t0, tf, nt)
        V = 1./2.*self.k*X**2
        prefix = '[Funció Ona] Calculant valors de l\'ona i valors esperats:'
        sufix = 'Acabat'
        print_progress(0, 5, prefix=prefix, suffix=sufix, bar_length=50)

        '''Calculem tots els valors i mostrem el progrés per pantalla'''
        Y = [np.array([self.eval(x=x, t=t) for x in X]) for t in T]
        print_progress(1, 5, prefix=prefix, suffix=sufix, bar_length=50)

        EXP_VAL_X = np.array([self.expected_value(operator='x', t=t) for t in T]) / (math.sqrt(self.m*self.omega))
        print_progress(2, 5, prefix=prefix, suffix=sufix, bar_length=50)

        STD_X = np.sqrt(np.array([self.expected_value(operator='x2', t=t) for t in T]) - EXP_VAL_X ** 2) / \
                (math.sqrt(self.m*self.omega))
        print_progress(3, 5, prefix=prefix, suffix=sufix, bar_length=50)

        EXP_VAL_P = np.array([self.expected_value(operator='p', t=t) for t in T]) * (math.sqrt(self.m*self.omega))
        print_progress(4, 5, prefix=prefix, suffix=sufix, bar_length=50)

        STD_P = np.sqrt(np.array([self.expected_value(operator='p2', t=t) for t in T]) - EXP_VAL_P ** 2) * \
                (math.sqrt(self.m*self.omega))
        print_progress(5, 5, prefix=prefix, suffix=sufix, bar_length=50)

        '''Definició/Inicialització de tots els plots a les 2 differents subfigures. 1a: Plot XY, 2a: Plot XP.'''
        fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
        fig.set_size_inches(w=17,h=16./9.*17,forward=True)
        ax1.set_xlim(np.min(X), np.max(X))
        ylim = 1.25*np.max([abs(np.min(Y[0])), abs(np.max(Y[0]))])
        ax1.set_ylim(-ylim, ylim)

        '''Inicialització plot XY'''
        ax1.grid(True, which='both')
        linia_real = ax1.plot(X, Y[0].real, 'b', label=r'$Re[\psi(x,t)]$', animated=True)[0]
        linia_imag = ax1.plot(X, Y[0].imag, 'r', label=r'$Im[\psi(x,t)]$', animated=True)[0]
        linia_prob = ax1.plot(X, abs(Y[0])**2, 'y', label=r'$|\psi(x,t)|^2$', animated=True)[0]
        linia_exp = ax1.plot([EXP_VAL_X[0],EXP_VAL_X[0]], [-ylim,ylim], 'k', label=r'$\langle x \rangle_\psi(t)$',
                             animated=True)[0]
        errobj = ax1.errorbar(EXP_VAL_X[0], ylim/4., color='k', xerr=STD_X[0], yerr=0, label=r'$\Delta x(t)$',
                              animated=True)
        linia_pot = ax1.plot(X, V-ylim, 'g', label=r'$V(x)$', animated=False)[0]
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        lines = [linia_real, linia_imag, linia_prob, linia_exp, linia_pot]

        '''Inicialització plot XP'''
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k')
        ax2.axvline(x=0, color='k')
        xlim_xp = max(abs(np.min(EXP_VAL_X-STD_X)), abs(np.max(EXP_VAL_X+STD_X)))*1.25
        ylim_xp = max(abs(np.min(EXP_VAL_P-STD_P)), abs(np.max(EXP_VAL_P+STD_P)))*1.25
        lim_xp = max(xlim_xp,ylim_xp)
        ax2.set_xlim(-lim_xp, lim_xp)
        ax2.set_ylim(-lim_xp, lim_xp)
        errobj_xp = ax2.errorbar(EXP_VAL_X[0], EXP_VAL_P[0], color='r', xerr=STD_X[0], yerr=STD_P[0],
                             label=r'$(\Delta x(t),\Delta p(t))$',
                             animated=True)
        ellipse = Ellipse(xy=(EXP_VAL_X[0], EXP_VAL_P[0]), width=2. * STD_X[0], height=2. * STD_P[0], color='c',
                          alpha=0.5)
        heisenberg = ax2.text(xlim_xp*0.6, -ylim_xp*0.9, r'$\Delta x\Delta p=$' + "{0:.2f}".format(STD_X[0]*STD_P[0]),
                              fontsize=12)
        ax2.add_patch(ellipse)
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles, labels)



        '''Funció auxiliar per computar la animació. Avalua per cada temps T la funció d'ona Y als X donats,
        el seu modul al quadrat, els valors esperats i std al plot XY i al XP....'''
        def animate(n):
            '''Plot XY'''
            lines[0].set_ydata(Y[n].real)
            lines[1].set_ydata(Y[n].imag)
            lines[2].set_ydata(abs(Y[n])**2)
            lines[3].set_xdata([EXP_VAL_X[n],EXP_VAL_X[n]])
            lines_err = adjust_err_bar(errobj=errobj,x=EXP_VAL_X[n],y=ylim/4.,x_error=STD_X[n],y_error=0)
            '''Plot XP'''
            # patches = [ax2.add_patch(Ellipse(xy=(EXP_VAL_X[n],EXP_VAL_P[n]), width=2.*STD_X[n], height=2.*STD_P[n],
            # color='c', alpha=0.5))]
            ellipse.center = (EXP_VAL_X[n], EXP_VAL_P[n])
            ellipse.width = 2.*STD_X[n]
            ellipse.height = 2.*STD_P[n]
            lines_err_xp = adjust_err_bar(errobj=errobj_xp,x=EXP_VAL_X[n],y=EXP_VAL_P[n],x_error=STD_X[n],
                                          y_error=STD_P[n])
            heisenberg.set_text(r'$\Delta x\Delta p=$' + "{0:.2f}".format(STD_X[n]*STD_P[n]))
            '''Tornem tot el que volem actualitzar. La resta de la figura es mantindrà intacta per estalviar recursos.'''
            return lines + lines_err + lines_err_xp + [ellipse] + [heisenberg]

        '''Animació eficient.'''
        ani = animation.FuncAnimation(fig, animate, range(len(T)),
                                      interval=100, blit=True, repeat=True)
        plt.show()
        return ani, plt


'''Main per fer petites proves'''
if __name__ == '__main__':
    xi = -10
    xf = 10
    ti = 0
    tf = 10
    nx = 480
    nt = 100

    '''Per començar, definim la nostra funció inicial amb els coeficients. Podem escollir altres coeficients també,
    per exemple [1,1] per la combinació 1/sqrt(2)*(psi_0 + psi_1).'''
    coeffs = np.array([1])
    estat = Estat(coeffs=coeffs, m=M, k=K)
    #
    '''Animació de l'evolució de la funció d'ona. Es poden guardar les animacions fàcilment com un video amb
    ani.save('nom_fitxer').'''
    ani0, _ = estat.ona.plot(x0=xi,xf=xf,t0=ti,tf=tf,nx=nx,nt=nt)
    # ani0.save('ona0.mp4',bitrate=6500)

    '''Aplicar un operador es tan senzill com fer estat.operador(valor0).'''
    # estat.traslacio(x0=3)
    # ani_traslacio, _ = estat.ona.plot(x0=xi,xf=xf,t0=ti,tf=tf,nx=nx,nt=nt)
    # ani_traslacio.save('ona_trasl.mp4', bitrate=6500)
    # plot_ehrenfest(estat, t0=ti, tf=tf, nt=nt)
    estat.kick(p0=3)
    ani, _ = estat.ona.plot(x0=xi,xf=xf,t0=ti,tf=tf,nx=nx,nt=nt)
    # ani.save('ona_kick.mp4', bitrate=6500)
    '''Podem, a més del propi plot de l'evolució de l'ona al espai habitual i al espai x-p, imprimir
    la comprovació del teorema d'Ehrenfest, imprimir els valors esperats i les indeterminacions, i imprimir també
    el valor de la incertesa de Heisenberg (delta_x*delta_p).'''
    plot_ehrenfest(estat, t0=ti, tf=tf, nt=nt)
    plot_valoresp(estat, t0=ti, tf=tf, nt=nt)
    plot_heisenberg(estat, t0=ti, tf=tf, nt=nt)

    '''Podem fer una traslació i un kick simultanis també.'''
    # estat.traslacio_kick(x0=3,p0=3)
    # ani, _ = estat.ona.plot(x0=xi, xf=xf, t0=ti, tf=tf, nx=nx, nt=nt)
    # ani.save('ona_kick_traslacio.mp4', bitrate=6500)
    # plot_ehrenfest(estat, t0=ti, tf=tf, nt=nt)

    '''Si no coneixem o no necessitem els coeficients inicials (com en el cas de la secció 4), podem definir l'estat
    a partir d'una funció fx(x). Opcionalment, es poden enviar arguments extra a la funció (la x ha d'anar al inici de
    fx(x,*args) SEMPRE! i NO la enviem aquí, es tracta internament).'''
    # fx_args = (p0,x0,m,k,xi) en aquest ordre!!
    (p0,x0,m,k,xi_scale) = (2, 0, M, K, 1)
    estat = Estat(m=M,k=K,fx=psi_squeezed,fx_args=(p0,x0,m,k,xi_scale))
    # ani, _ = estat.ona.plot(x0=xi,xf=xf,t0=ti,tf=tf,nx=nx,nt=nt)
    # plot_ehrenfest(estat, t0=ti, tf=tf, nt=nt)
    # plot_valoresp(estat, t0=ti, tf=tf, nt=nt)
    plot_heisenberg(estat, t0=ti, tf=tf, nt=nt)
    # ani.save('estat_squeezed_kick_xi0_heisenberg.mp4',bitrate=6500)



    print('Sortida Correcta')
