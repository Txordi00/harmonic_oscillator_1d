# coding=utf-8
import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt
import sys

'''Funció per actualitzar la línia d'error.'''
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
    return [ln, errx_top, errx_bot, erry_top, erry_bot, barsx, barsy]

'''Funció per mostrar progressos de còmput en una sola línia de terminal.'''
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 1, bar_length = 100, fill = '█'):
    '''
        iteration   : Iteració actual (int)
        total       : Total iteracions (int)
        prefix      : Frase per mostrar abans (str)
        suffix      : Frase per mostrar després (str)
        decimals    : Nombre de decimals (int)
        bar_length   : Longitud de la barra (int)
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(bar_length * iteration // total)
    bar = fill * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

'''Funció per mostrar les dos igualtats del teorema de Ehrenfest.'''
def plot_ehrenfest(estat, t0=0,tf=10, nt=100):
    m = estat.m
    k = estat.k
    T = np.linspace(t0, tf, nt)

    potential = lambda x: 1./2.*k*x**2

    prefix = '[Ehrenfest] Calculant valors esperats i derivades:'
    sufix = 'Acabat'
    print_progress(0, 5, prefix=prefix, suffix=sufix, bar_length=50)

    X = np.array([estat.valor_esperat(t=t,operator='x') for t in T])
    print_progress(1, 5, prefix=prefix, suffix=sufix, bar_length=50)

    V = np.array([-derivative(func=potential, x0=x,dx=1e-2,n=1) for x in X])
    print_progress(2, 5, prefix=prefix, suffix=sufix, bar_length=50)

    P = np.array([estat.valor_esperat(t=t,operator='p') for t in T])
    print_progress(3, 5, prefix=prefix, suffix=sufix, bar_length=50)

    DX = m*np.array([derivative(func=estat.valor_esperat, x0=t, dx=1e-2, n=1, args=('x',)) for t in T])
    print_progress(4, 5, prefix=prefix, suffix=sufix, bar_length=50)

    DP = np.array([derivative(func=estat.valor_esperat, x0=t, dx=1e-2, n=1, args=('p',)) for t in T])
    print_progress(5, 5, prefix=prefix, suffix=sufix, bar_length=50)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(w=17, h=16. / 9. * 17, forward=True)

    ax1.plot(T, P, 'b', label=r'$\langle p \rangle(t)$')
    ax1.plot(T, DX, 'r', label=r'$m \frac{d \langle x \rangle}{dt}(t) $')
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1)

    ax2.plot(T, DP, 'b', label=r'$ \frac{d \langle p \rangle}{dt}(t) $')
    ax2.plot(T, V, 'r', label=r'$ -\frac{d V(\langle x \rangle)}{dx}(t) $')
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2)

    plt.show()

    return plt

