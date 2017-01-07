# coding=utf-8
import numpy as np

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
