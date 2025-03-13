from iminuit import Minuit
from iminuit.cost import LeastSquares, ExtendedBinnedNLL
from scipy.stats import chi2

def LS_fit(data_x:list, data_y:list, y_err:list, model:'function', disp = 1, **kwrds):
    """
    Fit dei dati con metodo dei minimi quadrati.

    Ritorna in ordine: parametri, valori, errori, p-value, gradi di lib., chi-quadro, matrice di covarianza
    
    Usare x, y, *z = LS_fit(...), dove *z racchiude tutte gli altri valori
    """
    cost_function = LeastSquares(data_x, data_y, y_err, model)

    my_minuit = Minuit(cost_function, **kwrds)
    my_minuit.migrad()
    my_minuit.hesse()

    params = my_minuit.parameters
    values = my_minuit.values
    uncert = my_minuit.errors
    chi_quadro = my_minuit.fval
    dof = my_minuit.ndof
    cov = my_minuit.covariance

    pval = 1. - chi2.cdf(chi_quadro, df = dof)

    if disp : display(my_minuit) # type: ignore

    return params, values, uncert, pval, dof, chi_quadro, cov


def Binned_fit(bin_content:list, bin_edges:list, modello:'function', disp = 1, **kwrds):
    """
    Fit di dati da un istogramma.

    Ritorna in ordine: parametri, valori, errori, p-value, gradi di lib., chi-quadro, matrice di covarianza
    """
    cost_function = ExtendedBinnedNLL(bin_content, bin_edges, modello)

    my_minuit = Minuit(cost_function, **kwrds)
    my_minuit.migrad()
    my_minuit.hesse()

    params = my_minuit.parameters
    values = my_minuit.values
    uncert = my_minuit.errors
    chi_quadro = my_minuit.fval
    dof = my_minuit.ndof
    cov = my_minuit.covariance

    pval = 1. - chi2.cdf(chi_quadro, df = dof)

    if disp : display(my_minuit) # type: ignore

    return params, values, uncert, pval, dof, chi_quadro, cov

def TestCompatibilita(x0:float, sigma0:float, x1:float, sigma1:float = 0) -> float:
    """
    Test di compatibilità tra due valori.
    """
    sigma = (sigma0**2 + sigma1**2)**0.5
    z = abs(x0 - x1) / sigma

    return z

def TrasportoErroriX2Y(x:list, dx:float, dy:float, modello:'function') -> list:
    """
    Trasporto degli errori da x a y seguendo il modello nell'approssimazione
    di errori piccoli.
    """
    sy = []

    for i, j, k in zip(x, dx, dy):
        yl = modello(i - j)
        yr = modello(i + j)
        delta_y = (yr - yl)/2

        sy.append((delta_y**2 + k**2) ** .5)

    return sy

