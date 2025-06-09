from iminuit import Minuit
from iminuit.cost import LeastSquares, ExtendedBinnedNLL
from scipy.stats import chi2
import pandas as pd
import numpy as np

def LS_fit(data_x:list, data_y:list, y_err:list, model:'function', disp = 1, **kwrds):
    """
    Fit dei dati con metodo dei minimi quadrati.

    Ritorna in ordine: parametri, valori, errori, p-value, gradi di lib., chi-quadro, matrice di covarianza
    
    Usare x, y, *z = LS_fit(...), dove *z racchiude tutte gli altri valori
    """
    cost_function = LeastSquares(data_x, data_y, y_err, model) # type: ignore

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

def LS_fit_lims(data_x:list, data_y:list, y_err:list, model:'function', limits:list, disp = 1, **kwrds):
    """
    Fit dei dati con metodo dei minimi quadrati.

    A differenza del metodo LS_fit, permette di imporre dei limiti sui parametri.
    I limiti devono essere passati come una lista di liste, dove la prima lista contiene i nomi dei parametri e la seconda i limiti.
    """
    cost_function = LeastSquares(data_x, data_y, y_err, model) # type: ignore

    my_minuit = Minuit(cost_function, **kwrds)

    for i in range(len(limits)):
        my_minuit.limits[[i][0]] = limits[i][1]

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
    cost_function = ExtendedBinnedNLL(bin_content, bin_edges, modello) # type: ignore

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

    for i, j, k in zip(x, dx, dy): # type: ignore
        yl = modello(i - j) # type: ignore
        yr = modello(i + j) # type: ignore
        delta_y = (yr - yl)/2

        sy.append((delta_y**2 + k**2) ** .5)

    return sy

def crop_df(df: pd.DataFrame, N: int = 10, thr = 0) -> pd.DataFrame:
    """
    Remove groups of consecutive numbers under the threshold from the DataFrame.
    """
    is_zero = df.iloc[:,1] <= thr
    group_id = (is_zero != is_zero.shift()).cumsum()
    removal_mask = pd.Series(False, index=df.index)
    for grp, group_indices in df.groupby(group_id).groups.items():
        if is_zero.loc[group_indices[0]] and len(group_indices) > N:
            removal_mask.loc[group_indices] = True
    return df[~removal_mask].copy()

def cut_df(df: pd.DataFrame, sec: tuple) -> pd.DataFrame:
    """
    Rimuove i dati al di fuori del range definito da sec.
    """
    return df[(df.iloc[:,0] >= sec[0]) & (df.iloc[:,0] <= sec[1])].copy()

def assign_errors(df: pd.DataFrame, lim = 30) -> np.ndarray:
    """
    Assegna un errore a ciascun valore in base al valore stesso.
    L'errore è considerato gaussiano se ci sono abbastanza eventi.
    Se il numero di eventi è minore di lim, va riconsiderato.
    """
    ys = list(df.iloc[:,1])
    tot = np.sum(ys)
    ers = np.zeros(len(df))
    minerr = np.sqrt(lim)
    
    i = 0
    for y in ys:
        if y > lim: # type: ignore
            ers[i] = np.sqrt(y) # type: ignore
        else:
            ers[i] = minerr # FIXME
            #ers[i] = np.sqrt(y * y/tot * (1 - y/tot))
        i += 1

    return ers


def read_corretto(path: str, skiprs: int = 65, titles: list = ['ADC', 'Counts']):
    return pd.read_csv(path, sep='\s+', skiprows = skiprs, header=None, encoding= 'ISO-8859-1', names= titles) # type: ignore

#alcune gaussiane utili
def gauss(x, mu, sigma, S):
    return S * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gauss_cost(x, mu, sigma, S, a):
    return gauss(x, mu, sigma, S) + a

def gauss_pol1(x, mu, sigma, S, a, b,):
    return gauss(x, mu, sigma, S) + b*x + a

def gauss_pol2(x, mu, sigma, S, a, b, c):
    return gauss(x, mu, sigma, S) + a + b*x + c*x**2

def gauss_pol3 (x, mu, sigma, S, a, b, c, d):
    return gauss(x, mu, sigma, S) + a + b*x + c*x**2 + d*x**3

def gauss_exp(x, mu, sigma, S, a, lam):
    return gauss(x, mu, sigma, S) + a * np.exp(-lam*x) 

def conv_ADC_Energy(C, psau = 1):
    """
    Prende in entrata il numero di ADC chennels e restituisce l'energia in eV.
    """
    # return (2/50)*(1/2**12)*(4e-9)*1/25*C/1.602176634e-19*3.6 per motivi di efficienza ho calcolato il coefficiente
    if psau: return 35108.489 * C # con psau
    else: return 877712.225 * C # senza psau