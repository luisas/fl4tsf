# -*- coding: utf-8 -*-
import math

import numpy as np
import pandas as pd
import pywt
from numba import njit

from ..misc import check_random_state, check_random_state_children
from ..signal import signal_distort, signal_resample


def ecg_simulate(
    duration=10,
    length=None,
    sampling_rate=1000,
    noise=0.01,
    heart_rate=70,
    heart_rate_std=1,
    method="ecgsyn",
    random_state=None,
    random_state_distort="spawn",
    morphology_params=None,
    **kwargs,
):
    """**Simulate an ECG/EKG signal**

    Generate an artificial (synthetic) ECG signal of a given duration and sampling rate using either
    the ECGSYN dynamical model (McSharry et al., 2003) or a simpler model based on Daubechies
    wavelets to roughly approximate cardiac cycles.

    Parameters
    ----------
    duration : int
        Desired recording length in seconds.
    sampling_rate : int
        The desired sampling rate (in Hz, i.e., samples/second).
    length : int
        The desired length of the signal (in samples).
    noise : float
        Noise level (amplitude of the laplace noise).
    heart_rate : int
        Desired simulated heart rate (in beats per minute). The default is 70. Note that for the
        ``"ECGSYN"`` method, random fluctuations are to be expected to mimick a real heart rate.
        These fluctuations can cause some slight discrepancies between the requested heart rate and
        the empirical heart rate, especially for shorter signals.
    heart_rate_std : int
        Desired heart rate standard deviation (beats per minute).
    method : str
        The model used to generate the signal. Can be ``"simple"`` for a simulation based on
        Daubechies wavelets that roughly approximates a single cardiac cycle. If ``"ecgsyn"``
        (default), will use the model desbribed `McSharry et al. (2003)
        <https://physionet.org/content/ecgsyn/>`_. If
        ``"multileads"``, will return a DataFrame containing 12-leads (see `12-leads ECG simulation
        <https://neuropsychology.github.io/NeuroKit/examples/ecg_generate_12leads/ecg_generate_12leads.html>`_).
    random_state : None, int, numpy.random.RandomState or numpy.random.Generator
        Seed for the random number generator. See for ``misc.check_random_state`` for further information.
    random_state_distort : {'legacy', 'spawn'}, None, int, numpy.random.RandomState or numpy.random.Generator
        Random state to be used to distort the signal. If ``"legacy"``, use the same random state used to
        generate the signal (discouraged as it creates dependent random streams). If ``"spawn"``, spawn
        independent children random number generators from the random_state argument. If any of the other types,
        generate independent children random number generators from the random_state_distort provided (this
        allows generating multiple version of the same signal distorted by different random noise realizations).
    morphology_params : dict, optional
        Custom morphology parameters to control ECG wave shapes. Should contain keys:
        - 'ti': array of angles of extrema (in degrees) for P, Q, R, S, T waves
        - 'ai': array of z-position of extrema for P, Q, R, S, T waves  
        - 'bi': array of Gaussian width of peaks for P, Q, R, S, T waves
        If None, uses default parameters for normal ECG morphology.
    **kwargs
        Other keywords parameters for ECGSYN algorithm, such as ``"lfhfratio"``.

    Returns
    -------
    array
        Vector containing the ECG signal.

    Examples
    ----------

    * **Example 1:** Simulate single lead ECG

    .. ipython:: python

      import neurokit2 as nk

      ecg1 = nk.ecg_simulate(duration=10, method="simple")
      ecg2 = nk.ecg_simulate(duration=10, method="ecgsyn")

      # Visualize result
      @savefig p_ecg_simulate1.png scale=100%
      nk.signal_plot([ecg1, ecg2], labels=["simple", "ecgsyn"], subplots=True)
      @suppress
      plt.close()

    * **Example 2:** Simulate with custom morphology (Long QT)

    .. ipython:: python
    
      # Custom parameters for Long QT syndrome (with frozen morphology)
      long_qt_params = {
          'ti': np.array([-70, -15, 0, 15, 130]),  # Extended T-wave timing
          'ai': np.array([1.2, -5, 30, -7.5, 0.75]),  # Default amplitudes
          'bi': np.array([0.25, 0.1, 0.1, 0.1, 0.6]),   # Wider T-wave
          'freeze_morphology': True  # Preserve pathological timing regardless of HR
      }
      
      ecg_long_qt = nk.ecg_simulate(duration=10, morphology_params=long_qt_params)
      
      # Or use the helper function
      ecg_long_qt2 = nk.ecg_simulate(duration=10, morphology_params=create_long_qt_params(1.4))

    See Also
    --------
    .rsp_simulate, .eda_simulate, .ppg_simulate, .emg_simulate


    References
    -----------
    * McSharry, P. E., Clifford, G. D., Tarassenko, L., & Smith, L. A. (2003). A dynamical model for
      generating synthetic electrocardiogram signals. IEEE transactions on biomedical engineering,
      50 (3), 289-294.

    """
    # Seed the random generator for reproducible results
    rng = check_random_state(random_state)

    # Generate number of samples automatically if length is unspecified
    if length is None:
        length = duration * sampling_rate
    if duration is None:
        duration = length / sampling_rate

    # Run appropriate method
    if method.lower() in ["simple", "daubechies"]:
        signals = _ecg_simulate_daubechies(
            duration=duration, length=length, sampling_rate=sampling_rate, heart_rate=heart_rate
        )
    else:
        approx_number_beats = int(np.round(duration * (heart_rate / 60)))
        if method.lower() in ["multi", "multilead", "multileads", "multichannel"]:
            # Gamma, a (12,5) matrix to modify the five waves' amplitudes of 12 leads (P, Q, R, S, T)
            gamma = np.array(
                [
                    [1, 0.1, 1, 1.2, 1],
                    [2, 0.2, 0.2, 0.2, 3],
                    [1, -0.1, -0.8, -1.1, 2.5],
                    [-1, -0.05, -0.8, -0.5, -1.2],
                    [0.05, 0.05, 1, 1, 1],
                    [1, -0.05, -0.1, -0.1, 3],
                    [-0.5, 0.05, 0.2, 0.5, 1],
                    [0.05, 0.05, 1.3, 2.5, 2],
                    [1, 0.05, 1, 2, 1],
                    [1.2, 0.05, 1, 2, 2],
                    [1.5, 0.1, 0.8, 1, 2],
                    [1.8, 0.05, 0.5, 0.1, 2],
                ]
            )

            signals, results = _ecg_simulate_ecgsyn(
                sfecg=sampling_rate,
                N=approx_number_beats,
                hrmean=heart_rate,
                hrstd=heart_rate_std,
                sfint=sampling_rate,
                gamma=gamma,
                morphology_params=morphology_params,
                rng=rng,
                **kwargs,
            )
        else:
            signals, results = _ecg_simulate_ecgsyn(
                sfecg=sampling_rate,
                N=approx_number_beats,
                hrmean=heart_rate,
                hrstd=heart_rate_std,
                sfint=sampling_rate,
                gamma=np.ones((1, 5)),
                morphology_params=morphology_params,
                rng=rng,
                **kwargs,
            )
        # Cut to match expected length
        for i in range(len(signals)):
            signals[i] = signals[i][0:length]

    # Add random noise
    if noise > 0:
        # Seed for random noise
        random_state_distort = check_random_state_children(random_state, random_state_distort, n_children=len(signals))
        # Call signal_distort on each signal
        for i in range(len(signals)):
            signals[i] = signal_distort(
                signals[i],
                sampling_rate=sampling_rate,
                noise_amplitude=noise,
                noise_frequency=[5, 10, 100],
                noise_shape="laplace",
                random_state=random_state_distort[i],
                silent=True,
            )

    # Format
    if len(signals) == 1:
        ecg = signals[0]
    else:
        ecg = pd.DataFrame(
            np.array(signals).T,
            columns=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        )

    return ecg


# =============================================================================
# Numba-Optimized ODE Solver
# =============================================================================

@njit(fastmath=True, cache=True)
def _rk4_solver_numba(x0, t_eval, dt, rr, ti, sfint, ai, bi):
    """
    Fast, Numba-JIT compiled, fixed-step 4th-order Runge-Kutta solver.
    Optimized specifically for the ECG derivs function.
    """
    n_steps = len(t_eval)
    x = np.zeros((len(x0), n_steps), dtype=np.float64)
    x[:, 0] = x0

    for k in range(n_steps - 1):
        t = t_eval[k]
        x_k = x[:, k]
        
        k1 = _ecg_simulate_derivsecgsyn_numba(t, x_k, rr, ti, sfint, ai, bi)
        k2 = _ecg_simulate_derivsecgsyn_numba(t + 0.5*dt, x_k + 0.5*dt*k1, rr, ti, sfint, ai, bi)
        k3 = _ecg_simulate_derivsecgsyn_numba(t + 0.5*dt, x_k + 0.5*dt*k2, rr, ti, sfint, ai, bi)
        k4 = _ecg_simulate_derivsecgsyn_numba(t + dt, x_k + dt*k3, rr, ti, sfint, ai, bi)

        x[:, k+1] = x_k + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return x


@njit(fastmath=True, cache=True)
def _ecg_simulate_derivsecgsyn_numba(t, x, rr, ti, sfint, ai, bi):
    """Numba-JIT compiled derivative function for ECG simulation."""
    ta = math.atan2(x[1], x[0])
    r0 = 1.0
    a0 = 1.0 - math.sqrt(x[0] * x[0] + x[1] * x[1]) / r0

    ip = int(math.floor(t * sfint))
    if ip >= len(rr):
        ip = len(rr) - 1
    w0 = 2 * math.pi / rr[ip]

    fresp = 0.25
    zbase = 0.005 * math.sin(2 * math.pi * fresp * t)

    dx1dt = a0 * x[0] - w0 * x[1]
    dx2dt = a0 * x[1] + w0 * x[0]

    # Calculate dti efficiently
    dx3dt = 0.0
    for i in range(len(ti)):
        dti = (ta - ti[i]) - round((ta - ti[i]) / (2 * math.pi)) * 2 * math.pi
        dx3dt -= ai[i] * dti * math.exp(-0.5 * (dti / bi[i]) ** 2)
    
    dx3dt -= 1.0 * (x[2] - zbase)

    return np.array([dx1dt, dx2dt, dx3dt])


# =============================================================================
# Daubechies
# =============================================================================
def _ecg_simulate_daubechies(duration=10, length=None, sampling_rate=1000, heart_rate=70):
    """Generate an artificial (synthetic) ECG signal of a given duration and sampling rate.

    It uses a 'Daubechies' wavelet that roughly approximates a single cardiac cycle.
    This function is based on `this script <https://github.com/diarmaidocualain/ecg_simulation>`_.

    """
    # The "Daubechies" wavelet is a rough approximation to a real, single, cardiac cycle
    cardiac = np.array(pywt.Wavelet("db10").rec_lo)

    # Add the gap after the pqrst when the heart is resting.
    cardiac = np.concatenate([cardiac, np.zeros(10)])

    # Calculate the number of beats in capture time period
    num_heart_beats = int(duration * heart_rate / 60)

    # Concatenate together the number of heart beats needed
    ecg = np.tile(cardiac, num_heart_beats)

    # Change amplitude
    ecg = ecg * 10

    # Resample
    ecg = signal_resample(
        ecg,
        sampling_rate=int(len(ecg) / 10),
        desired_length=length,
        desired_sampling_rate=sampling_rate,
    )

    # Return the signal in a list to match
    # with the potential multichannel output of ecgsyn
    return [ecg]


# =============================================================================
# ECGSYN with Numba Optimization
# =============================================================================
def _ecg_simulate_ecgsyn(
    sfecg=256,
    N=256,
    hrmean=60,
    hrstd=1,
    lfhfratio=0.5,
    sfint=512,
    ti=(-70, -15, 0, 15, 100),
    ai=(1.2, -5, 30, -7.5, 0.75),
    bi=(0.25, 0.1, 0.1, 0.1, 0.4),
    gamma=np.ones((1, 5)),
    morphology_params=None,
    rng=None,
    **kwargs,
):
    """
    High-performance ECG simulation using Numba-optimized ODE solver.
    
    This function is a python translation of the matlab script by `McSharry & Clifford (2013)
    <https://physionet.org/content/ecgsyn>`_, optimized with Numba JIT compilation.

    Parameters
    ----------
    sfecg:
        ECG sampling frequency [256 Hertz]
    N:
        approximate number of heart beats [256]
    hrmean:
        Mean heart rate [60 beats per minute]
    hrstd:
        Standard deviation of heart rate [1 beat per minute]
    lfhfratio:
        LF/HF ratio [0.5]
    sfint:
        Internal sampling frequency [256 Hertz]
    ti:
        angles of extrema (in degrees). Order of extrema is (P Q R S T).
    ai:
        z-position of extrema.
    bi:
        Gaussian width of peaks.
    gamma:
        This determines the different leads.
    morphology_params:
        Custom morphology parameters dictionary with keys 'ti', 'ai', 'bi'.
        Set 'freeze_morphology': True to skip HR-dependent scaling for pathological conditions.

    Returns
    -------
    tuple
        (signals, results) where signals is a list of ECG signals
    """

    # Convert default parameters to numpy arrays
    if not isinstance(ti, np.ndarray):
        ti = np.array(ti, dtype=np.float64)
    if not isinstance(ai, np.ndarray):
        ai = np.array(ai, dtype=np.float64)
    if not isinstance(bi, np.ndarray):
        bi = np.array(bi, dtype=np.float64)

    # Handle custom morphology parameters BEFORE HR scaling
    freeze_morphology = False
    if morphology_params is not None:
        ti_user = morphology_params.get("ti", ti)
        ai_user = morphology_params.get("ai", ai) 
        bi_user = morphology_params.get("bi", bi)
        freeze_morphology = morphology_params.get("freeze_morphology", False)
        
        # Convert to numpy arrays
        if not isinstance(ti_user, np.ndarray):
            ti_user = np.array(ti_user, dtype=np.float64)
        if not isinstance(ai_user, np.ndarray):
            ai_user = np.array(ai_user, dtype=np.float64)
        if not isinstance(bi_user, np.ndarray):
            bi_user = np.array(bi_user, dtype=np.float64)
    else:
        ti_user, ai_user, bi_user = ti, ai, bi

    # Convert angles from degrees to radians
    ti_user = ti_user * np.pi / 180

    # Apply HR-dependent scaling only if not frozen (for pathological conditions)
    if not freeze_morphology:
        hrfact = np.sqrt(hrmean / 60)
        hrfact2 = np.sqrt(hrfact)
        bi_user = hrfact * bi_user
        ti_user = np.array([hrfact2, hrfact, 1, hrfact, hrfact2], dtype=np.float64) * ti_user

    # Check that sfint is an integer multiple of sfecg
    q = int(np.round(sfint / sfecg))
    qd = sfint / sfecg
    if q != qd:
        raise ValueError(
            f"Internal sampling frequency (sfint) must be an integer multiple of the ECG sampling frequency"
            f" (sfecg). Your current choices are: sfecg = {sfecg} and sfint = {sfint}."
        )

    # Define frequency parameters for rr process
    # flo and fhi correspond to the Mayer waves and respiratory rate respectively
    flo = 0.1
    fhi = 0.25
    flostd = 0.01
    fhistd = 0.01

    # Calculate time scales for rr and total output
    sfrr = 1
    trr = 1 / sfrr
    rrmean = 60 / hrmean
    n = 2 ** (np.ceil(np.log2(N * rrmean / trr)))

    rr0 = _ecg_simulate_rrprocess(flo, fhi, flostd, fhistd, lfhfratio, hrmean, hrstd, sfrr, n, rng)

    # Upsample rr time series from 1 Hz to sfint Hz
    rr = signal_resample(rr0, sampling_rate=1, desired_sampling_rate=sfint)

    # Make the rrn time series
    dt = 1 / sfint
    rrn = np.zeros(len(rr), dtype=np.float64)
    tecg = 0
    i = 0
    while i < len(rr):
        tecg += rr[i]
        ip = int(np.round(tecg / dt))
        if ip >= len(rrn):
            ip = len(rrn) - 1
        rrn[i:ip] = rr[i]
        i = ip
    Nt = ip

    # Prepare for ODE integration
    x0 = np.array([1.0, 0.0, 0.04], dtype=np.float64)
    t_eval = np.linspace(0, (Nt - 1) * dt, Nt, dtype=np.float64)

    # Initialize results containers
    results = []
    signals = []

    # Loop over leads
    for lead in range(len(gamma)):
        # Prepare lead-specific parameters
        ai_lead = gamma[lead] * ai_user
        
        # Solve using fast Numba-compiled RK4 solver
        X0 = _rk4_solver_numba(x0, t_eval, dt, rrn, ti_user, sfint, ai_lead, bi_user)
        
        # Store results (dummy for compatibility)
        results.append({"y": X0})
        
        # Downsample to required sfecg
        X = X0[:, np.arange(0, X0.shape[1], q).astype(int)]

        # Scale signal to lie between -0.4 and 1.2 mV
        z = X[2, :].copy()
        zmin = np.min(z)
        zmax = np.max(z)
        zrange = zmax - zmin
        if zrange > 0:
            z = (z - zmin) * 1.6 / zrange - 0.4
        else:
            z = np.zeros_like(z) - 0.4

        signals.append(z)

    return signals, results


def _ecg_simulate_rrprocess(
    flo=0.1,
    fhi=0.25,
    flostd=0.01,
    fhistd=0.01,
    lfhfratio=0.5,
    hrmean=60,
    hrstd=1,
    sfrr=1,
    n=256,
    rng=None,
):
    """Generate RR interval process for heart rate variability."""
    w1 = 2 * np.pi * flo
    w2 = 2 * np.pi * fhi
    c1 = 2 * np.pi * flostd
    c2 = 2 * np.pi * fhistd
    sig2 = 1
    sig1 = lfhfratio
    rrmean = 60 / hrmean
    rrstd = 60 * hrstd / (hrmean * hrmean)

    df = sfrr / n
    w = np.arange(n) * 2 * np.pi * df
    dw1 = w - w1
    dw2 = w - w2

    Hw1 = sig1 * np.exp(-0.5 * (dw1 / c1) ** 2) / np.sqrt(2 * np.pi * c1 ** 2)
    Hw2 = sig2 * np.exp(-0.5 * (dw2 / c2) ** 2) / np.sqrt(2 * np.pi * c2 ** 2)
    Hw = Hw1 + Hw2
    Hw0 = np.concatenate((Hw[0 : int(n / 2)], Hw[int(n / 2) - 1 :: -1]))
    Sw = (sfrr / 2) * np.sqrt(Hw0)

    ph0 = 2 * np.pi * rng.uniform(size=int(n / 2 - 1))
    ph = np.concatenate([[0], ph0, [0], -np.flipud(ph0)])
    SwC = Sw * np.exp(1j * ph)
    x = (1 / n) * np.real(np.fft.ifft(SwC))

    xstd = np.std(x)
    if xstd > 0:
        ratio = rrstd / xstd
        return rrmean + x * ratio
    else:
        return np.full(len(x), rrmean)


# =============================================================================
# Helper Functions for Parameter Modification
# =============================================================================

def create_long_qt_params(qt_prolongation_factor=1.3, freeze_morphology=True):
    """
    Create morphology parameters for Long QT syndrome.
    
    Parameters
    ----------
    qt_prolongation_factor : float
        Factor by which to prolong the QT interval (default 1.3 = 30% longer)
    freeze_morphology : bool
        If True, skip HR-dependent scaling to preserve pathological timing
        
    Returns
    -------
    dict
        Morphology parameters with prolonged T-wave
    """
    return {
        'ti': np.array([-70, -15, 0, 15, 100 * qt_prolongation_factor]),
        'ai': np.array([1.2, -5, 30, -7.5, 0.75]),
        'bi': np.array([0.25, 0.1, 0.1, 0.1, 0.4 * qt_prolongation_factor]),
        'freeze_morphology': freeze_morphology
    }


def create_short_qt_params(qt_shortening_factor=0.7, freeze_morphology=True):
    """
    Create morphology parameters for Short QT syndrome.
    
    Parameters
    ----------
    qt_shortening_factor : float
        Factor by which to shorten the QT interval (default 0.7 = 30% shorter)
    freeze_morphology : bool
        If True, skip HR-dependent scaling to preserve pathological timing
        
    Returns
    -------
    dict
        Morphology parameters with shortened T-wave
    """
    return {
        'ti': np.array([-70, -15, 0, 15, 100 * qt_shortening_factor]),
        'ai': np.array([1.2, -5, 30, -7.5, 0.75]),
        'bi': np.array([0.25, 0.1, 0.1, 0.1, 0.4 * qt_shortening_factor]),
        'freeze_morphology': freeze_morphology
    }


def create_high_t_wave_params(t_wave_amplitude_factor=1.5, freeze_morphology=False):
    """
    Create morphology parameters for prominent T-waves (e.g., hyperkalemia).
    
    Parameters
    ----------
    t_wave_amplitude_factor : float
        Factor by which to increase T-wave amplitude
    freeze_morphology : bool
        If False, allows normal HR-dependent scaling (default for amplitude changes)
        
    Returns
    -------
    dict
        Morphology parameters with enhanced T-waves
    """
    return {
        'ti': np.array([-70, -15, 0, 15, 100]),
        'ai': np.array([1.2, -5, 30, -7.5, 0.75 * t_wave_amplitude_factor]),
        'bi': np.array([0.25, 0.1, 0.1, 0.1, 0.4]),
        'freeze_morphology': freeze_morphology
    }