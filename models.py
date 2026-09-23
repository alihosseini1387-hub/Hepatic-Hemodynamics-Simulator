"""
مدل‌های ریاضی شبیه‌سازی همودینامیک کبد
Version: 8.0.0 (Two-Compartment Model)
"""

import numpy as np

mmHg_to_Pa = 133.322
rho_blood = 1060
g = 9.81


def calc_mu_apparent(mu_inf, tau_y, gamma_dot):
    if gamma_dot <= 0:
        return mu_inf
    return mu_inf * (1 + np.sqrt(tau_y / (mu_inf * gamma_dot))) ** 2


def calc_shear_rate(Q, r0):
    Q_sin = Q / 1000000000
    if r0 <= 0:
        return 1.0
    return (4 * Q_sin) / (np.pi * r0 ** 3)


def calc_sinusoid_pressure_drop(Q, mu, L, r0, beta):
    Q_sin = Q / 1000000000
    if beta == 0:
        return (8 * mu * L * Q_sin) / (np.pi * r0 ** 4)
    else:
        return (8 * mu * L * Q_sin) / (3 * np.pi * r0 ** 4 * beta) * (1 / (1 - beta) ** 3 - 1)


def calc_Kf_nonlinear(Kf0, deltaP):
    if deltaP < 12:
        return Kf0
    else:
        exponent = 0.05 * (20 - 11) * ((deltaP - 11) / (20 - 11)) ** 2
        return Kf0 * np.exp(exponent)


def calc_Pi_nonlinear(Pi0, deltaP):
    if deltaP < 12:
        return Pi0
    else:
        return Pi0 * (deltaP - 11) ** (1 / 3)


def calc_Jlymph(Jmax, Km, Pi):
    if Km + Pi == 0:
        return 0
    return (Jmax * Pi) / (Km + Pi)


def calc_alpha(Q_portal, Q_artery, A_portal, A_hepatic, h, r0, beta, L, mu_inf, tau_y):
    Q_total = Q_portal + Q_artery
    vp = Q_portal / A_portal if A_portal > 0 else 0
    vh = Q_total * 0.327 / A_hepatic if A_hepatic > 0 else 0
    gamma_dot = calc_shear_rate(Q_total, r0)
    mu_app = calc_mu_apparent(mu_inf, tau_y, gamma_dot)
    dp_sin = calc_sinusoid_pressure_drop(Q_total, mu_app, L, r0, beta)
    dp_h = rho_blood * g * h
    dp_v = 0.5 * rho_blood * (vh ** 2 - vp ** 2)
    dp_total = dp_sin + dp_h + dp_v
    if dp_total > 0:
        alpha = dp_sin / dp_total
        alpha = np.clip(alpha, 0.1, 0.9)
    else:
        alpha = 0.54
    return alpha, Q_total, vp, vh, dp_sin, dp_h, dp_v, dp_total, mu_app


def calc_Jv(deltaP, Kf, alpha, sigma, Pi, dpi, P_hepatic):
    Pc = (alpha * deltaP) + P_hepatic
    hydrostatic = Pc - Pi
    oncotic = sigma * dpi
    Jv = Kf * (hydrostatic - oncotic)
    return Jv


def calc_Jnet(Jv, Jlymph):
    return Jv - Jlymph


def calc_J_capsule(Kf_capsule_0, deltaP, Pi, P_peritoneum, threshold=12):
    """
    عبور مایع از کپسول گلیسون
    
    J_capsule = Kf_capsule(ΔP) × (Pi - P_peritoneum)
    Kf_capsule(ΔP) = Kf_capsule_0 × max(0, ΔP - 12)
    """
    Kf_capsule = Kf_capsule_0 * max(0, deltaP - threshold)
    driving_force = max(0, Pi - P_peritoneum)
    return Kf_capsule * driving_force


def calc_J_perit_lymph(J_perit_max, K_perit, V_asc):
    """تخلیه لنفاوی صفاق"""
    if (K_perit + V_asc) <= 0:
        return 0
    return (J_perit_max * V_asc) / (K_perit + V_asc)


def predict_ascites_two_compartment(Kf_sinusoid, alpha, sigma, dpi,
                                     P_hepatic, Pi0, k_elastance,
                                     Kf_capsule_0, k_abdominal,
                                     P_perit_0, Jmax, Km,
                                     J_perit_max, K_perit,
                                     deltaP_0, time_weeks,
                                     V_int_0=0.0, V_asc_0=0.0,
                                     dt=0.001):
    """
    مدل دو-کپارتمانه تشکیل آسیت
    
    کمپارتمان ۱: فضای بین‌بافتی کبد (V_int)
    کمپارتمان ۲: حفره صفاقی (V_asc)
    
    معادلات:
    ---------
    dV_int/dt = Jv - Jlymph - J_capsule
    dV_asc/dt = J_capsule - J_perit_lymph
    J_capsule = Kf_capsule_0 × max(0, ΔP - 12) × (Pi - P_peritoneum)
    P_peritoneum = P_perit_0 + k_abdominal × V_asc
    ΔP = ΔP_0 + k_abdominal × V_asc  (حلقه بازخورد)
    
    خروجی:
    -------
    time_array, V_int_array, V_asc_array, Jv_array, Jlymph_array,
    Jcapsule_array, Jperit_array, Pi_array, Pperit_array, deltaP_array
    """
    time_min = time_weeks * 7 * 24 * 60
    dt_min = dt * 7 * 24 * 60
    
    n_steps = int(time_min / dt_min)
    if n_steps < 1:
        n_steps = 1
    
    time_array = np.linspace(0, time_weeks, n_steps + 1)
    V_int_array = np.zeros(n_steps + 1)
    V_asc_array = np.zeros(n_steps + 1)
    Jv_array = np.zeros(n_steps + 1)
    Jlymph_array = np.zeros(n_steps + 1)
    Jcapsule_array = np.zeros(n_steps + 1)
    Jperit_array = np.zeros(n_steps + 1)
    Pi_array = np.zeros(n_steps + 1)
    Pperit_array = np.zeros(n_steps + 1)
    deltaP_array = np.zeros(n_steps + 1)
    
    V_int_array[0] = V_int_0
    V_asc_array[0] = V_asc_0
    Pi_array[0] = Pi0
    Pperit_array[0] = P_perit_0
    deltaP_array[0] = deltaP_0
    
    for i in range(n_steps):
        V_int = V_int_array[i]
        V_asc = V_asc_array[i]
        
        # فشار پورتال (حلقه بازخورد)
        deltaP = deltaP_0 + k_abdominal * V_asc
        deltaP_array[i] = deltaP
        
        # فشار سینوزوئیدی
        Pc = alpha * deltaP + P_hepatic
        
        # فشار بین‌بافتی
        Pi = Pi0 + k_elastance * V_int
        Pi_array[i] = Pi
        
        # Kf غیرخطی
        Kf_eff = calc_Kf_nonlinear(Kf_sinusoid, deltaP)
        
        # Jv
        Jv = Kf_eff * ((Pc - Pi) - sigma * dpi)
        Jv = max(0, Jv)
        Jv_array[i] = Jv
        
        # Jlymph
        Jlymph = calc_Jlymph(Jmax, Km, Pi)
        Jlymph_array[i] = Jlymph
        
        # فشار صفاقی
        P_perit = P_perit_0 + k_abdominal * V_asc
        Pperit_array[i] = P_perit
        
        # J_capsule
        J_capsule = calc_J_capsule(Kf_capsule_0, deltaP, Pi, P_perit)
        Jcapsule_array[i] = J_capsule
        
        # J_perit_lymph
        J_perit = calc_J_perit_lymph(J_perit_max, K_perit, V_asc)
        Jperit_array[i] = J_perit
        
        # به‌روزرسانی دو کمپارتمان
        dV_int = (Jv - Jlymph - J_capsule) * dt_min
        dV_asc = (J_capsule - J_perit) * dt_min
        
        V_int_array[i + 1] = max(0, V_int + dV_int)
        V_asc_array[i + 1] = max(0, V_asc + dV_asc)
    
    return (time_array, V_int_array, V_asc_array,
            Jv_array, Jlymph_array, Jcapsule_array,
            Jperit_array, Pi_array, Pperit_array, deltaP_array)


def analyze_system(deltaP, params):
    Kf0 = params['Kf0']
    alpha = params['alpha']
    sigma = params['sigma']
    Pi0 = params['Pi0']
    Jmax = params.get('Jmax', 30)
    Km = params.get('Km', 0.5)
    dpi = params['dPi']
    Kf = calc_Kf_nonlinear(Kf0, deltaP)
    Pi = calc_Pi_nonlinear(Pi0, deltaP)
    P_hep = 4
    Jv = calc_Jv(deltaP, Kf, alpha, sigma, Pi, dpi, P_hep)
    Jlymph = calc_Jlymph(Jmax, Km, Pi)
    Jnet = calc_Jnet(Jv, Jlymph)
    if Jnet <= 0:
        status = "تخلیه کامل"
        risk = "کم"
    elif Jnet < 5:
        status = "تجمع خفیف"
        risk = "متوسط"
    elif Jnet < 15:
        status = "تجمع متوسط"
        risk = "بالا"
    else:
        status = "تجمع شدید"
        risk = "بسیار بالا"
    return {'deltaP': deltaP, 'Kf': Kf, 'Pi': Pi, 'Jv': Jv,
            'Jlymph': Jlymph, 'Jnet': Jnet, 'status': status, 'risk': risk}


def analyze_system_range(deltaP_range, params):
    results = {'deltaP': deltaP_range, 'Kf': [], 'Pi': [], 'Jv': [], 'Jlymph': [], 'Jnet': []}
    for dp in deltaP_range:
        res = analyze_system(dp, params)
        results['Kf'].append(res['Kf'])
        results['Pi'].append(res['Pi'])
        results['Jv'].append(res['Jv'])
        results['Jlymph'].append(res['Jlymph'])
        results['Jnet'].append(res['Jnet'])
    return results
