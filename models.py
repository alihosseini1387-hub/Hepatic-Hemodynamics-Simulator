"""
مدل‌های ریاضی شبیه‌سازی همودینامیک کبد
بر اساس مقاله: شبیه‌سازی جریان خون در کبد بر اساس اصول و معادلات مکانیک سیالات
Version: 4.0.0 (Dynamic Ascites Model with Saturation)
"""

import numpy as np

# ======================== ثابت‌های فیزیکی ========================
mmHg_to_Pa = 133.322
rho_blood = 1060
g = 9.81


def calc_mu_apparent(mu_inf, tau_y, gamma_dot):
    """محاسبه ویسکوزیته ظاهری خون با مدل کاسون"""
    if gamma_dot <= 0:
        return mu_inf
    return mu_inf * (1 + np.sqrt(tau_y / (mu_inf * gamma_dot)))**2


def calc_shear_rate(Q, r0):
    """محاسبه نرخ برش در سینوزوئید"""
    Q_sin = Q / 1000000000
    if r0 <= 0:
        return 1.0
    return (4 * Q_sin) / (np.pi * r0**3)


def calc_sinusoid_pressure_drop(Q, mu, L, r0, beta):
    """محاسبه افت فشار سینوزوئید با شعاع متغیر"""
    Q_sin = Q / 1000000000
    if beta == 0:
        return (8 * mu * L * Q_sin) / (np.pi * r0**4)
    else:
        return (8 * mu * L * Q_sin) / (3 * np.pi * r0**4 * beta) * (1 / (1 - beta)**3 - 1)


def calc_Kf_nonlinear(Kf0, deltaP):
    """محاسبه ضریب فیلتراسیون غیرخطی Kf"""
    if deltaP < 12:
        return Kf0
    else:
        exponent = 0.05 * (20 - 11) * ((deltaP - 11) / (20 - 11))**2
        return Kf0 * np.exp(exponent)


def calc_Pi_nonlinear(Pi0, deltaP):
    """محاسبه فشار میان‌بافتی غیرخطی Pi"""
    if deltaP < 12:
        return Pi0
    else:
        return Pi0 * (deltaP - 11)**(1/3)


def calc_Jlymph(Jmax, Km, Pi):
    """محاسبه تخلیه لنفاوی با معادله Michaelis-Menten"""
    if Km + Pi == 0:
        return 0
    return (Jmax * Pi) / (Km + Pi)


def calc_alpha(Q_portal, Q_artery, A_portal, A_hepatic, h, r0, beta, L, mu_inf, tau_y):
    """محاسبه پارامتر α"""
    Q_total = Q_portal + Q_artery
    vp = Q_portal / A_portal if A_portal > 0 else 0
    vh = Q_total * 0.327 / A_hepatic if A_hepatic > 0 else 0
    gamma_dot = calc_shear_rate(Q_total, r0)
    mu_app = calc_mu_apparent(mu_inf, tau_y, gamma_dot)
    dp_sin = calc_sinusoid_pressure_drop(Q_total, mu_app, L, r0, beta)
    dp_h = rho_blood * g * h
    dp_v = 0.5 * rho_blood * (vh**2 - vp**2)
    dp_total = dp_sin + dp_h + dp_v
    if dp_total > 0:
        alpha = dp_sin / dp_total
        alpha = np.clip(alpha, 0.1, 0.9)
    else:
        alpha = 0.54
    return alpha, Q_total, vp, vh, dp_sin, dp_h, dp_v, dp_total, mu_app


def calc_Jv(deltaP, Kf, alpha, sigma, Pi, dpi, P_hepatic):
    """محاسبه شار تراوش"""
    Pc = (alpha * deltaP) + P_hepatic
    hydrostatic = Pc - Pi
    oncotic = sigma * dpi
    Jv = Kf * (hydrostatic - oncotic)
    return Jv


def calc_Jnet(Jv, Jlymph):
    """محاسبه نرخ خالص تجمع"""
    return Jv - Jlymph


# ======================== Pi دینامیک با محدودیت اشباع ========================

def calc_Pi_dynamic(Pi0, k_elastance, V, Pi_max=10.0):
    """
    محاسبه Pi با محدودیت اشباع
    
    در این مدل، Pi با حجم افزایش می‌یابد اما به یک حد ماکسیمم
    (Pi_max) محدود می‌شود. این محدودیت نشان‌دهنده ظرفیت محدود
    بافت کبد برای افزایش فشار بین‌بافتی است.
    """
    Pi = Pi0 + k_elastance * V
    return min(Pi, Pi_max)


# ======================== مدل دینامیک آسیت ========================

def predict_ascites_volume_dynamic(Kf, alpha, sigma, dpi, P_hepatic, Pi0,
                                   k_elastance, Jmax, Km, deltaP,
                                   time_hours, V0=0, dt=0.01, Pi_max=10.0):
    """
    حل معادله دیفرانسیل dV/dt = Jnet(t) با روش Euler
    
    پارامترها:
    -----------
    Kf : ضریب فیلتراسیون (ml/min/mmHg)
    alpha : سهم مقاومت سینوزوئیدی
    sigma : ضریب انعکاس
    dpi : اختلاف فشار انکوتیک (mmHg)
    P_hepatic : فشار ورید هپاتیک (mmHg)
    Pi0 : فشار بین‌بافتی پایه (mmHg)
    k_elastance : ضریب الاستانس بافت (mmHg/mL)
    Jmax : حداکثر ظرفیت لنفاوی (ml/min)
    Km : ثابت Michaelis-Menten (mmHg)
    deltaP : گرادیان فشار پورتال-هپاتیک (mmHg)
    time_hours : زمان شبیه‌سازی (ساعت)
    V0 : حجم اولیه آسیت (mL)
    dt : گام زمانی (ساعت)
    Pi_max : حداکثر فشار بین‌بافتی (mmHg)
    
    خروجی:
    -------
    time_array, V_array, Jnet_array, Pi_array, Jv_array, Jlymph_array
    """
    time_min = time_hours * 60
    dt_min = dt * 60
    n_steps = int(time_min / dt_min)
    if n_steps < 1:
        n_steps = 1
    
    time_array = np.linspace(0, time_hours, n_steps + 1)
    V_array = np.zeros(n_steps + 1)
    Jnet_array = np.zeros(n_steps + 1)
    Pi_array = np.zeros(n_steps + 1)
    Jv_array = np.zeros(n_steps + 1)
    Jlymph_array = np.zeros(n_steps + 1)
    
    V_array[0] = V0
    Pi_array[0] = Pi0
    
    Pc = alpha * deltaP + P_hepatic
    Jv_array[0] = Kf * ((Pc - Pi0) - sigma * dpi)
    Jlymph_array[0] = (Jmax * Pi0) / (Km + Pi0) if (Km + Pi0) > 0 else 0
    Jnet_array[0] = Jv_array[0] - Jlymph_array[0]
    
    for i in range(n_steps):
        # محاسبه Pi با محدودیت اشباع
        Pi = calc_Pi_dynamic(Pi0, k_elastance, V_array[i], Pi_max)
        Pi_array[i] = Pi
        
        Pc = alpha * deltaP + P_hepatic
        Jv = Kf * ((Pc - Pi) - sigma * dpi)
        Jlymph = (Jmax * Pi) / (Km + Pi) if (Km + Pi) > 0 else 0
        Jnet = Jv - Jlymph
        
        Jv_array[i] = Jv
        Jlymph_array[i] = Jlymph
        Jnet_array[i] = Jnet
        
        V_array[i + 1] = V_array[i] + Jnet * dt_min
        V_array[i + 1] = max(0, V_array[i + 1])
    
    # مقدار نهایی
    Pi_final = calc_Pi_dynamic(Pi0, k_elastance, V_array[-1], Pi_max)
    Pc_final = alpha * deltaP + P_hepatic
    Jv_final = Kf * ((Pc_final - Pi_final) - sigma * dpi)
    Jlymph_final = (Jmax * Pi_final) / (Km + Pi_final) if (Km + Pi_final) > 0 else 0
    Jnet_array[-1] = Jv_final - Jlymph_final
    Pi_array[-1] = Pi_final
    Jv_array[-1] = Jv_final
    Jlymph_array[-1] = Jlymph_final
    
    return time_array, V_array, Jnet_array, Pi_array, Jv_array, Jlymph_array


# ======================== تحلیل سیستم ========================

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
        status = "تخلیه کامل - بدون آسیت"
        risk = "کم"
    elif Jnet < 5:
        status = "تجمع خفیف - خطر پایین آسیت"
        risk = "متوسط"
    elif Jnet < 15:
        status = "تجمع متوسط - خطر آسیت"
        risk = "بالا"
    else:
        status = "تجمع شدید - خطر بالای آسیت"
        risk = "بسیار بالا"
    
    return {
        'deltaP': deltaP, 'Kf': Kf, 'Pi': Pi, 'Jv': Jv,
        'Jlymph': Jlymph, 'Jnet': Jnet, 'status': status, 'risk': risk
    }


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
