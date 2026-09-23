"""
مدل‌های ریاضی شبیه‌سازی همودینامیک کبد
بر اساس مقاله: شبیه‌سازی جریان خون در کبد بر اساس اصول و معادلات مکانیک سیالات
"""

import numpy as np

# ======================== ثابت‌های فیزیکی ========================
mmHg_to_Pa = 133.322  # تبدیل mmHg به پاسکال
rho_blood = 1060      # چگالی خون (kg/m³)
g = 9.81              # شتاب گرانش (m/s²)

# ======================== مدل کاسون (رفتار غیرنیوتنی خون) ========================

def calc_mu_apparent(mu_inf, tau_y, gamma_dot):
   
    
    return mu_inf * (1 + np.sqrt(tau_y / (mu_inf * gamma_dot)))**2


def calc_shear_rate(Q, r0):
   
    Q_sin=Q/1000000000
    if r0 <= 0:
        return 1.0
    return (4 * Q_sin) / (np.pi * r0**3)


# ======================== افت فشار سینوزوئیدی ========================

def calc_sinusoid_pressure_drop(Q, mu, L, r0, beta):
    
    Q_sin=Q/1000000000
    if beta == 0:
        # قانون پوازوی برای شعاع ثابت
        return (8 * mu * L * Q_sin) / (np.pi * r0**4)
    else:
        return (8 * mu * L * Q_sin) / (3 * np.pi * r0**4 * beta) * (1 / (1 - beta)**3 - 1)


# ======================== توابع غیرخطی (مقاله) ========================

def calc_Kf_nonlinear(Kf0, deltaP):
   
    if deltaP < 12:
        return Kf0
    else:
        exponent = 0.05  *(20-11)* ((deltaP - 11) / (20 - 11))**2
        return Kf0 * np.exp(exponent)


def calc_Pi_nonlinear(Pi0, deltaP):
    
    if deltaP < 12:
        return Pi0
    else:
        return Pi0 * (deltaP - 11)**(1/3)


def calc_Jlymph(Jmax, Km, Pi):
   
    return (Jmax * Pi) / (Km + Pi)
# ======================== محاسبه α (پارامتر مؤثر) ========================
def calc_alpha(Q_portal, Q_artery, A_portal, A_hepatic, h, r0, beta, L, mu_inf, tau_y):
    
    # دبی کل
    Q_total = Q_portal + Q_artery
    
    # سرعت‌ها
    vp = Q_portal / A_portal if A_portal > 0 else 0
    vh = Q_total*0.327 / A_hepatic if A_hepatic > 0 else 0
    
    # محاسبه نرخ برش و ویسکوزیته ظاهری با مدل کاسون
    gamma_dot = calc_shear_rate(Q_total, r0)
    mu_app = calc_mu_apparent(mu_inf, tau_y, gamma_dot)
    
    # افت فشار سینوزوئیدی (رابطه ۳)
    dp_sin = calc_sinusoid_pressure_drop(Q_total, mu_app, L, r0, beta)
    
    # افت فشار ارتفاع (ترم پتانسیل گرانشی)
    dp_h = rho_blood * g * h
    
    # افت فشار جنبشی
    dp_v = 0.5 * rho_blood * (vh**2 - vp**2)
    
    # کل افت فشار
    dp_total = dp_sin + dp_h + dp_v
    
    # محاسبه α (نسبت افت فشار سینوزوئیدی به کل)
    if dp_total > 0:
        alpha = dp_sin / dp_total
        alpha = np.clip(alpha, 0.1, 0.9)  # محدود کردن به محدوده معقول
    else:
        alpha = 0.54  # مقدار پیش‌فرض از مقاله
    
    return alpha, Q_total, vp, vh, dp_sin, dp_h, dp_v, dp_total, mu_app


# ======================== محاسبه تراوش (استارلینگ اصالح‌شده) ========================

def calc_Jv(deltaP, Kf, alpha, sigma, Pi,dpi ,P_hepatic):
    
   
 
    
    # فشار سینوزوئیدی
    Pc = (alpha * deltaP) + P_hepatic
    
    # ترم‌های معادله استارلینگ
    hydrostatic = Pc - Pi
    oncotic = sigma * (dpi)
    
    # شار تراوش
    Jv = Kf * (hydrostatic - oncotic)
    
    return Jv


def calc_Jnet(Jv, Jlymph):
   
    return Jv - Jlymph


def predict_ascites_volume(Jnet, time_hours, V0=0):
  
    time_min = time_hours * 60
    return V0 + Jnet * time_min


# ======================== تحلیل کامل سیستم ========================

def analyze_system(deltaP, params):
    
    # استخراج پارامترها
    Kf0 = params['Kf0']
    alpha = params['alpha']
    sigma = params['sigma']
    Pi0 = params['Pi0']
    Jmax = params.get('Jmax', 30)
    Km = params.get('Km', 0.5)
    dpi=params['dPi']
    
    # محاسبه پارامترهای غیرخطی
    Kf = calc_Kf_nonlinear(Kf0, deltaP)
    Pi = calc_Pi_nonlinear(Pi0, deltaP)
    
    # محاسبه تراوش
    P_hep=4
    Jv = calc_Jv(deltaP, Kf, alpha, sigma, Pi, dpi, P_hep)
    
    # محاسبه تخلیه لنفاوی
    Jlymph = calc_Jlymph(Jmax, Km, Pi)
    
    # محاسبه نرخ خالص تجمع
    Jnet = calc_Jnet(Jv, Jlymph)
    
    # وضعیت بالینی
    if Jnet <= 0:
        status = "✅ تخلیه کامل - بدون آسیت"
        risk = "کم"
    elif Jnet < 5:
        status = "🟡 تجمع خفیف - خطر پایین آسیت"
        risk = "متوسط"
    elif Jnet < 15:
        status = "🟠 تجمع متوسط - خطر آسیت"
        risk = "بالا"
    else:
        status = "🔴 تجمع شدید - خطر بالای آسیت"
        risk = "بسیار بالا"
    
    return {
        'deltaP': deltaP,
        'Kf': Kf,
        'Pi': Pi,
        'Jv': Jv,
        'Jlymph': Jlymph,
        'Jnet': Jnet,
        'status': status,
        'risk': risk
    }


def analyze_system_range(deltaP_range, params):
   
    results = {
        'deltaP': deltaP_range,
        'Kf': [],
        'Pi': [],
        'Jv': [],
        'Jlymph': [],
        'Jnet': []
    }
    
    for dp in deltaP_range:
        res = analyze_system(dp, params)
        results['Kf'].append(res['Kf'])
        results['Pi'].append(res['Pi'])
        results['Jv'].append(res['Jv'])
        results['Jlymph'].append(res['Jlymph'])
        results['Jnet'].append(res['Jnet'])
    
    return results
