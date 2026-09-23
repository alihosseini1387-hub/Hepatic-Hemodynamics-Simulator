"""
مدل‌های ریاضی شبیه‌سازی همودینامیک کبد
بر اساس مقاله: شبیه‌سازی جریان خون در کبد بر اساس اصول و معادلات مکانیک سیالات

این فایل شامل مدل‌های ریاضی برای:
- محاسبه ویسکوزیته ظاهری خون (مدل کاسون)
- محاسبه افت فشار سینوزوئیدی
- محاسبه پارامتر α
- محاسبه شار تراوش (Starling اصلاح‌شده)
- محاسبه تخلیه لنفاوی (Michaelis-Menten)
- پیش‌بینی دینامیک حجم آسیت (معادله دیفرانسیل)
"""

import numpy as np

# ======================== ثابت‌های فیزیکی ========================
mmHg_to_Pa = 133.322  # تبدیل mmHg به پاسکال
rho_blood = 1060      # چگالی خون (kg/m³)
g = 9.81              # شتاب گرانش (m/s²)


# ======================== مدل کاسون (رفتار غیرنیوتنی خون) ========================

def calc_mu_apparent(mu_inf, tau_y, gamma_dot):
    """
    محاسبه ویسکوزیته ظاهری خون با مدل کاسون
    
    پارامترها:
    -----------
    mu_inf : ویسکوزیته در نرخ برش بی‌نهایت (Pa·s)
    tau_y : تنش تسلیم (Pa)
    gamma_dot : نرخ برش (s⁻¹)
    
    خروجی:
    -------
    mu_app : ویسکوزیته ظاهری (Pa·s)
    """
    if gamma_dot <= 0:
        return mu_inf
    return mu_inf * (1 + np.sqrt(tau_y / (mu_inf * gamma_dot)))**2


def calc_shear_rate(Q, r0):
    """
    محاسبه نرخ برش در سینوزوئید
    
    پارامترها:
    -----------
    Q : دبی جریان (m³/s)
    r0 : شعاع سینوزوئید (m)
    
    خروجی:
    -------
    gamma_dot : نرخ برش (s⁻¹)
    """
    Q_sin = Q / 1000000000  # تبدیل به m³/s
    if r0 <= 0:
        return 1.0
    return (4 * Q_sin) / (np.pi * r0**3)


# ======================== افت فشار سینوزوئیدی ========================

def calc_sinusoid_pressure_drop(Q, mu, L, r0, beta):
    """
    محاسبه افت فشار سینوزوئید با شعاع متغیر (معادله ۳)
    
    پارامترها:
    -----------
    Q : دبی جریان (m³/s)
    mu : ویسکوزیته خون (Pa·s)
    L : طول سینوزوئید (m)
    r0 : شعاع پایه سینوزوئید (m)
    beta : ضریب تغییر شعاع
    
    خروجی:
    -------
    delta_P : افت فشار سینوزوئیدی (Pa)
    """
    Q_sin = Q / 1000000000  # تبدیل به m³/s
    if beta == 0:
        # قانون پوازوی برای شعاع ثابت
        return (8 * mu * L * Q_sin) / (np.pi * r0**4)
    else:
        return (8 * mu * L * Q_sin) / (3 * np.pi * r0**4 * beta) * (1 / (1 - beta)**3 - 1)


# ======================== توابع غیرخطی (مقاله) ========================

def calc_Kf_nonlinear(Kf0, deltaP):
    """
    محاسبه ضریب فیلتراسیون غیرخطی Kf
    
    پارامترها:
    -----------
    Kf0 : ضریب فیلتراسیون پایه (ml/min/mmHg)
    deltaP : گرادیان فشار پورتال-هپاتیک (mmHg)
    
    خروجی:
    -------
    Kf : ضریب فیلتراسیون مؤثر (ml/min/mmHg)
    """
    if deltaP < 12:
        return Kf0
    else:
        exponent = 0.05 * (20 - 11) * ((deltaP - 11) / (20 - 11))**2
        return Kf0 * np.exp(exponent)


def calc_Pi_nonlinear(Pi0, deltaP):
    """
    محاسبه فشار میان‌بافتی غیرخطی Pi (مکانیسم جبرانی)
    
    پارامترها:
    -----------
    Pi0 : فشار میان‌بافتی پایه (mmHg)
    deltaP : گرادیان فشار پورتال-هپاتیک (mmHg)
    
    خروجی:
    -------
    Pi : فشار میان‌بافتی مؤثر (mmHg)
    """
    if deltaP < 12:
        return Pi0
    else:
        return Pi0 * (deltaP - 11)**(1/3)


def calc_Jlymph(Jmax, Km, Pi):
    """
    محاسبه تخلیه لنفاوی با معادله Michaelis-Menten
    
    پارامترها:
    -----------
    Jmax : حداکثر ظرفیت لنفاوی (ml/min)
    Km : ثابت Michaelis-Menten (mmHg)
    Pi : فشار میان‌بافتی (mmHg)
    
    خروجی:
    -------
    Jlymph : نرخ تخلیه لنفاوی (ml/min)
    """
    if Km + Pi == 0:
        return 0
    return (Jmax * Pi) / (Km + Pi)


# ======================== محاسبه α (پارامتر مؤثر) ========================

def calc_alpha(Q_portal, Q_artery, A_portal, A_hepatic, h, r0, beta, L, mu_inf, tau_y):
    """
    محاسبه پارامتر α (سهم مقاومت سینوزوئیدی از کل افت فشار)
    
    پارامترها:
    -----------
    Q_portal : دبی ورید باب (m³/s)
    Q_artery : دبی سرخرگ کبدی (m³/s)
    A_portal : سطح مقطع ورید باب (m²)
    A_hepatic : سطح مقطع ورید فوق‌کبدی (m²)
    h : اختلاف ارتفاع (m)
    r0 : شعاع سینوزوئید (m)
    beta : ضریب تغییر شعاع
    L : طول سینوزوئید (m)
    mu_inf : ویسکوزیته پایه (Pa·s)
    tau_y : تنش تسلیم (Pa)
    
    خروجی:
    -------
    alpha : پارامتر مؤثر
    Q_total : دبی کل (m³/s)
    vp : سرعت ورید باب (m/s)
    vh : سرعت ورید فوق‌کبدی (m/s)
    dp_sin : افت فشار سینوزوئیدی (Pa)
    dp_h : افت فشار ارتفاع (Pa)
    dp_v : افت فشار جنبشی (Pa)
    dp_total : کل افت فشار (Pa)
    mu_app : ویسکوزیته ظاهری (Pa·s)
    """
    
    # دبی کل
    Q_total = Q_portal + Q_artery
    
    # سرعت‌ها
    vp = Q_portal / A_portal if A_portal > 0 else 0
    vh = Q_total * 0.327 / A_hepatic if A_hepatic > 0 else 0
    
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


# ======================== محاسبه تراوش (استارلینگ اصلاح‌شده) ========================

def calc_Jv(deltaP, Kf, alpha, sigma, Pi, dpi, P_hepatic):
    """
    محاسبه شار تراوش با معادله Starling اصلاح‌شده
    
    پارامترها:
    -----------
    deltaP : گرادیان فشار پورتال-هپاتیک (mmHg)
    Kf : ضریب فیلتراسیون (ml/min/mmHg)
    alpha : پارامتر مؤثر
    sigma : ضریب انعکاس
    Pi : فشار میان‌بافتی (mmHg)
    dpi : اختلاف فشار انکوتیک (πc - πi) (mmHg)
    P_hepatic : فشار ورید هپاتیک (mmHg)
    
    خروجی:
    -------
    Jv : شار تراوش (ml/min)
    """
    
    # فشار سینوزوئیدی
    Pc = (alpha * deltaP) + P_hepatic
    
    # ترم‌های معادله استارلینگ
    hydrostatic = Pc - Pi
    oncotic = sigma * dpi
    
    # شار تراوش
    Jv = Kf * (hydrostatic - oncotic)
    
    return Jv


def calc_Jnet(Jv, Jlymph):
    """
    محاسبه نرخ خالص تجمع مایع
    
    پارامترها:
    -----------
    Jv : شار تراوش (ml/min)
    Jlymph : تخلیه لنفاوی (ml/min)
    
    خروجی:
    -------
    Jnet : نرخ خالص تجمع (ml/min)
    """
    return Jv - Jlymph


# ======================== مدل دینامیک آسیت (جایگزین مدل استاتیک) ========================

def predict_ascites_volume_dynamic(Kf, alpha, sigma, dpi, P_hepatic, Pi0, 
                                   k_elastance, Jmax, Km, deltaP, 
                                   time_hours, V0=0, dt=0.01):
    """
    حل معادله دیفرانسیل dV/dt = Jnet(t) با روش Euler
    
    این تابع، مدل دینامیک تشکیل آسیت # را پیاده‌سازی می‌کند که تعداد در آن
    فشار میان‌ب گامافتی (Pi) با افزایش حجم مایع افزایش می‌یابد و باعث
    کاهش تدریجی Jnet و رسیدن حجم آسیت به حد اشباع می‌شود.
    
    معادله دیفرانسیل:
    -----------------
    dV/dt = Jnet(t) = Jv(t) - Jlymph(t)
    
    که در آن:
    Pi(V) = Pi0 + k_elastance × V
    Jv = Kf × [(Pc - Pi) - σ × Δπ]
    Jlymph = (Jmax × Pi) / (Km + Pi)
    
    پارامترها:
    -----------
    Kf : ضریب فیلتراسیون (ml/min/mmHg)
    alpha : سهم مقاومت سینوزوئیدی
    sigma : ضریب انعکاس
    dpi : اختلاف فشار انکوتیک (πc - πi) (mmHg)
    P_hepatic : فشار ورید هپاتیک (mmHg)
    Pi0 : فشار بین‌بافتی پایه (mmHg)
    k_elastance : ضریب الاستانس بافت (mmHg/mL)
    Jmax : حداکثر ظرفیت لنفاوی (ml/min)
    Km : ثابت Michaelis-Menten (mmHg)
    deltaP : گرادیان فشار پورتال-هپاتیک (mmHg)
    time_hours : زمان شبیه‌سازی (ساعت)
    V0 : حجم اولیه آسیت (mL)
    dt : گام زمانی (ساعت)
    
    خروجی:
    -------
    time_array : آرایه زمان (ساعت)
    V_array : آرایه حجم آسیت (mL)
    Jnet_array : آرایه نرخ خالص تجمع (ml/min)
    Pi_array : آرایه فشار بین‌بافتی (mmHg)
    """
    
    # تبدیل زمان به دقیقه
    time_min = time_hours * 60
    dt_min = dt * 60
    
   ‌ها
    n_steps = int(time_min / dt_min)
    if n_steps < 1:
        n_steps = 1
    
    # آرایه‌های خروجی
    time_array = np.linspace(0, time_hours, n_steps + 1)
    V_array = np.zeros(n_steps + 1)
    Jnet_array = np.zeros(n_steps + 1)
    Pi_array = np.zeros(n_steps + 1)
    
    # مقدار اولیه
    V_array[0] = V0
    Pi_array[0] = Pi0
    
    # محاسبه Jnet اولیه
    Pc = alpha * deltaP + P_hepatic
    Jv_initial = Kf * ((Pc - Pi0) - sigma * dpi)
    Jlymph_initial = (Jmax * Pi0) / (Km + Pi0) if (Km + Pi0) > 0 else 0
    Jnet_array[0] = Jv_initial - Jlymph_initial
    
    # حلقه اصلی (روش Euler)
    for i in range(n_steps):
        # ۱. محاسبه Pi بر اساس حجم فعلی
        Pi = Pi0 + k_elastance * V_array[i]
        Pi_array[i] = Pi
        
        # ۲. محاسبه فشار سینوزوئیدی
        Pc = alpha * deltaP + P_hepatic
        
        # ۳. محاسبه Jv (تراوش)
        Jv = Kf * ((Pc - Pi) - sigma * dpi)
        
        # ۴. محاسبه Jlymph (تخلیه لنفاوی)
        Jlymph = (Jmax * Pi) / (Km + Pi) if (Km + Pi) > 0 else 0
        
        # ۵. محاسبه Jnet
        Jnet = Jv - Jlymph
        Jnet_array[i] = Jnet
        
        # ۶. گام Euler
        V_array[i + 1] = V_array[i] + Jnet * dt_min
        V_array[i + 1] = max(0, V_array[i + 1])  # حجم منفی نمی‌شه
    
    # مقدار نهایی
    Pi_final = Pi0 + k_elastance * V_array[-1]
    Pc_final = alpha * deltaP + P_hepatic
    Jv_final = Kf * ((Pc_final - Pi_final) - sigma * dpi)
    Jlymph_final = (Jmax * Pi_final) / (Km + Pi_final) if (Km + Pi_final) > 0 else 0
    Jnet_array[-1] = Jv_final - Jlymph_final
    Pi_array[-1] = Pi_final
    
    return time_array, V_array, Jnet_array, Pi_array


# ======================== تحلیل کامل سیستم ========================

def analyze_system(deltaP, params):
    """
    تحلیل کامل سیستم برای یک مقدار deltaP مشخص
    
    پارامترها:
    -----------
    deltaP : گرادیان فشار پورتال-هپاتیک (mmHg)
    params : دیکشنری پارامترها
    
    خروجی:
    -------
    results : دیکشنری نتایج
    """
    
    # استخراج پارامترها
    Kf0 = params['Kf0']
    alpha = params['alpha']
    sigma = params['sigma']
    Pi0 = params['Pi0']
    Jmax = params.get('Jmax', 30)
    Km = params.get('Km', 0.5)
    dpi = params['dPi']
    
    # محاسبه پارامترهای غیرخطی
    Kf = calc_Kf_nonlinear(Kf0, deltaP)
    Pi = calc_Pi_nonlinear(Pi0, deltaP)
    
    # محاسبه تراوش
    P_hep = 4
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
    """
    تحلیل سیستم برای یک محدوده از deltaP
    
    پارامترها:
    -----------
    deltaP_range : آرایه مقادیر deltaP (mmHg)
    params : دیکشنری پارامترها
    
    خروجی:
    -------
    results : دیکشنری نتایج
    """
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
