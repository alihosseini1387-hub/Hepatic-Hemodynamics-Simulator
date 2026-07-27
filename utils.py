"""
توابع کمکی برای اپلیکیشن شبیه‌سازی همودینامیک کبد
Helper functions for Hepatic Hemodynamics Simulation App
"""

# دیکشنری متن‌های قابل نمایش در این فایل
# (فعلاً توابع این فایل متن مستقیم ندارند، فقط برای هماهنگی با بقیه اضافه شده)
LANG = "fa"  # این توسط app.py مدیریت می‌شود

def format_flow_rate(Q, unit='ml/min'):
    """
    فرمت‌سازی نرخ جریان
    Format flow rate
    
    پارامترها / Parameters:
    ----------
    Q : float
        نرخ جریان (هر واحد) / Flow rate (any unit)
    unit : str
        واحد نمایش / Display unit
    
    بازگشت / Returns:
    --------
    str : رشته فرمت‌شده / Formatted string
    """
    if Q < 1:
        return f"{Q*1000:.2f} µl/{unit.replace('ml', '')}"
    elif Q < 1000:
        return f"{Q:.2f} {unit}"
    else:
        return f"{Q/1000:.2f} L/{unit.replace('ml', '')}"


def get_clinical_interpretation(deltaP, Jv, Jnet, lang="fa"):
    """
    تفسیر بالینی نتایج
    Clinical interpretation of results
    
    پارامترها / Parameters:
    ----------
    deltaP : float
        اختلاف فشار (mmHg) / Pressure difference (mmHg)
    Jv : float
        شار تراوش (ml/min) / Filtration flux (ml/min)
    Jnet : float
        نرخ خالص تجمع (ml/min) / Net accumulation rate (ml/min)
    lang : str
        زبان / Language: "fa" or "en"
    
    بازگشت / Returns:
    --------
    dict : تفسیر بالینی / Clinical interpretation
    """
    
    # ====== متن‌های دو زبانه ======
    texts = {
        "fa": {
            "normal": "طبیعی",
            "warning": "مرز هشدار",
            "mild_ht": "پرفشاری خفیف-متوسط",
            "severe_ht": "پرفشاری شدید",
            "desc_normal": "فشار پورتال در محدوده فیزیولوژیک",
            "desc_warning": "فشار پورتال افزایش یافته، نیاز به پایش",
            "desc_mild": "خطر تشکیل آسیت",
            "desc_severe": "خطر بالای آسیت، نیاز به مداخله",
            "fluid_balanced": "تعادل مایعات برقرار",
            "fluid_mild": "تجمع خفیف",
            "fluid_moderate": "تجمع متوسط",
            "fluid_severe": "تجمع شدید",
            "risk_none": "بدون خطر آسیت",
            "risk_low": "خطر پایین آسیت",
            "risk_moderate": "خطر آسیت",
            "risk_high": "خطر بالای آسیت",
            "time_min": "دقیقه",
            "time_hour": "ساعت",
            "time_day": "روز",
            "ascites_warning": "⚠️ {time} تا تشکیل ۵۰۰ میلی‌لیتر آسیت",
            "ascites_safe": "✅ بدون خطر تشکیل آسیت",
            "deltaP_label": "اختلاف فشار",
            "status_label": "وضعیت",
            "fluid_label": "وضعیت مایعات",
        },
        "en": {
            "normal": "Normal",
            "warning": "Warning Zone",
            "mild_ht": "Mild-Moderate Hypertension",
            "severe_ht": "Severe Hypertension",
            "desc_normal": "Portal pressure in physiological range",
            "desc_warning": "Elevated portal pressure, monitoring required",
            "desc_mild": "Risk of ascites formation",
            "desc_severe": "High risk of ascites, intervention needed",
            "fluid_balanced": "Fluid balance maintained",
            "fluid_mild": "Mild accumulation",
            "fluid_moderate": "Moderate accumulation",
            "fluid_severe": "Severe accumulation",
            "risk_none": "No ascites risk",
            "risk_low": "Low ascites risk",
            "risk_moderate": "Ascites risk",
            "risk_high": "High ascites risk",
            "time_min": "minutes",
            "time_hour": "hours",
            "time_day": "days",
            "ascites_warning": "⚠️ {time} until 500 mL ascites formation",
            "ascites_safe": "✅ No ascites formation risk",
            "deltaP_label": "Pressure Difference",
            "status_label": "Status",
            "fluid_label": "Fluid Status",
        }
    }
    
    t = texts[lang]
    
    # تعیین وضعیت بر اساس ΔP
    if deltaP < 10:
        status = t["normal"]
        color = "🟢"
        description = t["desc_normal"]
    elif deltaP < 12:
        status = t["warning"]
        color = "🟡"
        description = t["desc_warning"]
    elif deltaP < 16:
        status = t["mild_ht"]
        color = "🟠"
        description = t["desc_mild"]
    else:
        status = t["severe_ht"]
        color = "🔴"
        description = t["desc_severe"]
    
    # تعیین وضعیت بر اساس Jnet
    if Jnet <= 0:
        fluid_status = t["fluid_balanced"]
        fluid_risk = t["risk_none"]
    elif Jnet < 5:
        fluid_status = t["fluid_mild"]
        fluid_risk = t["risk_low"]
    elif Jnet < 15:
        fluid_status = t["fluid_moderate"]
        fluid_risk = t["risk_moderate"]
    else:
        fluid_status = t["fluid_severe"]
        fluid_risk = t["risk_high"]
    
    # پیش‌بینی زمان تشکیل آسیت
    if Jnet > 0:
        time_to_ascites = 500 / (Jnet * 60)
        if time_to_ascites < 1:
            time_str = f"{time_to_ascites*60:.0f} {t['time_min']}"
        elif time_to_ascites < 24:
            time_str = f"{time_to_ascites:.1f} {t['time_hour']}"
        else:
            time_str = f"{time_to_ascites/24:.1f} {t['time_day']}"
        ascites_prediction = t["ascites_warning"].format(time=time_str)
    else:
        ascites_prediction = t["ascites_safe"]
    
    return {
        'status': status,
        'color': color,
        'description': description,
        'fluid_status': fluid_status,
        'fluid_risk': fluid_risk,
        'ascites_prediction': ascites_prediction
    }


def generate_report(results, params, lang="fa"):
    """
    تولید گزارش کامل از نتایج
    Generate complete report of results
    
    پارامترها / Parameters:
    ----------
    results : dict
        نتایج تحلیل / Analysis results
    params : dict
        پارامترهای مدل / Model parameters
    lang : str
        زبان / Language: "fa" or "en"
    
    بازگشت / Returns:
    --------
    str : گزارش متنی / Text report
    """
    
    texts = {
        "fa": {
            "title": "📋 گزارش تحلیل همودینامیک کبد",
            "model_params": "📊 پارامترهای مدل:",
            "mu_inf": "ویسکوزیته پایه (μ∞)",
            "tau_y": "تنش تسلیم (τy)",
            "r0": "شعاع سینوزوئید (r₀)",
            "beta": "ضریب مخروطی (β)",
            "kf0": "ضریب فیلتراسیون (Kf₀)",
            "sigma": "ضریب انعکاس (σ)",
            "pi0": "فشار میان‌بافتی (Pi₀)",
            "results": "📈 نتایج تحلیل:",
            "deltaP": "اختلاف فشار (ΔP)",
            "kf_eff": "ضریب فیلتراسیون مؤثر (Kf)",
            "pi_eff": "فشار میان‌بافتی مؤثر (Pi)",
            "jv": "شار تراوش (Jv)",
            "jlymph": "تخلیه لنفاوی (Jlymph)",
            "jnet": "نرخ خالص تجمع (Jnet)",
            "clinical": "🏥 تفسیر بالینی:",
            "status": "وضعیت",
            "fluid_status": "وضعیت مایعات",
        },
        "en": {
            "title": "📋 Hepatic Hemodynamics Analysis Report",
            "model_params": "📊 Model Parameters:",
            "mu_inf": "Baseline Viscosity (μ∞)",
            "tau_y": "Yield Stress (τy)",
            "r0": "Sinusoid Radius (r₀)",
            "beta": "Tapering Coefficient (β)",
            "kf0": "Filtration Coefficient (Kf₀)",
            "sigma": "Reflection Coefficient (σ)",
            "pi0": "Interstitial Pressure (Pi₀)",
            "results": "📈 Analysis Results:",
            "deltaP": "Pressure Difference (ΔP)",
            "kf_eff": "Effective Filtration Coefficient (Kf)",
            "pi_eff": "Effective Interstitial Pressure (Pi)",
            "jv": "Filtration Flux (Jv)",
            "jlymph": "Lymphatic Drainage (Jlymph)",
            "jnet": "Net Accumulation Rate (Jnet)",
            "clinical": "🏥 Clinical Interpretation:",
            "status": "Status",
            "fluid_status": "Fluid Status",
        }
    }
    
    t = texts[lang]
    
    report = []
    report.append("=" * 60)
    report.append(t["title"])
    report.append("=" * 60)
    report.append("")
    
    # پارامترهای مدل
    report.append(t["model_params"])
    report.append(f"  • {t['mu_inf']}: {params.get('mu_inf', 0.0035):.4f} Pa.s")
    report.append(f"  • {t['tau_y']}: {params.get('tau_y', 0.005):.4f} Pa")
    report.append(f"  • {t['r0']}: {params.get('r0', 5)*1e6:.0f} μm")
    report.append(f"  • {t['beta']}: {params.get('beta', 0.4):.2f}")
    report.append(f"  • {t['kf0']}: {params.get('Kf0', 3):.1f} ml/min/mmHg")
    report.append(f"  • {t['sigma']}: {params.get('sigma', 0.22):.2f}")
    report.append(f"  • {t['pi0']}: {params.get('Pi0', 0.5):.1f} mmHg")
    report.append("")
    
    # نتایج
    report.append(t["results"])
    report.append(f"  • {t['deltaP']}: {results['deltaP']:.1f} mmHg")
    report.append(f"  • {t['kf_eff']}: {results['Kf']:.3f} ml/min/mmHg")
    report.append(f"  • {t['pi_eff']}: {results['Pi']:.2f} mmHg")
    report.append(f"  • {t['jv']}: {results['Jv']:.2f} ml/min")
    report.append(f"  • {t['jlymph']}: {results['Jlymph']:.2f} ml/min")
    report.append(f"  • {t['jnet']}: {results['Jnet']:.2f} ml/min")
    report.append("")
    
    # تفسیر بالینی
    clinical = get_clinical_interpretation(
        results['deltaP'], 
        results['Jv'], 
        results['Jnet'],
        lang=lang
    )
    
    report.append(t["clinical"])
    report.append(f"  • {t['status']}: {clinical['color']} {clinical['status']}")
    report.append(f"  • {clinical['description']}")
    report.append(f"  • {t['fluid_status']}: {clinical['fluid_status']}")
    report.append(f"  • {clinical['ascites_prediction']}")
    report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)