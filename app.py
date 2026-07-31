
"""
اپلیکیشن شبیه‌سازی همودینامیک کبد
بر اساس مقاله: شبیه‌سازی جریان خون در کبد بر اساس اصول و معادلات مکانیک سیالات
سیدعلی حسینی
Ali hosseini
Hepatic Hemodynamics Simulation App
Based on the paper: Simulation of hepatic blood flow based on fluid mechanics principles
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from models import (
    mmHg_to_Pa, rho_blood, g,
    calc_mu_apparent, calc_shear_rate,
    calc_sinusoid_pressure_drop,
    calc_Kf_nonlinear, calc_Pi_nonlinear,
    calc_Jlymph, calc_Jnet,
    calc_alpha, calc_Jv,
)

from utils import get_clinical_interpretation

# ======================== صفحه خوش‌آمدگویی ========================
if "first_run" not in st.session_state:
    st.session_state.first_run = True

if st.session_state.first_run:
    
    st.empty()
    
   
    col1, col2, col3 = st.columns([1, 2, 1])

    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 50px 0;">
            <h1 style="font-size: 60px;">🩸</h1>
            <h1 style="font-size: 40px; color: #ff4b4b;">شبیه‌ساز همودینامیک کبد</h1>
            <h3 style="color: #666;">Hepatic Hemodynamics Simulator</h3>
            <br>
            <style>
            .stButton button {
            background: linear-gradient(90deg, #ff4b4b, #ff6b6b) !important;
            color: white !important;
            font-size: 20px !important;
            padding: 15px 60px !important;
            border-radius: 30px !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(255,75,75,0.4) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }
        .stButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 8px 30px rgba(255,75,75,0.6) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.button(" Enter🚀 ", use_container_width=True):
        st.session_state.first_run = False
        st.rerun()

    
    st.stop()  
    


st.set_page_config(
    page_title="Hepatic Hemodynamics Simulator",
    page_icon="🩸",
    layout="wide"
)


TEXTS = {
    "en": {
    
    "app_title": "🩸 Hepatic Filtration Simulator",
    "app_subtitle": "Based on the paper *Simulation of hepatic blood flow based on fluid mechanics principles*, Mandegar Alborz High School",
    
        "preset_title": "📋 Clinical Scenarios",
        "preset_custom": "🔘 Custom (Manual)",
        "preset_normal": "🟢 Normal (Healthy)",
        "preset_mild": "🟡 Mild Portal Hypertension",
        "preset_cirrhosis": "🔴 Cirrhosis with Ascites",
        "preset_refractory": "🔴🔴 Refractory Ascites",
        "preset_apply": "✅ Apply Scenario",
        "preset_loaded": "✅ Scenario '{name}' loaded. Click 'Apply Scenario' to activate.",
        
        "clear_cache": "🗑️ Clear Cache",
        "cache_cleared": "✅ Cache cleared successfully!",
        
        "pdf_report": "📄 PDF Report",
        "pdf_download": "📥 Download Full Report",
        "pdf_title": "Hepatic Hemodynamics Report",
        "pdf_date": "Date",
        "pdf_params": "1. Input Parameters",
        "pdf_results": "2. Results",
        "pdf_clinical": "3. Clinical Interpretation",
        "pdf_status": "Status",
        "pdf_description": "Description",
        "pdf_ascites": "Ascites Prediction",
        "pdf_install_warning": "⚠️ To download PDF, install: `pip install fpdf`",
        
        "upload_csv": "📤 Upload Patient Data (CSV)",
        "upload_help": "Select CSV file with columns: ΔP, Kf0, sigma, Pi0, Jmax, Km, dPi",
        "upload_run": "🚀 Run Model on Data",
        "upload_status": "⚠️ Ascites Risk",
        "upload_compensated": "✅ Compensated",
        "upload_download": "📥 Download Results (CSV)",
        "upload_error": "❌ Error reading file: {e}",
        
        "3d_title": "📊 Interactive 3D Plot",
        "3d_info": "ℹ️ Select two parameters for 3D visualization:",
        "3d_param1": "First Parameter (X)",
        "3d_param2": "Second Parameter (Y)",
        "3d_plot": "🎲 Draw 3D Plot",
        "3d_title_plot": "Simultaneous Effect of {p1} and {p2} on Jv",
        "3d_x": "{p1}",
        "3d_y": "{p2}",
        "3d_z": "Jv (ml/min)",
        
        "reset_title": "🔄 Reset All Settings",
        "reset_confirm": "⚠️ Are you sure you want to reset all settings?",
        
        "validation_warning_flow": "⚠️ Portal vein flow must be positive!",
        "validation_warning_area": "⚠️ Cross-sectional area must be greater than zero!",
    "settings": "⚙️ Settings",
    "mode_label": "Mode",
    "mode_auto": "🔄 Auto",
    "mode_manual": "✋ Manual",
    "hemo_params": "Hemodynamic Parameters",
    "portal_flow": "Portal Vein Flow (L/min)",
    "artery_flow": "Hepatic Artery Flow (L/min)",
    "portal_area": "Portal Vein Cross-Section (cm²)",
    "hepatic_area": "Hepatic Vein Cross-Section (cm²)",
    "height_diff": "Height Difference h (cm)",
    "sinusoid_params": "Sinusoid Parameters (Casson Model)",
    "mu_inf": "μ∞ (Pa·s)",
    "tau_y": "τy (Pa)",
    "r0": "r₀ (μm)",
    "L": "L (μm)","afz":"Parameter Increase","kahesh":"Parameter Decrease",
    "beta": "β (Tapering Coefficient)",
    "filtration_params": "Filtration Parameters (Starling)",
    "kf0": "Kf₀",
    "sigma": "σ",
    "pi0": "Pi₀ (mmHg)",
    "dpi": "Δπ (mmHg)",
    "lymph_params": "Lymphatic Parameters",
    "jmax": "Jmax (ml/min)",
    "km": "Km (mmHg)",
    "max_dp": "Max ΔP (mmHg)",
    "manual_alpha": "α (Manual)",
    "param12":"parametr",
    
    "alpha": "α",
    "shear_rate": "Shear Rate",
    "mu_app": "μ apparent",
    "dp_total": "Total ΔP",
    "calc_details": "📊 Calculation Details",
    "total_flow": "Total Flow",
    "portal_vel": "Portal Vein Velocity",
    "hepatic_vel": "Hepatic Vein Velocity",
    "sinusoid_drop": "Sinusoid Pressure Drop",
    "height_drop": "Height Pressure Drop",
    "kinetic_drop": "Kinetic Pressure Drop",
    "total_drop": "Total Pressure Drop",
    "auto_results": "📊 Auto-Mode Results",
    "deltaP": "ΔP",
    "kf_eff": "Kf",
    "pi_eff": "Pi","P_hep":"P(hepatic vein)",
    "jv": "Jv",
    "jnet": "Jnet",
    "clinical_title": "🏥 Clinical Interpretation (ΔP = {dp:.2f} mmHg)",
    "status": "Status",
    "fluid_status": "Fluid Status",
    "ascites_pred": "Ascites Prediction",
    "key_values": "📊 Key Values at Different Points",
    "dp_mmHg": "ΔP (mmHg)",
    "jlymph": "Jlymph (ml/min)",
    "manual_results": "📊 Manual-Mode Results",
    "filtration_curves": "📊 Filtration Curves",
    "jv_curve": "Jv (Filtration)",
    "jlymph_curve": "Jlymph (Lymphatic)",
    "jnet_curve": "Jnet (Net)",
    "zero_line": "J = 0",
    "threshold_line": "Threshold 12 mmHg",
    "curves_title": "📊 Filtration, Lymphatic Drainage, and Net Accumulation Curves",
    "xaxis_dp": "ΔP (mmHg)",
    "yaxis_flow": "Flow Rate (ml/min)",
    "nonlinear_behavior": "📊 Nonlinear Parameter Behavior",
    "kf_title": "Nonlinear Filtration Coefficient Kf",
    "kf_yaxis": "Kf (ml/min/mmHg)",
    "pi_title": "Nonlinear Interstitial Pressure Pi (Compensatory Mechanism)",
    "pi_yaxis": "Pi (mmHg)",
    "key_points": "📊 Key Values",
    "clinical_interpretation": "🏥 Clinical Interpretation",
    "caption": "α = {alpha:.3f} | h = {h} cm | r₀ = {r0} μm | β = {beta} | Q_total = {q:.1f} L/min | μ = {mu:.5f} Pa·s",
    "info_text": "📌 Interpretation: At low pressures (< 12 mmHg), the lymphatic system can drain the fluid. Beyond the threshold, Kf grows exponentially and filtration exceeds lymphatic capacity, leading to 'Hydraulic Breakdown' and ascites formation.",
    "clinical_expander": "📖 Complete Clinical Interpretation",
    "table_title": "📌 Clinical Ranges",
    "table_col1": "ΔP Range (mmHg)",
    "table_col2": "Status",
    "table_col3": "Ascites Risk",
    "table_col4": "Dominant Mechanism",
    "row1_1": "< 8",
    "row1_2": "Normal (Physiological)",
    "row1_3": "🟢 Very Low",
    "row1_4": "Starling Balance + Lymphatic Drainage",
    "row2_1": "8 – 12",
    "row2_2": "Warning Zone",
    "row2_3": "🟡 Slight Increase",
    "row2_4": "Onset of Nonlinear Kf Growth",
    "row3_1": "12 – 16",
    "row3_2": "Mild-Moderate Portal Hypertension",
    "row3_3": "🔴 Moderate",
    "row3_4": "**Hydraulic Breakdown** (Jnet > 0)",
    "row4_1": "> 16","row4_2": "Severe Portal Hypertension",
    "row4_3": "🔴 High",
    "row4_4": "Exponential Filtration Growth + Lymphatic Saturation",
    "mechanisms_title": "🔬 Key Mechanisms",
    "mech1": "1. Hydraulic Breakdown: At ΔP ≥ 12 mmHg, the combination of viscosity reduction and Kf increase leads to accelerated filtration growth.",
    "mech2": "2. Clinical Threshold: The 12 mmHg point matches clinical observations (Garcia-Tsao et al., 2017) for ascites formation threshold.",
    "mech3": "3. Compensatory Mechanism: Increased Pi partially reduces filtration, but is insufficient at high pressures.",
    "mech4": "4. Lymphatic Saturation: The lymphatic system has limited capacity and cannot fully drain fluid after passing the threshold.",
    "clinical_app": "🏥 Clinical Application",
    "clinical_app_text": "This model can serve as a simple, low-cost clinical tool for:",
    "clinical_app_1": "- Predicting ascites risk in patients with portal hypertension",
    "clinical_app_2": "- Evaluating response to portal pressure-reducing treatments",
    "clinical_app_3": "- Designing future CFD studies for more detailed simulations",
    "comparison_title": "📊 Comparison with Previous Models",
    "comparison_col1": "Feature",
    "comparison_col2": "Siggers (2013)",
    "comparison_col3": "Dongaonkar et al. (2018)",
    "comparison_col4": "Dongaonkar et al. (2020)",
    "comparison_col5": "**Our Model**",
    "comp_bernoulli": "Modified Bernoulli",
    "comp_viscous": "Viscous Pressure Drop",
    "comp_kf": "Kf Nonlinear Pressure-Dependent",
    "comp_pi": "Pi Nonlinear Pressure-Dependent",
    "comp_negative": "Negative Filtration Flux",
    "comp_casson": "Casson Model (Non-Newtonian)",
    "comp_lymph": "Lymphatic Drainage (Michaelis-Menten)",
    "innovation_title": "**Main Innovation:**",
    "innovation_text": "Combination of modified Bernoulli equation, Poiseuille's law with variable radius (β), Casson model, and Starling equation with nonlinear Kf and Pi in a unified framework.",
    "footer": "🩸 Hepatic Hemodynamics Simulator | Based on the paper: Simulation of hepatic blood flow based on fluid mechanics principles | Mandegar Alborz Research Center | Academic Year 2025-2026",
    "lang_label": "Language",
    "lang_en": "🇬🇧 English",
    "lang_fa": "🇮🇷 Persian",
    
    
    "sensitivity_title": "📊 Advanced Sensitivity Analysis",
    "sensitivity_subtitle": "Advanced sensitivity analysis with Monte Carlo, Heatmap and Tornado Diagram",
    "sensitivity_1d": "📈 One-Dimensional",
    "sensitivity_2d": "🎯 Two-Dimensional (Heatmap)",
    "sensitivity_tornado": "🌪️ Tornado Diagram",
    "sensitivity_monte": "🎲 Monte Carlo",
    "sensitivity_report": "📊 Comprehensive Report",
    "sensitivity_1d_desc": "Investigate the effect of one parameter on outputs.",
    "sensitivity_2d_desc": "Investigate the simultaneous effect of two parameters.",
    "sensitivity_tornado_desc": "See the impact of all parameters simultaneously and prioritized.",
    "sensitivity_monte_desc": "Uncertainty analysis with thousands of random simulations.",
    "sensitivity_report_desc": "Complete sensitivity analysis summary with all indicators.",
    "select_param": "🔍 Select Parameter:",
    "param_kf0": "Kf₀ (Filtration Coefficient)",
    "param_sigma": "σ (Reflection Coefficient)",
    "param_pi0": "Pi₀ (Interstitial Pressure)",
    "param_jmax": "Jmax (Lymphatic Capacity)",
    "param_dpi": "Δπ (Oncotic Difference)",
    "param_km": "Km (Michaelis Constant)",
    "param_min": "min:",
    "param_max": "max:",
    "n_points": "Number of Points:",
    "fixed_deltaP": "ΔP (mmHg):",
    "output_type": "Output:",
    "output_jv": "Jv (Filtration)",
    "output_jnet": "Jnet (Net)",
    "output_both": "Both",
    "run_analysis": "🚀 Run",
    "param1": "First Parameter:",
    "param2": "Second Parameter:",
    "heatmap_output": "Output:",
    "heatmap_min": "Minimum",
    "heatmap_max": "Maximum",
    "tornado_output": "Output:","mc_simulations": "Number of Simulations:",
    "mc_uncertainty": "Uncertainty Level:",
    "mc_low": "Low (±5%)",
    "mc_medium": "Medium (±15%)",
    "mc_high": "High (±30%)",
    "mc_mean": "Mean",
    "mc_ci": "95% CI",
    "mc_risk": "Ascites Risk",
    "mc_confidence": "Confidence Level",
    "report_generate": "📊 Generate Report",
    "report_param": "Parameter",
    "report_base": "Base Value",
    "report_min": "Jv_min",
    "report_max": "Jv_max",
    "report_sensitivity": "Sensitivity",
    "report_status": "Status",
    "status_low": "Low",
    "status_medium": "Medium",
    "status_high": "High",
    "download_csv": "📥 Download Report (CSV)","nnn" :"Prioritizing parameters",
    "sens_jv": "Jv Sensitivity",
    "sens_jnet": "Jnet Sensitivity",
    "sens_range": "Range",
    "sens_threshold": "Zero-Crossing",
    "has_threshold": "Has",
    "no_threshold": "None",
    "sens_effect": "Effect of {param} on Outputs",
    "sens_base": "Base Value",
    "sens_high": "High",
    "sens_low": "Low",
    "sens_heatmap_title": "Heatmap: {p1} vs {p2}",
    "sens_tornado_title": "Tornado Diagram: Effect of Parameters on {output}",
    "sens_mc_title": "Jnet Distribution in {n} Simulations",
    "sens_mc_ci_label": "Ascites Threshold",
    "sens_box_title": "Jv and Jnet Distribution",
    
   
    "bernoulli_title": "⚡ Bernoulli Sensitivity Analysis (Hemodynamics)",
    "bernoulli_subtitle": "Sensitivity analysis of parameters affecting pressure drop, α and flow rate",
    "bernoulli_1d": "📈 One-Dimensional (Bernoulli)",
    "bernoulli_2d": "🎯 Two-Dimensional (Bernoulli)",
    "bernoulli_report": "📊 Bernoulli Report",
    "bernoulli_desc": "Investigate the effect of hemodynamic parameters on α, pressure drop and flow rate.",
    "param_qportal": "Q_portal (Portal Vein Flow)",
    "param_qartery": "Q_artery (Hepatic Artery Flow)",
    "param_aportal": "A_portal (Portal Vein Area)",
    "param_ahepatic": "A_hepatic (Hepatic Vein Area)",
    "param_h": "h (Height Difference)",
    "param_r0": "r₀ (Sinusoid Radius)",
    "param_L": "L (Sinusoid Length)",
    "param_beta": "β (Tapering Coefficient)",
    "param_mu": "μ∞ (Viscosity)",
    "param_tau": "τy (Yield Stress)",
    "output_alpha": "α (Effective Parameter)",
    "output_dpsin": "ΔP_sin (Sinusoid Drop)",
    "output_dptotal": "ΔP_total (Total Drop)",
    "output_qtotal": "Q_total (Total Flow)",
    "bernoulli_effect": "Effect of {param} on Hemodynamic Parameters",
    "bernoulli_heatmap": "Bernoulli Heatmap: {p1} vs {p2}",
    "bernoulli_report_title": "📊 Bernoulli Sensitivity Analysis Report",
    "bernoulli_sensitivity": "α Sensitivity",
    "bernoulli_alpha_min": "α_min",
    "bernoulli_alpha_max": "α_max",
    "bernoulli_high": "High",
    "bernoulli_low": "Low",
    "bernoulli_medium": "Medium","sens_medium":"Medium", "mc_ci_label":"Ascites Threshold","sens_anz":"📊Sensitivity Analysis Table","sens_anz2":"Comparison of Parameter Sensitivity Analysis"
        },
    "fa": {
    "app_title": "🩸 شبیه‌ساز تراوش کبد",
    "app_subtitle": "ساخته شده بر اساس مقاله *شبیه‌سازی جریان خون در کبد بر اساس اصول و معادلات مکانیک سیالات*، دبیرستان ماندگار البرز",
    
   
    "settings": "⚙️ تنظیمات",
    "mode_label": "حالت α",
    "mode_auto": "🔄 خودکار",
    "mode_manual": "✋ دستی",
    "hemo_params": "پارامترهای همودینامیک",
    "portal_flow": "دبی ورید باب (L/min)",
    "artery_flow": "دبی سرخرگ کبدی (L/min)",
    "portal_area": "سطح مقطع ورید باب (cm²)",
    "hepatic_area": "سطح مقطع ورید فوق‌کبدی (cm²)",
    "height_diff": "اختلاف ارتفاع h (cm)",
    "sinusoid_params": "پارامترهای سینوزوئیدی (مدل کاسون)",
    "mu_inf": "μ∞ (Pa·s)",
    "tau_y": "τy (Pa)",
    "r0": "r₀ (μm)",
    "L": "L (μm)",
    "beta": "β (ضریب مخروطی شدن)",
    "filtration_params": "پارامترهای تراوش (استارلینگ)",
    "kf0": "Kf₀",
    "sigma": "σ",
    "pi0": "Pi₀ (mmHg)",
    "dpi": "Δπ (mmHg)",
    "lymph_params": "پارامترهای لنفاوی",
    "jmax": "Jmax (ml/min)",
    "km": "Km (mmHg)",
    "max_dp": "حداکثر ΔP (mmHg)",
    "manual_alpha": "α (دستی)",
          "preset_title": "📋 سناریوهای بالینی",
        "preset_custom": "🔘 سفارشی (دستی)",
        "preset_normal": "🟢 سالم (طبیعی)",
        "preset_mild": "🟡 پرفشاری پورتال خفیف",
        "preset_cirrhosis": "🔴 سیروز با آسیت",
        "preset_refractory": "🔴🔴 آسیت مقاوم به درمان",
        "preset_apply": "✅ اعمال سناریو",
        "preset_loaded": "✅ سناریوی '{name}' بارگذاری شد. برای اعمال، روی 'اعمال سناریو' کلیک کنید.",
        
        "clear_cache": "🗑️ پاک‌سازی کش",
        "cache_cleared": "✅ کش با موفقیت پاک شد!",
        
        "pdf_report": "📄 گزارش PDF",
        "pdf_download": "📥 دانلود گزارش کامل",
        "pdf_title": "گزارش همودینامیک کبد",
        "pdf_date": "تاریخ",
        "pdf_params": "۱. پارامترهای ورودی",
        "pdf_results": "۲. نتایج",
        "pdf_clinical": "۳. تفسیر بالینی",
        "pdf_status": "وضعیت",
        "pdf_description": "توضیحات",
        "pdf_ascites": "پیش‌بینی آسیت",
        "pdf_install_warning": "⚠️ برای دانلود PDF، کتابخانه‌ی fpdf را نصب کنید: `pip install fpdf`",
        
        "upload_csv": "📤 بارگذاری داده‌های بیمار (CSV)",
        "upload_help": "انتخاب فایل CSV با ستون‌های: ΔP, Kf0, sigma, Pi0, Jmax, Km, dPi",
        "upload_run": "🚀 اجرای مدل روی داده‌ها",
        "upload_status": "⚠️ خطر آسیت",
        "upload_compensated": "✅ جبران‌شده",
        "upload_download": "📥 دانلود نتایج (CSV)",
        "upload_error": "❌ خطا در خواندن فایل: {e}",
        
        "3d_title": "📊 نمودار سه‌بعدی تعاملی",
        "3d_info": "ℹ️ برای رسم نمودار سه‌بعدی، حداقل دو پارامتر را انتخاب کنید:",
        "3d_param1": "پارامتر اول (X)",
        "3d_param2": "پارامتر دوم (Y)",
        "3d_plot": "🎲 رسم نمودار ۳D",
        "3d_title_plot": "تأثیر هم‌زمان {p1} و {p2} بر Jv",
        "3d_x": "{p1}",
        "3d_y": "{p2}",
        "3d_z": "Jv (ml/min)",
        
        "reset_title": "🔄 بازنشانی همه‌ی تنظیمات",
        "reset_confirm": "⚠️ آیا مطمئن هستید که می‌خواهید همه‌ی تنظیمات را بازنشانی کنید؟",
        
        "validation_warning_flow": "⚠️ دبی ورید باب باید مثبت باشد!",
        "validation_warning_area": "⚠️ سطح مقطع باید بزرگتر از صفر باشد!",
    
    "alpha": "α",
    "shear_rate": "نرخ برش",
    "mu_app": "μ ظاهری",
    "dp_total": "ΔP کل",
    "calc_details": "📊 جزئیات محاسبات","afz":"افزایش پارامتر","kahesh":"کاهش پارامتر",
    "total_flow": "دبی کل",
    "portal_vel": "سرعت ورید باب",
    "hepatic_vel": "سرعت ورید فوق‌کبدی",
    "sinusoid_drop": "افت سینوزوئیدی",
    "height_drop": "افت ارتفاع",
    "kinetic_drop": "افت جنبشی",
    "total_drop": "کل افت",
    "auto_results": "📊 نتایج تحلیل در حالت خودکار",
    "deltaP": "ΔP",
    "kf_eff": "Kf",
    "pi_eff": "Pi",
    "jv": "Jv",
    "jnet": "Jnet",
    "clinical_title": "🏥 (ΔP = {dp:.2f})اختلاف فشار",
    "status": "وضعیت",
    "fluid_status": "وضعیت مایعات",
    "ascites_pred": "پیش‌بینی آسیت",
    "key_values": "📊 مقادیر در نقاط مختلف",
    "dp_mmHg": "ΔP (mmHg)",
    "jlymph": "Jlymph (ml/min)",
    "manual_results": "📊 نتایج تحلیل در حالت دستی",
    "filtration_curves": "📊 منحنی‌های تراوش",
    "jv_curve": "Jv (تراوش)",
    "jlymph_curve": "Jlymph (لنفاوی)",
    "jnet_curve": "Jnet (خالص)",
    "zero_line": "J = 0",
    "threshold_line": "آستانه ۱۲ mmHg",
    "curves_title": "📊 منحنی‌های تراوش، تخلیه لنفاوی و نرخ خالص تجمع",
    "xaxis_dp": "ΔP (mmHg)",
    "yaxis_flow": "نرخ جریان (ml/min)",
    "nonlinear_behavior": "📊 رفتار غیرخطی پارامترها",
    "kf_title": "ضریب فیلتراسیون غیرخطی Kf",
    "kf_yaxis": "Kf (ml/min/mmHg)",
    "pi_title": "فشار میان‌بافتی غیرخطی Pi (مکانیسم جبرانی)",
    "pi_yaxis": "Pi (mmHg)",
    "key_points": "📊 مقادیر کلیدی",
    "clinical_interpretation": "🏥 تفسیر بالینی",
    "caption": "α = {alpha:.3f} | h = {h} cm | r₀ = {r0} μm | β = {beta} | Q_total = {q:.1f} L/min | μ = {mu:.5f} Pa·s",
    "info_text": "📌 تفسیر: در فشارهای پایین (< ۱۲ mmHg)، سیستم لنفاوی قادر به تخلیه مایع است. پس از عبور از آستانه، Kf به‌صورت نمایی رشد کرده و تراوش از ظرفیت لنفاوی سبقت می‌گیرد که منجر به «شکست هیدرولیکی» و تشکیل آسیت می‌شود.",
    "clinical_expander": "📖 تفسیر بالینی کامل",
    "table_title": "📌 محدوده‌های بالینی",
    "table_col1": "محدوده ΔP (mmHg)",
    "table_col2": "وضعیت",
    "table_col3": "خطر آسیت",
    "table_col4": "مکانیسم غالب",
    "row1_1": "< ۸",
    "row1_2": "طبیعی (فیزیولوژیک)","param12":"مقدار پارامتر",
    "row1_3": "🟢 بسیار کم",
    "row1_4": "تعادل استارلینگ + تخلیه لنفاوی",
    "row2_1": "۸ – ۱۲",
    "row2_2": "مرز هشدار",
    "row2_3": "🟡 افزایش جزئی",
    "row2_4": "شروع رشد غیرخطی Kf",
    "row3_1": "۱۲ – ۱۶",
    "row3_2": "پرفشاری پورتال خفیف-متوسط",
    "row3_3": "🔴 متوسط",
    "row3_4": "**شکست هیدرولیکی** (Jnet > 0)",
    "row4_1": "> ۱۶",
    "row4_2": "پرفشاری شدید",
    "row4_3": "🔴 بالا",
    "row4_4": "رشد نمایی تراوش + اشباع لنفاوی", "nnn": "اولویت‌بندی پارامترها",
    "mechanisms_title": "🔬 مکانیسم‌های کلیدی",
    "mech1": "1. شکست هیدرولیکی: در ΔP ≥ ۱۲ mmHg، ترکیب کاهش ویسکوزیته و افزایش Kf منجر به افزایش شتاب‌دار تراوش می‌شود.",
    "mech2": "2.آستانه بالینی: نقطه ۱۲ mmHg به‌عنوان آستانه تشکیل آسیت با مشاهدات بالینی (Garcia-Tsao et al., 2017) همخوانی دارد.",
    "mech3": "3. مکانیسم جبرانی: افزایش Pi تا حدی تراوش را کاهش می‌دهد، اما در فشارهای بالا ناکافی است.",
    "mech4": "4. اشباع لنفاوی: سیستم لنفاوی با ظرفیت محدود، پس از عبور از آستانه قادر به تخلیه کامل مایع نیست.",
    "clinical_app": "🏥 کاربرد بالینی",
    "clinical_app_text": "این مدل می‌تواند به‌عنوان یک ابزار بالینی ساده و ارزان‌قیمت برای:",
    "clinical_app_1": "- پیش‌بینی خطر آسیت در بیماران با پرفشاری پورتال",
    "clinical_app_2": "- ارزیابی پاسخ به درمان‌های کاهش‌دهنده فشار پورتال",
    "clinical_app_3": "- طراحی مطالعات CFD آینده برای شبیه‌سازی دقیق‌تر",
    "comparison_title": "📊 مقایسه با مدل‌های پیشین",
    "comparison_col1": "ویژگی",
    "comparison_col2": "Siggers (2013)",
    "comparison_col3": "Dongaonkar et al. (2018)",
    "comparison_col4": "Dongaonkar et al. (2020)",
    "comparison_col5": "**مدل حاضر**",
    "comp_bernoulli": "برنولی اصالح‌شده",
    "comp_viscous": "افت فشار ویسکوزی",
    "comp_kf": "Kf غیرخطی وابسته به فشار",
    "comp_pi": "Pi غیرخطی وابسته به فشار",
    "comp_negative": "شار منفی تراوش",
    "comp_casson": "مدل کاسون (غیرنیوتنی)",
    "comp_lymph": "تخلیه لنفاوی (مایکلـیس-منتن)",
    "innovation_title": "**نوآوری اصلی:**",
    "innovation_text": "ترکیب معادله برنولی اصالح‌شده، قانون پوازوی با شعاع متغیر (β)، مدل کاسون، و معادله استارلینگ با Kf و Pi غیرخطی در یک چارچوب یکپارچه.",
    "footer": "🩸 شبیه‌ساز همودینامیک کبد | بر اساس مقاله: شبیه‌سازی جریان خون در کبد بر اساس اصول و معادلات مکانیک سیالات | پژوهش‌سرای ماندگار البرز | سال تحصیلی ۱۴۰۵-۱۴۰۴",
    "lang_label": "زبان",
    "lang_en": "🇬🇧 English",
    "lang_fa": "🇮🇷 فارسی",
    
    
    "sensitivity_title": "📊 تحلیل حساسیت پیشرفته",
    "sensitivity_subtitle": "تحلیل حساسیت پیشرفته با قابلیت Monte Carlo، Heatmap و Tornado Diagram",
    "sensitivity_1d": "📈 یک‌بعدی",
    "sensitivity_2d": "🎯 دو‌بعدی (Heatmap)",
    "sensitivity_tornado": "🌪️ Tornado Diagram",
    "sensitivity_monte": "🎲 Monte Carlo",
    "sensitivity_report": "📊 گزارش جامع",
    "sensitivity_1d_desc": "تأثیر تغییرات یک پارامتر بر خروجی‌ها را بررسی کن.",
    "sensitivity_2d_desc": "تأثیر هم‌زمان دو پارامتر بر خروجی را بررسی کن.",
    "sensitivity_tornado_desc": "تأثیر همه پارامترها را به‌صورت هم‌زمان و اولویت‌بندی‌شده ببین.",
    "sensitivity_monte_desc": "تحلیل عدم‌قطعیت با هزاران شبیه‌سازی تصادفی.",
    "sensitivity_report_desc": "خلاصه کامل تحلیل حساسیت با تمام شاخص‌ها.",
    "select_param": "🔍 انتخاب پارامتر:",
    "param_kf0": "Kf₀ (ضریب فیلتراسیون)",
    "param_sigma": "σ (ضریب انعکاس)",
    "param_pi0": "Pi₀ (فشار میان‌بافتی)",
    "param_jmax": "Jmax (ظرفیت لنفاوی)",
    "param_dpi": "Δπ (اختلاف انکوتیک)",
    "param_km": "Km (ثابت مایکل‌یس)",
    "param_min": "min:",
    "param_max": "max:",
    "n_points": "تعداد نقاط:",
    "fixed_deltaP": "فشار ΔP (mmHg):",
    "output_type": "خروجی:",
    "output_jv": "Jv (تراوش)",
    "output_jnet": "Jnet (خالص)",
    "output_both": "هر دو",
    "run_analysis": "🚀 اجرا",
    "param1": "پارامتر اول:",
    "param2": "پارامتر دوم:",
    "heatmap_output": "خروجی:",
    "heatmap_min": "حداقل",
    "heatmap_max": "حداکثر",
    "tornado_output": "خروجی:",
    "mc_simulations": "تعداد شبیه‌سازی:",
    "mc_uncertainty": "سطح عدم‌قطعیت:",
    "mc_low": "کم (±5%)",
    "mc_medium": "متوسط (±15%)",
    "mc_high": "زیاد (±30%)",
    "mc_mean": "میانگین",
    "mc_ci": "فاصله اطمینان 95%",
    "mc_risk": "خطر آسیت",
    "mc_confidence": "ضریب اطمینان",
    "report_generate": "📊 تولید گزارش",
    "report_param": "پارامتر",
    "report_base": "مقدار پایه",
    "report_min": "Jv_min",
    "report_max": "Jv_max",
    "report_sensitivity": "حساسیت",
    "report_status": "وضعیت",
    "status_low": "پایین",
    "status_medium": "متوسط",
    "status_high": "بالا",
    "download_csv": "📥 دانلود گزارش (CSV)",
    "sens_jv": "حساسیت Jv",
    "sens_jnet": "حساسیت Jnet",
    "sens_range": "محدوده",
    "sens_threshold": "نقطه تلاقی با صفر",
    "has_threshold": "دارد",
    "no_threshold": "ندارد",
    "sens_effect": "تأثیر {param} بر خروجی‌ها",
    "sens_base": "مقدار پایه",
    "sens_high": "زیاد",
    "sens_low": "کم",
    "sens_heatmap_title": "Heatmap: {p1} vs {p2}",
    "sens_tornado_title": "Tornado Diagram: تأثیر پارامترها بر {output}",
    "sens_mc_title": "توزیع Jnet در {n} شبیه‌سازی",
    "sens_mc_ci_label": "آستانه آسیت",
    "sens_box_title": "توزیع Jv و Jnet",
    

    "bernoulli_title": "⚡ تحلیل حساسیت برنولی (همودینامیک)",
    "bernoulli_subtitle": "تحلیل حساسیت پارامترهای مؤثر بر افت فشار، α و دبی",
    "bernoulli_1d": "📈 یک‌بعدی (برنولی)",
    "bernoulli_2d": "🎯 دو‌بعدی (برنولی)",
    "bernoulli_report": "📊 گزارش برنولی",
    "bernoulli_desc": "تأثیر تغییرات پارامترهای همودینامیک بر α، افت فشار و دبی را بررسی کن.",
    "param_qportal": "Qportal (دبی ورید باب)",
    "param_qartery": "Qartery (دبی سرخرگ کبدی)",
    "param_aportal": "Aportal (سطح مقطع ورید باب)",
    "param_ahepatic": "Ahepatic (سطح مقطع ورید فوق‌کبدی)",
    "param_h": "h (اختلاف ارتفاع)",
    "param_r0": "r₀ (شعاع سینوزوئید)",
    "param_L": "L (طول سینوزوئید)",
    "param_beta": "β (ضریب مخروطی)",
    "param_mu": "μ∞ (ویسکوزیته)",
    "param_tau": "τy (تنش تسلیم)",
    "output_alpha": "α (پارامتر مؤثر)",
    "output_dpsin": "ΔP_sin (افت سینوزوئیدی)",
    "output_dptotal": "ΔP_total (افت کل)",
    "output_qtotal": "Q_total (دبی کل)",
    "bernoulli_effect": "تأثیر {param} بر پارامترهای همودینامیک",
    "bernoulli_heatmap": "Heatmap برنولی: {p1} vs {p2}",
    "bernoulli_report_title": "📊 گزارش تحلیل حساسیت برنولی",
    "bernoulli_sensitivity": "حساسیت α",
    "bernoulli_alpha_min": "α min",
    "bernoulli_alpha_max": "α max",
    "bernoulli_high": "زیاد",
    "bernoulli_low": "کم","P_hep":"فشار سیاهرگ فوق کبدی",
    "bernoulli_medium": "متوسط","sens_medium":"متوسط","sens_anz":"📊جدول شاخص‌های حساسیت","sens_anz2":"مقایسه تحلیل حساسیت پارامتر ها"
    }
    
}


# ======================== مدیریت زبان ========================
if "lang" not in st.session_state:
    st.session_state.lang = "en"

def set_lang_en():
    st.session_state.lang = "en"
    st.rerun()

def set_lang_fa():
    st.session_state.lang = "fa"
    st.rerun()

# ======================== سایدبار ========================
with st.sidebar:
    # انتخابگر زبان
    st.write("🌐 زبان / Language")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇬🇧 English", use_container_width=True):
            
            set_lang_en()
    with col2:
        if st.button("🇮🇷فارسی", use_container_width=True):
           
            set_lang_fa()
    st.divider()

    
    lang = st.session_state.lang
    
    t = TEXTS[lang]
    
     
     
    
   
    if lang == "fa":
        
       st.markdown("""
    <style>

        
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stMarkdown p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] .stSelectbox div,
        section[data-testid="stSidebar"] .stRadio div,
        section[data-testid="stSidebar"] .stNumberInput div,
        .stAlert, .stAlert p, .stAlert div {
            text-align: right !important;
            direction: rtl !important;
        }
        
        section[data-testid="stSidebar"] .stSlider div {
            direction: ltr !important;
        }
    </style>
""", unsafe_allow_html=True)
     
     
    st.header(t["settings"])
    
    mode = st.radio(t["mode_label"], [t["mode_manual"], t["mode_auto"]])
    
    st.header(t["hemo_params"])
    
    if mode == t["mode_manual"]:
        alpha = st.slider(t["manual_alpha"], 0.20, 0.95, 0.54, 0.01)
    else:
        alpha = None
    
    Q_portal = (st.number_input(t["portal_flow"], 0.3, 2.0, 1.1, 0.05)/ 1000/ 60)
    Q_artery = (
        st.number_input(
            t["artery_flow"], 0.1, 0.8, 0.35, 0.05
        )
        / 1000
        / 60
    )
    
    A_portal = (
        st.number_input(
            t["portal_area"], 0.5, 5.0, 1.1, 0.1
        )
        * 1e-4
    )
    A_hepatic = (
        st.number_input(
            t["hepatic_area"], 0.5, 10.0, 0.6, 0.1
        )
        * 1e-4
    )
    P_hep = (
        st.number_input(
            t["P_hep"], 0.0, 8.0, 4.0, 0.5
        )
        
    )
    h_cm = st.number_input(
        t["height_diff"], 0.0, 10.0, 4.0, 0.1
    )
    h = h_cm / 100
    
    st.header(t["sinusoid_params"])
    
    mu_inf = st.number_input(
        t["mu_inf"],
        0.001, 0.01, 0.0040, 0.0005,
        format="%.4f"
    )
    
    tau_y = st.number_input(
        t["tau_y"],
        0.001, 0.01, 0.005, 0.0005,
        format="%.4f"
    )
    
    r0_um = st.number_input(
        t["r0"], 2.0, 6.0, 4.0, 0.05
    )
    r0 = r0_um * 1e-6
    
    L_um = st.number_input(
        t["L"], 100, 500, 365, 5
    )
    L = L_um * 1e-6
    
    beta = st.number_input(t["beta"], 0.0, 0.8, 0.10, 0.01)
    
    st.header(t["filtration_params"])
    
    Kf0 = st.slider(t["kf0"], 1.0, 8.0, 3.0, 0.1)
    sigma = st.slider(t["sigma"], 0.1, 0.4, 0.22, 0.01)
    Pi0 = st.slider(t["pi0"], 0.1, 2.0, 0.5, 0.1)
    dPi = st.slider(t["dpi"], 20, 25, 22, 1)
    
    st.header(t["lymph_params"])
    Jmax = st.number_input(
        t["jmax"], 10, 50, 40, 1
    )
    
    Km = st.number_input(
        t["km"], 0.1, 2.0, 0.74, 0.01
    )
    
    max_deltaP = st.slider(t["max_dp"], 12, 30, 20)
    




if mode == t["mode_auto"]:
    alpha, Q_total, vp, vh, dp_sin, dp_h, dp_v, dp_total, mu_app = calc_alpha(
        Q_portal, Q_artery, A_portal, A_hepatic, h, r0, beta, L, mu_inf, tau_y
    )
else:
    Q_total = Q_portal + Q_artery
    vp = Q_portal / A_portal if A_portal > 0 else 0
    vh = Q_total / A_hepatic if A_hepatic > 0 else 0
    gamma_dot = calc_shear_rate(Q_total, r0)
    mu_app = calc_mu_apparent(mu_inf, tau_y, gamma_dot)
    dp_sin = calc_sinusoid_pressure_drop(Q_total, mu_app, L, r0, beta)
    dp_h = rho_blood * g * h
    dp_v = 0.5 * rho_blood * (vh**2 - vp**2)
    dp_total = dp_sin + dp_h + dp_v


params = {
    'alpha': alpha,
    'Kf0': Kf0,
    'sigma': sigma,
    'Pi0': Pi0,
    'Jmax': Jmax,
    'Km': Km,
    'mu_inf': mu_inf,
    'tau_y': tau_y,
    'r0': r0_um,
    'beta': beta,
    'dPi': dPi
}


def color_jnet(val):
    if val <= 0:
        return 'background-color: #d4edda'
    elif val < 5:
        return 'background-color: #fff3cd'
    elif val < 15:
        return 'background-color: #ffe5b4'
    else:
        return 'background-color: #f8d7da'


st.title(t["app_title"])
st.markdown(t["app_subtitle"])


if mode == t["mode_auto"]:
    col1, col2 = st.columns(2)

    with col1:
        st.metric(t["alpha"], f"{alpha:.3f}")
        st.metric(t["shear_rate"], f"{calc_shear_rate(Q_total, r0):.1f} s⁻¹")
    with col2:
        st.metric(t["mu_app"], f"{mu_app:.5f} Pa·s")
        st.metric(t["dp_total"], f"{(dp_total / mmHg_to_Pa):.2f} mmHg")

   
    with st.sidebar.expander(t["calc_details"]):
        st.write(f"{t['total_flow']}: {(Q_total * 1000 * 60):.2f} L/min")
        st.write(f"{t['portal_vel']}: {vp:.3f} m/s")
        st.write(f"{t['hepatic_vel']}: {vh:.3f} m/s")
        st.write(f"{t['shear_rate']}: {calc_shear_rate(Q_total, r0):.1f} s⁻¹")
        st.write(f"{t['mu_app']}: {mu_app:.5f} Pa·s")
        st.write(f"{t['sinusoid_drop']}: {dp_sin / mmHg_to_Pa:.2f} mmHg")
        st.write(f"{t['height_drop']}: {dp_h / mmHg_to_Pa:.2f} mmHg")
        st.write(f"{t['kinetic_drop']}: {dp_v / mmHg_to_Pa:.2f} mmHg")
        st.write(f"{t['total_drop']}: {dp_total / mmHg_to_Pa:.2f} mmHg")



if mode == t["mode_auto"]:
    # ====== حالت خودکار ======
    st.subheader(t["auto_results"])
    
    deltaP_analysis = (dp_total / mmHg_to_Pa)
    Kf = calc_Kf_nonlinear(Kf0, deltaP_analysis)
    Pi = calc_Pi_nonlinear(Pi0, deltaP_analysis)
    Jv = calc_Jv(deltaP_analysis, Kf, alpha, sigma, Pi, dPi, P_hep)
    Jlymph = calc_Jlymph(Jmax, Km, Pi)
    Jnet = calc_Jnet(Jv, Jlymph)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(t["deltaP"], f"{deltaP_analysis:.2f} mmHg")
        st.metric(t["kf_eff"], f"{Kf:.3f}")

    with col2:
        st.metric(t["pi_eff"], f"{Pi:.2f} mmHg")
        st.metric(t["jv"], f"{Jv:.2f} ml/min")
    
    st.metric(t["jnet"], f"{Jnet:.2f} ml/min")

    clinical = get_clinical_interpretation(deltaP_analysis, Jv, Jnet, lang=lang)

    st.info(
        f"""
        {t['clinical_title'].format(dp=deltaP_analysis)}

        • {t['status']}: {clinical['color']} {clinical['status']}
        • {clinical['description']}
        • {t['fluid_status']}: {clinical['fluid_status']}
        • {clinical['ascites_prediction']}
        """
    )

   
    st.subheader(t["key_values"])

    key_points = [4, 8, 12, 16, 20]
    data = []

    for dp in key_points:
        Kf = calc_Kf_nonlinear(Kf0, dp)
        Pi = calc_Pi_nonlinear(Pi0, dp)
        Jv = calc_Jv(dp, Kf, alpha, sigma, Pi, dPi, P_hep)
        Jlymph = calc_Jlymph(Jmax, Km, Pi)
        Jnet = calc_Jnet(Jv, Jlymph)

        data.append({t["dp_mmHg"]: dp,
            t["kf_eff"]: round(Kf, 3),
            t["pi_eff"]: round(Pi, 2),
            t["jv"]: round(Jv, 2),
            t["jlymph"]: round(Jlymph, 2),
            t["jnet"]: round(Jnet, 2)
        })

    df = pd.DataFrame(data)

    st.dataframe(
        df.style.map(color_jnet, subset=[t["jnet"]]),
        use_container_width=True,
        hide_index=True
    )

else:
    
    dp_totall = st.number_input(t["deltaP"], 0.0, 20.0, 6.0)
    deltaP_analysis = dp_totall
    Kf = calc_Kf_nonlinear(Kf0, deltaP_analysis)
    Pi = calc_Pi_nonlinear(Pi0, deltaP_analysis)
    Jv = calc_Jv(deltaP_analysis, Kf, alpha, sigma, Pi, dPi, P_hep)
    Jlymph = calc_Jlymph(Jmax, Km, Pi)
    Jnet = calc_Jnet(Jv, Jlymph)

    st.subheader(t["manual_results"])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(t["deltaP"], f"{deltaP_analysis:.2f} mmHg")
        st.metric(t["kf_eff"], f"{Kf:.3f}")

    with col2:
        st.metric(t["pi_eff"], f"{Pi:.2f} mmHg")
        st.metric(t["jv"], f"{Jv:.2f} ml/min")
    
    st.metric(t["jnet"], f"{Jnet:.2f} ml/min")

    clinical = get_clinical_interpretation(deltaP_analysis, Jv, Jnet, lang=lang)

    st.info(
        f"""
        {t['clinical_title'].format(dp=deltaP_analysis)}

        • {t['status']}: {clinical['color']} {clinical['status']}
        • {clinical['description']}
        • {t['fluid_status']}: {clinical['fluid_status']}
        • {clinical['ascites_prediction']}
        """
    )

    # ====== نمودارهای کامل (فقط در حالت دستی) ======
    st.subheader(t["filtration_curves"])

    deltaP_range = np.linspace(0, max_deltaP, 300)

    Jv_list = []
    Kf_list = []
    Pi_list = []
    Jlymph_list = []
    Jnet_list = []

    for dp in deltaP_range:
        Kf = calc_Kf_nonlinear(Kf0, dp)
        Pi = calc_Pi_nonlinear(Pi0, dp)
        Jv = calc_Jv(dp, Kf, alpha, sigma, Pi, dPi, P_hep)
        Jlymph = calc_Jlymph(Jmax, Km, Pi)
        Jnet = calc_Jnet(Jv, Jlymph)

        Jv_list.append(Jv)
        Kf_list.append(Kf)
        Pi_list.append(Pi)
        Jlymph_list.append(Jlymph)
        Jnet_list.append(Jnet)

    # نمودار اصلی
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=deltaP_range,
            y=Jv_list,
            mode='lines',
            name=t["jv_curve"],
            line=dict(color='blue', width=3)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=deltaP_range,
            y=Jlymph_list,
            mode='lines',
            name=t["jlymph_curve"],
            line=dict(color='green', width=3, dash='dash')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=deltaP_range,
            y=Jnet_list,
            mode='lines',
            name=t["jnet_curve"],
            line=dict(color='red', width=3, dash='dot'),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        )
    )

    fig.add_hline(
        y=0,
        line_dash='dot',
        line_color='gray',
        annotation_text=t["zero_line"],
        annotation_position='bottom right'
    )

    fig.add_vline(
        x=12,
        line_dash='dot',
        line_color='red',
        annotation_text=t["threshold_line"],
        annotation_position='top'
    )

    fig.update_layout(
        title=t["curves_title"],
        xaxis_title=t["xaxis_dp"],
        yaxis_title=t["yaxis_flow"],
        template='plotly_white',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # ====== نمودارهای Kf و Pi ======
    st.subheader(t["nonlinear_behavior"])

    col1, col2 = st.columns(2)

    with col1:
        fig_kf = go.Figure()

        fig_kf.add_trace(
            go.Scatter(
                x=deltaP_range,
                y=Kf_list,
                mode='lines',
                name=t["kf_eff"],
                line=dict(color='purple', width=3)
            )
        )

        fig_kf.add_hline(
            y=Kf0,
            line_dash='dot',line_color='gray',
            annotation_text=f'{t["kf_eff"]}₀ = {Kf0}',
            annotation_position='bottom right'
        )

        fig_kf.add_vline(
            x=12,
            line_dash='dot',
            line_color='red',
            annotation_text=t["threshold_line"],
            annotation_position='top'
        )

        fig_kf.update_layout(
            title=t["kf_title"],
            xaxis_title=t["xaxis_dp"],
            yaxis_title=t["kf_yaxis"],
            template='plotly_white',
            height=350
        )

        st.plotly_chart(fig_kf, use_container_width=True)

    with col2:
        fig_pi = go.Figure()

        fig_pi.add_trace(
            go.Scatter(
                x=deltaP_range,
                y=Pi_list,
                mode='lines',
                name=t["pi_eff"],
                line=dict(color='orange', width=3)
            )
        )

        fig_pi.add_hline(
            y=Pi0,
            line_dash='dot',
            line_color='gray',
            annotation_text=f'{t["pi_eff"]}₀ = {Pi0}',
            annotation_position='bottom right'
        )

        fig_pi.add_vline(
            x=12,
            line_dash='dot',
            line_color='red',
            annotation_text=t["threshold_line"],
            annotation_position='top'
        )

        fig_pi.update_layout(
            title=t["pi_title"],
            xaxis_title=t["xaxis_dp"],
            yaxis_title=t["pi_yaxis"],
            template='plotly_white',
            height=350
        )

        st.plotly_chart(fig_pi, use_container_width=True)

    # ====== جدول مقادیر کلیدی ======
    st.subheader(t["key_points"])

    key_points = [4, 8, 12, 16, 20]
    data = []

    for dp in key_points:
        idx = int(dp / max_deltaP * len(deltaP_range))
        if idx >= len(deltaP_range):
            idx = len(deltaP_range) - 1

        data.append({
            t["dp_mmHg"]: dp,
            t["kf_eff"]: round(Kf_list[idx], 3),
            t["pi_eff"]: round(Pi_list[idx], 2),
            t["jv"]: round(Jv_list[idx], 2),
            t["jlymph"]: round(Jlymph_list[idx], 2),
            t["jnet"]: round(Jnet_list[idx], 2)
        })

    df = pd.DataFrame(data)

    st.dataframe(
        df.style.map(color_jnet, subset=[t["jnet"]]),
        use_container_width=True,
        hide_index=True
    )

    # ====== تفسیر بالینی ======
    st.subheader((t["clinical_interpretation"]))

    cols = st.columns(3)

    for i, dp in enumerate([8, 12, 16]):
        idx = int(dp / max_deltaP * len(deltaP_range))
        if idx >= len(deltaP_range):
            idx = len(deltaP_range) - 1

        res = {
            'deltaP': dp,
            'Jv': Jv_list[idx],
            'Jnet': Jnet_list[idx]
        }

        clinical = get_clinical_interpretation(dp, res['Jv'], res['Jnet'], lang=lang)

        with cols[i]:
            color_bg = '#d4edda' if clinical['status'] == (t["row1_2"] if lang == "fa" else "Normal (Physiological)") else '#fff3cd' if clinical['status'] == (t["row2_2"] if lang == "fa" else "Warning Zone") else '#f8d7da'

            st.markdown(f"""
            <div style="
                background-color: {color_bg};
                padding: 15px;
                border-radius: 10px;
                margin: 5px 0;
                border: 1px solid #ddd;
            ">
                <h4 style="margin: 0; text-align: center;">{clinical['color']} ΔP = {dp} mmHg</h4>
                <hr style="margin: 10px 0;">
                <b>{t['status']}:</b> {clinical['status']}<br>
                <b>{t['jv']}:</b> {res['Jv']:.2f} ml/min<br>
                <b>{t['jnet']}:</b> {res['Jnet']:.2f} ml/min<br>
                <b>{clinical['ascites_prediction']}</b>
            </div>
            """, unsafe_allow_html=True)

    st.caption(
        t["caption"].format(
            alpha=alpha,
            h=h_cm,
            r0=r0_um,
            beta=beta,
            q=(Q_total * 1000 * 60),
            mu=mu_app
        )
    )

    st.info(t["info_text"])
    
with st.expander(t["clinical_expander"], expanded=False):
    st.markdown(f"""
    ### {t["table_title"]}

    | {t["table_col1"]} | {t["table_col2"]} | {t["table_col3"]} | {t["table_col4"]} |
    |---|---|---|---|
    | {t["row1_1"]} | {t["row1_2"]} | {t["row1_3"]} | {t["row1_4"]} |
    | {t["row2_1"]} | {t["row2_2"]} | {t["row2_3"]} | {t["row2_4"]} |
    | {t["row3_1"]} | {t["row3_2"]} | {t["row3_3"]} | {t["row3_4"]} |
    | {t["row4_1"]} | {t["row4_2"]} | {t["row4_3"]} | {t["row4_4"]} |

    ### {t["mechanisms_title"]}

    {t["mech1"]}

    {t["mech2"]}

    {t["mech3"]}

    {t["mech4"]}

    ### {t["clinical_app"]}

    {t["clinical_app_text"]}

    {t["clinical_app_1"]}
    {t["clinical_app_2"]}
    {t["clinical_app_3"]}
    """)
    
 #===================================================================================================================================   
    
    






# ======================== تحلیل حساسیت برنولی (همودینامیک) ========================
with st.expander(t["bernoulli_title"], expanded=False):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 5px solid #00d2ff;
        color: white;
    ">
        <h4 style="margin: 0; color: #00d2ff;">⚡ {t['bernoulli_title']}</h4>
        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">
            {t['bernoulli_subtitle']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== تب‌های برنولی ==========
    btab1, btab2, btab3 = st.tabs([
        t["bernoulli_1d"],
        t["bernoulli_2d"],
        t["bernoulli_report"]
    ])
    
    # ======================== تب ۱: یک‌بعدی برنولی ========================
    with btab1:
        st.markdown(f"""
        <div style="
            background: #f0f2f6;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="margin: 0; font-size: 13px; color: #555;">
            📌 <b>{t['bernoulli_desc']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            bernoulli_params = {
                t["param_qportal"]: "Q_portal",
                t["param_qartery"]: "Q_artery",
                t["param_aportal"]: "A_portal",
                t["param_ahepatic"]: "A_hepatic",
                t["param_h"]: "h",
                t["param_r0"]: "r0",
                t["param_L"]: "L",
                t["param_beta"]: "beta",
                t["param_mu"]: "mu_inf",
                t["param_tau"]: "tau_y"
            }
            
            selected_bparam = st.selectbox(t["select_param"], list(bernoulli_params.keys()), key="bern_param")
            bparam_key = bernoulli_params[selected_bparam]
            
            # محدوده‌های پیشنهادی برای پارامترهای برنولی
            bernoulli_ranges = {
                "Q_portal": (1.0, 1.3, 1.1, 0.05),
                "Q_artery": (0.1, 0.6, 0.35, 0.05),
                "A_portal": (0.5, 3.0, 1.1, 0.1),
                "A_hepatic": (0.3, 1.2, 0.6, 0.1),
                "h": (3.0, 6.0, 4.0, 0.1),
                "r0": (4.0, 5.0, 4.5, 0.5),
                "L": (250, 400, 340, 5),
                "beta": (0.0, 0.4, 0.05, 0.05),
                "mu_inf": (0.002, 0.005, 0.004, 0.001),
                "tau_y": (0.002, 0.008, 0.005, 0.001)
            }
            
            min_val, max_val, default_val, step = bernoulli_ranges[bparam_key]
            
            # تنظیم min و max بر اساس نوع پارامتر
            if bparam_key in ["Q_portal", "Q_artery"]:
                b_range_min = st.number_input(t["param_min"], min_val, max_val, min_val, step, format="%.2f", key="brmin")
                b_range_max = st.number_input(t["param_max"], min_val, max_val, max_val, step, format="%.2f", key="brmax")
            elif bparam_key in ["A_portal", "A_hepatic"]:
                b_range_min = st.number_input(t["param_min"], min_val, max_val, min_val, step, format="%.1f", key="brmin")
                b_range_max = st.number_input(t["param_max"], min_val, max_val, max_val, step, format="%.1f", key="brmax")
            elif bparam_key in ["h", "r0", "L"]:
                b_range_min = st.number_input(t["param_min"], min_val, max_val, min_val, step, format="%.1f", key="brmin")
                b_range_max = st.number_input(t["param_max"], min_val, max_val, max_val, step, format="%.1f", key="brmax")
            else:
                b_range_min = st.number_input(t["param_min"], min_val, max_val, min_val, step, format="%.3f", key="brmin")
                b_range_max = st.number_input(t["param_max"], min_val, max_val, max_val, step, format="%.3f", key="brmax")
            
            b_n_points = st.slider(t["n_points"], 10, 100, 30, 5, key="bnpts")
            
            b_output_type = st.selectbox(t["output_type"], [t["output_alpha"],
                t["output_dpsin"],
                t["output_dptotal"],
                t["output_qtotal"],
                t["output_both"]
            ], key="bout_type")
            
            run_b1d = st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_b1d")
        
        with col2:
            if run_b1d:
                with st.spinner("Calculating...⏳"):
                    b_param_range = np.linspace(b_range_min, b_range_max, b_n_points)
                    
                    alpha_vals = []
                    dp_sin_vals = []
                    dp_total_vals = []
                    Q_total_vals = []
                    
                    for val in b_param_range:
                        temp_params = params.copy()
                        
                        # ایجاد کپی از پارامترهای اصلی
                        temp_Qp = Q_portal
                        temp_Qa = Q_artery
                        temp_Ap = A_portal
                        temp_Ah = A_hepatic
                        temp_h = h
                        temp_r0 = r0
                        temp_L = L
                        temp_beta = beta
                        temp_mu = mu_inf
                        temp_tau = tau_y
                        
                        # تغییر پارامتر مورد نظر
                        if bparam_key == "Q_portal":
                            temp_Qp = val / 1000 / 60
                        elif bparam_key == "Q_artery":
                            temp_Qa = val / 1000 / 60
                        elif bparam_key == "A_portal":
                            temp_Ap = val * 1e-4
                        elif bparam_key == "A_hepatic":
                            temp_Ah = val * 1e-4
                        elif bparam_key == "h":
                            temp_h = val / 100
                        elif bparam_key == "r0":
                            temp_r0 = val * 1e-6
                        elif bparam_key == "L":
                            temp_L = val * 1e-6
                        elif bparam_key == "beta":
                            temp_beta = val
                        elif bparam_key == "mu_inf":
                            temp_mu = val
                        elif bparam_key == "tau_y":
                            temp_tau = val
                        
                        # محاسبه با پارامتر جدید
                        temp_alpha, temp_Qtotal, _, _, temp_dpsin, _, _, temp_dptotal, _ = calc_alpha(
                            temp_Qp, temp_Qa, temp_Ap, temp_Ah, temp_h, temp_r0, temp_beta, temp_L, temp_mu, temp_tau
                        )
                        
                        alpha_vals.append(temp_alpha)
                        dp_sin_vals.append(temp_dpsin / mmHg_to_Pa)
                        dp_total_vals.append(temp_dptotal / mmHg_to_Pa)
                        Q_total_vals.append(temp_Qtotal * 1000 * 60)
                
                # ====== نمودار ======
                fig_b = go.Figure()
                
                if b_output_type in [t["output_alpha"], t["output_both"]]:
                    fig_b.add_trace(go.Scatter(
                        x=b_param_range, y=alpha_vals,
                        mode='lines+markers',
                        name=t["output_alpha"],
                        line=dict(color='#00d2ff', width=3),
                        marker=dict(size=6, color='#00d2ff')
                    ))
                
                if b_output_type in [t["output_dpsin"], t["output_both"]]:
                    fig_b.add_trace(go.Scatter(
                        x=b_param_range, y=dp_sin_vals,
                        mode='lines+markers',
                        name=t["output_dpsin"],
                        line=dict(color='#ff6b6b', width=3, dash='dash'),
                        marker=dict(size=6, color='#ff6b6b')
                    ))
                
                if b_output_type in [t["output_dptotal"], t["output_both"]]:
                    fig_b.add_trace(go.Scatter(x=b_param_range, y=dp_total_vals,
                        mode='lines+markers',
                        name=t["output_dptotal"],
                        line=dict(color='#7c3aed', width=3, dash='dot'),
                        marker=dict(size=6, color='#7c3aed')
                    ))
                
                if b_output_type in [t["output_qtotal"], t["output_both"]]:
                    fig_b.add_trace(go.Scatter(
                        x=b_param_range, y=Q_total_vals,
                        mode='lines+markers',
                        name=t["output_qtotal"],
                        line=dict(color='#f9a825', width=3, dash='dashdot'),
                        marker=dict(size=6, color='#f9a825'),
                        yaxis='y2'
                    ))
                
                fig_b.add_vline(x=default_val, line_dash='dash', line_color='orange',
                                annotation_text=f"{t['sens_base']} = {default_val:.2f}",
                                annotation_position='top')
                
                fig_b.update_layout(
                    title=f"<b>{t['bernoulli_effect'].format(param=selected_bparam)}</b>",
                    xaxis_title=selected_bparam,
                    yaxis_title=t["param12"],
                    yaxis2=dict(title=t["output_qtotal"], overlaying='y', side='right'),
                    template='plotly_white',
                    hovermode='x unified',
                    height=450,
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', bordercolor='#ddd', borderwidth=1)
                )
                
                st.plotly_chart(fig_b, use_container_width=True)
    
    # ======================== تب ۲: Heatmap برنولی ========================
with btab2:
    st.markdown(f"""
    <div style="
        background: #f0f2f6;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 15px;
    ">
        <p style="margin: 0; font-size: 13px; color: #555;">
        🎯 <b>{t['bernoulli_desc']}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        bparam1 = st.selectbox(t["param1"], list(bernoulli_params.keys()), index=0, key="bhm_p1")
        bp1_key = bernoulli_params[bparam1]
        min1, max1, _, _ = bernoulli_ranges[bp1_key]
        br1_min = st.number_input(f"{t['param_min']} {bparam1}:", min1, max1, min1, format="%.2f", key="bhm1min")
        br1_max = st.number_input(f"{t['param_max']} {bparam1}:", min1, max1, max1, format="%.2f", key="bhm1max")
        bn1 = st.slider(t["n_points"], 10, 30, 15, 5, key="bhm_n1")
    
    with col2:
        bparam2 = st.selectbox(t["param2"], list(bernoulli_params.keys()), index=1, key="bhm_p2")
        bp2_key = bernoulli_params[bparam2]
        min2, max2, _, _ = bernoulli_ranges[bp2_key]
        br2_min = st.number_input(f"{t['param_min']} {bparam2}:", min2, max2, min2, format="%.2f", key="bhm2min")
        br2_max = st.number_input(f"{t['param_max']} {bparam2}:", min2, max2, max2, format="%.2f", key="bhm2max")
        bn2 = st.slider(t["n_points"], 10, 30, 15, 5, key="bhm_n2")
    
    bhm_output = st.selectbox(t["heatmap_output"], [t["output_alpha"], t["output_dpsin"], t["output_dptotal"]], key="bhm_out")
    run_bhm = st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_bhm")
    
    if run_bhm:
        with st.spinner("⏳ Heatmap "):
            x_vals = np.linspace(br1_min, br1_max, bn1)
            y_vals = np.linspace(br2_min, br2_max, bn2)
            Z = np.zeros((bn2, bn1))
            
            progress = st.progress(0)
            for i, v1 in enumerate(x_vals):
                for j, v2 in enumerate(y_vals):
                    # ====== مقداردهی اولیه با پارامترهای فعلی ======
                    temp_Qp = Q_portal
                    temp_Qa = Q_artery
                    temp_Ap = A_portal
                    temp_Ah = A_hepatic
                    temp_h = h
                    temp_r0 = r0
                    temp_L = L
                    temp_beta = beta
                    temp_mu = mu_inf
                    temp_tau = tau_y
                    
                    # ====== جایگزینی پارامتر اول ======
                    if bp1_key == "Q_portal":
                        temp_Qp = v1 / 1000 / 60
                    elif bp1_key == "Q_artery":
                        temp_Qa = v1 / 1000 / 60
                    elif bp1_key == "A_portal":
                        temp_Ap = v1 * 1e-4
                    elif bp1_key == "A_hepatic":
                        temp_Ah = v1 * 1e-4
                    elif bp1_key == "h":
                        temp_h = v1 / 100
                    elif bp1_key == "r0":
                        temp_r0 = v1 * 1e-6
                    elif bp1_key == "L":
                        temp_L = v1 * 1e-6
                    elif bp1_key == "beta":
                        temp_beta = v1
                    elif bp1_key == "mu_inf":
                        temp_mu = v1
                    elif bp1_key == "tau_y":
                        temp_tau = v1
                    
                    # ====== جایگزینی پارامتر دوم ======
                    if bp2_key == "Q_portal":
                        temp_Qp = v2 / 1000 / 60
                    elif bp2_key == "Q_artery":
                        temp_Qa = v2 / 1000 / 60
                    elif bp2_key == "A_portal":
                        temp_Ap = v2 * 1e-4
                    elif bp2_key == "A_hepatic":
                        temp_Ah = v2 * 1e-4
                    elif bp2_key == "h":
                        temp_h = v2 / 100
                    elif bp2_key == "r0":
                        temp_r0 = v2 * 1e-6
                    elif bp2_key == "L":
                        temp_L = v2 * 1e-6
                    elif bp2_key == "beta":
                        temp_beta = v2
                    elif bp2_key == "mu_inf":
                        temp_mu = v2
                    elif bp2_key == "tau_y":
                        temp_tau = v2
                    
                    # ====== محاسبه با پارامترهای جدید ======
                    temp_alpha, _, _, _, temp_dpsin, _, _, temp_dptotal, _ = calc_alpha(
                        temp_Qp, temp_Qa, temp_Ap, temp_Ah, temp_h, temp_r0, temp_beta, temp_L, temp_mu, temp_tau
                    )
                    
                    if bhm_output == t["output_alpha"]:
                        Z[j, i] = temp_alpha
                    elif bhm_output == t["output_dpsin"]:
                        Z[j, i] = temp_dpsin / mmHg_to_Pa
                    else:
                        Z[j, i] = temp_dptotal / mmHg_to_Pa
                
                progress.progress((i + 1) / bn1)
            
            progress.empty()
        
        # ====== نمایش Heatmap ======
        fig_bhm = go.Figure(data=go.Heatmap(
            z=Z,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            hovertemplate=f'{bparam1}: %{{x:.2f}}<br>{bparam2}: %{{y:.2f}}<br>{bhm_output}: %{{z:.3f}}<extra></extra>'
        ))
        
        fig_bhm.update_layout(
            title=f"<b>{t['bernoulli_heatmap'].format(p1=bparam1, p2=bparam2)}</b>",
            xaxis_title=bparam1,
            yaxis_title=bparam2,
            template='plotly_white',
            height=550,
            coloraxis_colorbar=dict(title=bhm_output)
        )
        # ====== نمودار سه‌بعدی (3D Surface) برای تحلیل دو پارامتری ======
        fig_3d = go.Figure(data=[
        go.Surface(
            z=Z,
            x=x_vals,
            y=y_vals,
            colorscale='Viridis',
            hovertemplate=f'{param1}: %{{x:.2f}}<br>{param2}: %{{y:.2f}}<br>{output_hm}: %{{z:.2f}}<extra></extra>'
        )
    ])

    fig_3d.update_layout(
        title=f"<b>3D: {param1} & {param2} on {output_hm}</b><br><sup>{t['fixed_deltaP']} = {fixed_dp_hm} mmHg</sup>",
        scene=dict(
           xaxis_title=param1,
           yaxis_title=param2,
           zaxis_title=output_hm,
           camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        template=get_plotly_template(),
        height=600
    )

st.plotly_chart(fig_3d, use_container_width=True)
        
        st.plotly_chart(fig_bhm, use_container_width=True)
        
        min_idx = np.unravel_index(np.argmin(Z), Z.shape)
        max_idx = np.unravel_index(np.argmax(Z), Z.shape)
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"🟢 {t['heatmap_min']} {bhm_output}: {np.min(Z):.3f} در ({x_vals[min_idx[1]]:.2f}, {y_vals[min_idx[0]]:.2f})")
        with c2:
            st.warning(f"🔴 {t['heatmap_max']} {bhm_output}: {np.max(Z):.3f} در ({x_vals[max_idx[1]]:.2f}, {y_vals[max_idx[0]]:.2f})")
    # ======================== تب ۳: گزارش برنولی ========================
    with btab3:
        st.markdown(f"""
        <div style="
            background: #f0f2f6;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="margin: 0; font-size: 13px; color: #555;">
            📊 <b>{t['bernoulli_report']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(t["report_generate"], use_container_width=True, type="primary", key="gen_breport"):
            with st.spinner("⏳..."):
                report_data = []
                
                for key, name in bernoulli_params.items():
                    min_v, max_v, default_v, _ = bernoulli_ranges[name]
                    
                    # مقدار پایه
                    temp_alpha_base, temp_Qtotal_base, _, _, temp_dpsin_base, _, _, temp_dptotal_base, _ = calc_alpha(
                        Q_portal, Q_artery, A_portal, A_hepatic, h, r0, beta, L, mu_inf, tau_y
                    )
                    
                    # مقدار min
                    temp_Qp = Q_portal if name != "Q_portal" else min_v/1000/60
                    temp_Qa = Q_artery if name != "Q_artery" else min_v/1000/60
                    temp_Ap = A_portal if name != "A_portal" else min_v*1e-4
                    temp_Ah = A_hepatic if name != "A_hepatic" else min_v*1e-4
                    temp_h = h if name != "h" else min_v/100
                    temp_r0 = r0 if name != "r0" else min_v*1e-6
                    temp_L = L if name != "L" else min_v*1e-6
                    temp_beta = beta if name != "beta" else min_v
                    temp_mu = mu_inf if name != "mu_inf" else min_v
                    temp_tau = tau_y if name != "tau_y" else min_v
                    
                    temp_alpha_min, _, _, _, _, _, _, _, _ = calc_alpha(
                        temp_Qp, temp_Qa, temp_Ap, temp_Ah, temp_h, temp_r0, temp_beta, temp_L, temp_mu, temp_tau
                    )
                    
                    # مقدار max
                    temp_Qp = Q_portal if name != "Q_portal" else max_v/1000/60
                    temp_Qa = Q_artery if name != "Q_artery" else max_v/1000/60
                    temp_Ap = A_portal if name != "A_portal" else max_v*1e-4
                    temp_Ah = A_hepatic if name != "A_hepatic" else max_v*1e-4
                    temp_h = h if name != "h" else max_v/100
                    temp_r0 = r0 if name != "r0" else max_v*1e-6
                    temp_L = L if name != "L" else max_v*1e-6
                    temp_beta = beta if name != "beta" else max_v
                    temp_mu = mu_inf if name != "mu_inf" else max_v
                    temp_tau = tau_y if name != "tau_y" else max_v
                    
                    temp_alpha_max, _, _, _, _, _, _, _, _ = calc_alpha(
                        temp_Qp, temp_Qa, temp_Ap, temp_Ah, temp_h, temp_r0, temp_beta, temp_L, temp_mu, temp_tau
                    )
                    
                    sensitivity = (temp_alpha_max - temp_alpha_min) / (temp_alpha_base + 1e-10)
                    
                    report_data.append({
                        t['report_param']: key,
                        t['report_base']: default_v,
                        'α_min': temp_alpha_min,
                        'α_max': temp_alpha_max,
                        t['report_sensitivity']: sensitivity,
                        t['report_status']: f'🟢 {t["status_low"]}' if abs(sensitivity) < 0.1 else f'🟡 {t["status_medium"]}' if abs(sensitivity) < 0.3 else f'🔴 {t["status_high"]}'
                    })
                
                df_breport = pd.DataFrame(report_data)
                
                st.subheader(t["sens_anz"])
                st.dataframe(df_breport, use_container_width=True, hide_index=True)
                
                fig_brep = go.Figure()
                
                fig_brep.add_trace(go.Bar(
                    x=df_breport[t['report_param']],
                    y=df_breport[t['report_sensitivity']],
                    marker_color=['#00d2ff' if abs(x) < 0.1 else '#ff6b6b' if abs(x) < 0.3 else '#7c3aed' for x in df_breport[t['report_sensitivity']]],
                    text=df_breport[t['report_sensitivity']].round(3),
                    textposition='outside',
                    name=t['report_sensitivity']
                ))
                
                fig_brep.update_layout(
                    title=t["sens_anz2"],
                    xaxis_title=t['report_param'],
                    yaxis_title=t['report_sensitivity'],
                    template='plotly_white',
                    height=450
                )
                
                st.plotly_chart(fig_brep, use_container_width=True)
                
                csv_breport = df_breport.to_csv(index=False)
                st.download_button(
                    label=t["download_csv"],
                    data=csv_breport,
                    file_name="bernoulli_sensitivity_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )







#=============================================================================================================================================    

# ======================== تحلیل حساسیت پیشرفته ========================
with st.expander(t["sensitivity_title"], expanded=False):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 5px solid #00d2ff;
        color: white;
    ">
        <h4 style="margin: 0; color: #00d2ff;">🔬 {t['sensitivity_title']}</h4>
        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">
            {t['sensitivity_subtitle']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== تب‌های مختلف ==========
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t["sensitivity_1d"], 
        t["sensitivity_2d"], 
        t["sensitivity_tornado"],
        t["sensitivity_monte"],
        t["sensitivity_report"]
    ])
    
    # ======================== تب ۱: تحلیل یک‌بعدی ========================
    with tab1:
        st.markdown(f"""
        <div style="
            background: #f0f2f6;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="margin: 0; font-size: 13px; color: #555;">
            📌 <b>{t['sensitivity_1d_desc']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            param_options = {
                t["param_kf0"]: "Kf0",
                t["param_sigma"]: "sigma",
                t["param_pi0"]: "Pi0",
                t["param_jmax"]: "Jmax",
                t["param_dpi"]: "dPi",
                t["param_km"]: "Km"
            }
            
            selected_param = st.selectbox(t["select_param"], list(param_options.keys()), key="sens_param1")
            param_key = param_options[selected_param]
            
            ranges = {
                "Kf0": (1.0, 5.0, 3.0, 0.5),
                "sigma": (0.1, 0.4, 0.22, 0.02),
                "Pi0": (0.1, 2.0, 0.5, 0.1),
                "Jmax": (30, 50, 40, 5),
                "dPi": (18, 26, 22, 1),
                "Km": (0.1, 1.5, 0.74, 0.1)
            }
            
            min_val, max_val, default_val, step = ranges[param_key]
            
            range_min = st.number_input(t["param_min"], min_val, max_val, min_val, step, format="%.3f", key="rmin")
            range_max = st.number_input(t["param_max"], min_val, max_val, max_val, step, format="%.3f", key="rmax")
            n_points = st.slider(t["n_points"], 10, 100, 50, 5, key="npts")
            
            col_fixed1, col_fixed2 = st.columns(2)
            with col_fixed1:
                fixed_deltaP = st.slider(t["fixed_deltaP"], 0.0, 25.0, 12.0, 0.5, key="fixed_dp1")
            with col_fixed2:
                output_type = st.selectbox(t["output_type"], [t["output_jv"], t["output_jnet"], t["output_both"]], key="out_type")
            
            run_1d = st.button(t["run_analysis"], use_container_width=True, type="primary", key="run1d")

        with col2:
            if run_1d:
                with st.spinner("⏳ ..."):
                    param_range = np.linspace(range_min, range_max, n_points)
                    
                    Jv_vals, Jnet_vals, Kf_vals, Pi_vals = [], [], [], []
                    
                    for val in param_range:
                        temp_params = params.copy()
                        temp_params[param_key] = val
                        
                        Kf = calc_Kf_nonlinear(temp_params['Kf0'], fixed_deltaP)
                        Pi = calc_Pi_nonlinear(temp_params['Pi0'], fixed_deltaP)
                        Jv = calc_Jv(fixed_deltaP, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'],P_hep)
                        Jlymph = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi)
                        Jnet = calc_Jnet(Jv, Jlymph)
                        
                        Jv_vals.append(Jv); Jnet_vals.append(Jnet)
                        Kf_vals.append(Kf); Pi_vals.append(Pi)
                
                fig = go.Figure()
                
                if output_type in [t["output_jv"], t["output_both"]]:
                    fig.add_trace(go.Scatter(
                        x=param_range, y=Jv_vals,
                        mode='lines+markers',
                        name=t["output_jv"],
                        line=dict(color='#00d2ff', width=3),
                        marker=dict(size=6, color='#00d2ff')
                    ))
                
                if output_type in [t["output_jnet"], t["output_both"]]:
                    fig.add_trace(go.Scatter(
                        x=param_range, y=Jnet_vals,
                        mode='lines+markers',
                        name=t["output_jnet"],
                        line=dict(color='#ff6b6b', width=3, dash='dash'),
                        marker=dict(size=6, color='#ff6b6b')
                    ))
                
                fig.add_trace(go.Scatter(
                    x=param_range, y=Kf_vals,
                    mode='lines',
                    name='Kf',
                    line=dict(color='#7c3aed', width=2, dash='dot'),
                    yaxis='y2'
                ))
                
                fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)
                fig.add_vline(x=default_val, line_dash='dash', line_color='orange',
                              annotation_text=f"{t['sens_base']} = {default_val:.2f}",
                              annotation_position='top')
                
                fig.update_layout(
                    title=f"<b>{t['sens_effect'].format(param=selected_param)}</b><br><sup>{t['fixed_deltaP']} = {fixed_deltaP} mmHg</sup>",
                    xaxis_title=selected_param,
                    yaxis_title=t["yaxis_flow"],
                    yaxis2=dict(title="Kf (ml/min/mmHg)", overlaying='y', side='right'),
                    template='plotly_white',
                    hovermode='x unified',
                    height=450,
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.9)', bordercolor='#ddd', borderwidth=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                sens_jv = np.std(Jv_vals) / (np.mean(Jv_vals) + 1e-10)
                sens_jnet = np.std(Jnet_vals) / (np.mean(Jnet_vals) + 1e-10)
                
                c1, c2, c3 = st.columns(3)
                c1.metric(t["sens_jv"], f"{sens_jv:.3f}", 
                            delta=t["sens_high"] if sens_jv > 0.2 else t["sens_medium"] if sens_jv > 0.1 else t["sens_low"])
                c2.metric(t["sens_jnet"], f"{sens_jnet:.3f}",
                            delta=t["sens_high"] if sens_jnet > 0.2 else t["sens_medium"] if sens_jnet > 0.1 else t["sens_low"])
                c3.metric(t["sens_range"], f"{min(Jv_vals):.2f} - {max(Jv_vals):.2f}")
                
    
    # ======================== تب ۲: Heatmap ========================
    with tab2:
        st.markdown(f"""
        <div style="
            background: #f0f2f6;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="margin: 0; font-size: 13px; color: #555;">
            🎯 <b>{t['sensitivity_2d_desc']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            param1 = st.selectbox(t["param1"], list(param_options.keys()), index=0, key="hm_p1")
            p1_key = param_options[param1]
            min1, max1, _, _ = ranges[p1_key]
            r1_min = st.number_input(f"{t['param_min']} {param1}:", min1, max1, min1, format="%.2f", key="hm1min")
            r1_max = st.number_input(f"{t['param_max']} {param1}:", min1, max1, max1, format="%.2f", key="hm1max")
            n1 = st.slider(t["n_points"], 10, 50, 30, 5, key="hm_n1")
        
        with col2:
            param2 = st.selectbox(t["param2"], list(param_options.keys()), index=1, key="hm_p2")
            p2_key = param_options[param2]
            min2, max2, _, _ = ranges[p2_key]
            r2_min = st.number_input(f"{t['param_min']} {param2}:", min2, max2, min2, format="%.2f", key="hm2min")
            r2_max = st.number_input(f"{t['param_max']} {param2}:", min2, max2, max2, format="%.2f", key="hm2max")
            n2 = st.slider(t["n_points"], 10, 50, 30, 5, key="hm_n2")
        
        fixed_dp_hm = st.slider(t["fixed_deltaP"], 0.0, 25.0, 12.0, 0.5, key="hm_dp")
        output_hm = st.selectbox(t["heatmap_output"], [t["output_jnet"], t["output_jv"]], key="hm_out")
        run_hm = st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_hm")
        
        if run_hm:
            with st.spinner("⏳ Heatmap..."):
                x_vals = np.linspace(r1_min, r1_max, n1)
                y_vals = np.linspace(r2_min, r2_max, n2)
                Z = np.zeros((n2, n1))
                
                progress = st.progress(0)
                for i, v1 in enumerate(x_vals):
                    for j, v2 in enumerate(y_vals):
                        temp_params = params.copy()
                        temp_params[p1_key] = v1
                        temp_params[p2_key] = v2
                        
                        Kf = calc_Kf_nonlinear(temp_params['Kf0'], fixed_dp_hm)
                        Pi = calc_Pi_nonlinear(temp_params['Pi0'], fixed_dp_hm)
                        Jv = calc_Jv(fixed_dp_hm, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'],P_hep)
                        Jlymph = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi)
                        Jnet = calc_Jnet(Jv, Jlymph)
                        
                        Z[j, i] = Jnet if output_hm == t["output_jnet"] else Jv
                    
                    progress.progress((i + 1) / n1)
                
                progress.empty()
            
            fig_hm = go.Figure(data=go.Heatmap(
                z=Z,
                x=x_vals,
                y=y_vals,
                colorscale='RdYlGn',
                zmid=0,
                hovertemplate=f'{param1}: %{{x:.2f}}<br>{param2}: %{{y:.2f}}<br>{output_hm}: %{{z:.2f}}<extra></extra>'
            ))
            
            fig_hm.update_layout(
                title=f"<b>{t['sens_heatmap_title'].format(p1=param1, p2=param2)}</b><br><sup>{t['fixed_deltaP']} = {fixed_dp_hm} mmHg</sup>",
                xaxis_title=param1,
                yaxis_title=param2,
                template='plotly_white',
                height=550,
                coloraxis_colorbar=dict(title=output_hm)
            )
            
            st.plotly_chart(fig_hm, use_container_width=True)
            
            min_idx = np.unravel_index(np.argmin(Z), Z.shape)
            max_idx = np.unravel_index(np.argmax(Z), Z.shape)
            
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"🟢 {t['heatmap_min']} {output_hm}: {np.min(Z):.2f} in ({x_vals[min_idx[1]]:.2f}, {y_vals[min_idx[0]]:.2f})")
            with c2:
                st.warning(f"🔴 {t['heatmap_max']} {output_hm}: {np.max(Z):.2f} in ({x_vals[max_idx[1]]:.2f}, {y_vals[max_idx[0]]:.2f})")
    
    # ======================== تب ۳: Tornado Diagram ========================
    with tab3:
        st.markdown(f"""
        <div style="
            background: #f0f2f6;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="margin: 0; font-size: 13px; color: #555;">
            🌪️ <b>{t['sensitivity_tornado_desc']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            dp_tornado = st.slider(t["fixed_deltaP"], 0.0, 25.0, 12.0, 0.5, key="tor_dp")
        with col2:
            output_tornado = st.selectbox(t["tornado_output"], ["Jnet", "Jv"], key="tor_out")
        
        if st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_tor"):
            with st.spinner("⏳Tornado..."):
                results = []
                
                for key, name in param_options.items():
                    min_v, max_v, default_v, _ = ranges[name]
                    
                    temp_params = params.copy()
                    
                    Kf = calc_Kf_nonlinear(temp_params['Kf0'], dp_tornado)
                    Pi = calc_Pi_nonlinear(temp_params['Pi0'], dp_tornado)
                    Jv = calc_Jv(dp_tornado, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'],P_hep)
                    Jlymph = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi)
                    Jnet=Jv-Jlymph
                    base = Jnet if output_tornado == "Jnet" else Jv
                    
                    temp_params[name] = min_v
                    Kf = calc_Kf_nonlinear(temp_params['Kf0'], dp_tornado)
                    Pi = calc_Pi_nonlinear(temp_params['Pi0'], dp_tornado)
                    Jv = calc_Jv(dp_tornado, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'],P_hep)
                    Jlymph = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi)
                    Jnet=Jv-Jlymph
                    val_min = Jnet if output_tornado == "Jnet" else Jv
                    
                    temp_params[name] = max_v
                    Kf = calc_Kf_nonlinear(temp_params['Kf0'], dp_tornado)
                    Pi = calc_Pi_nonlinear(temp_params['Pi0'], dp_tornado)
                    Jv = calc_Jv(dp_tornado, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'],P_hep)
                    Jlymph = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi)
                    Jnet=Jv-Jlymph
                    val_max = Jnet if output_tornado == "Jnet" else Jv
                    
                    results.append({
                        'parameter': key,
                        'base': base,
                        'min': val_min - base,
                        'max': val_max - base,
                        'range': abs(val_max - val_min)
                    })
                
                results = sorted(results, key=lambda x: x['range'], reverse=True)
                
                fig_tor = go.Figure()
                
                names = [r['parameter'] for r in results]
                min_vals = [r['min'] for r in results]
                max_vals = [r['max'] for r in results]
                
                fig_tor.add_trace(go.Bar(
                    y=names,
                    x=min_vals,
                    name=t["kahesh"],
                    orientation='h',
                    marker_color='#ff6b6b',
                    text=[f"{v:.2f}" for v in min_vals],
                    textposition='outside'
                ))
                
                fig_tor.add_trace(go.Bar(
                    y=names,
                    x=max_vals,
                    name=t["afz"],
                    orientation='h',
                    marker_color='#00d2ff',
                    text=[f"{v:.2f}" for v in max_vals],
                    textposition='outside'
                ))
                
                fig_tor.update_layout(
                    title=f"<b>{t['sens_tornado_title'].format(output=output_tornado)}</b><br><sup>{t['fixed_deltaP']} = {dp_tornado} mmHg | {t['sens_base']} = {results[0]['base']:.2f}</sup>",
                    xaxis_title=f"{t['sens_range']} {output_tornado}",
                    yaxis_title="",
                    template='plotly_white',
                    height=500,
                    barmode='relative',
                    bargap=0.3,
                    legend=dict(x=0.02, y=0.98)
                )
                
                st.plotly_chart(fig_tor, use_container_width=True)
                
                st.subheader(t["nnn"])
                df_tor = pd.DataFrame(results)
                df_tor['range'] = df_tor['range'].round(3)
                df_tor['min'] = df_tor['min'].round(2)
                df_tor['max'] = df_tor['max'].round(2)
                st.dataframe(df_tor[['parameter', 'min', 'max', 'range']], use_container_width=True, hide_index=True)
    
    # ======================== تب ۴: Monte Carlo ========================
    with tab4:
        st.markdown(f"""
        <div style="
            background: #f0f2f6;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="margin: 0; font-size: 13px; color: #555;">
            🎲 <b>{t['sensitivity_monte_desc']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            n_simulations = st.slider(t["mc_simulations"], 100, 100000, 1000, 100, key="mc_n")
            mc_dp = st.slider(t["fixed_deltaP"], 0.0, 25.0, 12.0, 0.5, key="mc_dp")
        with col2:
            uncertainty_level = st.select_slider(t["mc_uncertainty"], options=[t["mc_low"], t["mc_medium"], t["mc_high"]], key="mc_unc")
            unc_factor = {t["mc_low"]: 0.05, t["mc_medium"]: 0.15, t["mc_high"]: 0.30}[uncertainty_level]
        
        if st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_mc"):
            with st.spinner("⏳ Loading Monte Carlo..."):
                np.random.seed(42)
                
                Kf_samples = np.random.normal(params['Kf0'], params['Kf0'] * unc_factor, n_simulations)
                sigma_samples = np.random.normal(params['sigma'], params['sigma'] * unc_factor, n_simulations)
                Pi_samples = np.random.normal(params['Pi0'], params['Pi0'] * unc_factor, n_simulations)
                dPi_samples = np.random.normal(params['dPi'], params['dPi'] * unc_factor, n_simulations)
                
                Kf_samples = np.clip(Kf_samples, 0.5, 10)
                sigma_samples = np.clip(sigma_samples, 0.05, 0.5)
                Pi_samples = np.clip(Pi_samples, 0.05, 3)
                dPi_samples = np.clip(dPi_samples, 15, 28)
                
                Jv_samples = []
                Jnet_samples = []
                
                progress = st.progress(0)
                for i in range(n_simulations):
                    Kf = calc_Kf_nonlinear(Kf_samples[i], mc_dp)
                    Pi = calc_Pi_nonlinear(Pi_samples[i], mc_dp)
                    Jv = calc_Jv(mc_dp, Kf, params['alpha'], sigma_samples[i], Pi, dPi_samples[i],P_hep)
                    Jlymph = calc_Jlymph(params['Jmax'], params['Km'], Pi)
                    Jnet = calc_Jnet(Jv, Jlymph)
                    
                    Jv_samples.append(Jv)
                    Jnet_samples.append(Jnet)
                    
                    if i % max(1, n_simulations // 20) == 0:
                        progress.progress((i + 1) / n_simulations)
                
                progress.empty()
            
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{t['output_jv']} {t['mc_mean']}", f"{np.mean(Jv_samples):.2f}", delta=f"±{np.std(Jv_samples):.2f}")
            c2.metric(f"{t['output_jv']} {t['mc_ci']}", f"[{np.percentile(Jv_samples, 2.5):.2f}, {np.percentile(Jv_samples, 97.5):.2f}]")
            c3.metric(t["mc_risk"], f"{np.mean(np.array(Jnet_samples) > 0) * 100:.1f}%")
  
            
            fig_mc = go.Figure()
            
            fig_mc.add_trace(go.Histogram(
                x=Jnet_samples,
                nbinsx=50,
                name=t["output_jnet"],
                marker_color='#7c3aed',
                opacity=0.7
            ))
            
            fig_mc.add_vline(x=0, line_dash='dash', line_color='red',
                            annotation_text=t["mc_ci_label"], annotation_position='top')
            
            fig_mc.update_layout(
                title=f"<b>{t['sens_mc_title'].format(n=n_simulations)}</b><br><sup>{t['fixed_deltaP']} = {mc_dp} mmHg | {t['mc_uncertainty']}: {uncertainty_level}</sup>",
                xaxis_title=f"{t['output_jnet']} (ml/min)",
                yaxis_title="تعداد",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig_mc, use_container_width=True)
            
            fig_box = go.Figure()
            
            fig_box.add_trace(go.Box(
                y=Jv_samples,
                name=t["output_jv"],
                marker_color='#00d2ff'
            ))
            
            fig_box.add_trace(go.Box(
                y=Jnet_samples,
                name=t["output_jnet"],
                marker_color='#ff6b6b'
            ))
            
            fig_box.update_layout(
                title=t["sens_box_title"],
                yaxis_title="ml/min",
                template='plotly_white',
                height=350
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
    
    # ======================== تب ۵: گزارش جامع ========================
    with tab5:
        st.markdown(f"""
        <div style="
            background: #f0f2f6;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
        ">
            <p style="margin: 0; font-size: 13px; color: #555;">
            📊 <b>{t['sensitivity_report_desc']}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(t["report_generate"], use_container_width=True, type="primary", key="gen_report"):
            with st.spinner("⏳ ..."):
                report_data = []
                
                for key, name in param_options.items():
                    min_v, max_v, default_v, _ = ranges[name]
                    
                    temp_params = params.copy()
                    
                    Kf = calc_Kf_nonlinear(temp_params['Kf0'], 12)
                    Pi = calc_Pi_nonlinear(temp_params['Pi0'], 12)
                    Jv = calc_Jv(12, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'],P_hep)
                    
                    base = Jnet
                    
                    test_vals = [min_v, default_v, max_v]
                    outputs = []
                    for val in test_vals:
                        temp_params[name] = val
                        Kf = calc_Kf_nonlinear(temp_params['Kf0'], 12)
                        Pi = calc_Pi_nonlinear(temp_params['Pi0'], 12)
                        Jv = calc_Jv(12, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'],P_hep)
                        
                        outputs.append(Jv)
                    
                    sensitivity = (outputs[2] - outputs[0]) / (outputs[1] + 1e-10)
                    
                    report_data.append({
                        t['report_param']: key,
                        t['report_base']: default_v,
                        t['report_min']: outputs[0],
                        t['report_max']: outputs[2],
                        t['report_sensitivity']: sensitivity,
                        t['report_status']: f'🟢 {t["status_low"]}' if abs(sensitivity) < 0.5 else f'🟡 {t["status_medium"]}' if abs(sensitivity) < 0.8 else f'🔴 {t["status_high"]}'
                    })
                
                df_report = pd.DataFrame(report_data)
                
                st.subheader(t["sens_anz"])
                st.dataframe(df_report, use_container_width=True, hide_index=True)
                
                fig_comp = go.Figure()
                
                fig_comp.add_trace(go.Bar(
                    x=df_report[t['report_param']],
                    y=df_report[t['report_sensitivity']],
                    marker_color=['#00d2ff' if abs(x) < 0.5 else '#ff6b6b' if abs(x) < 1.5 else '#7c3aed' for x in df_report[t['report_sensitivity']]],
                    text=df_report[t['report_sensitivity']].round(2),
                    textposition='outside',
                    name=t['report_sensitivity']
                ))
                
                fig_comp.update_layout(
                    title=t["sens_anz2"],
                    xaxis_title=t['report_param'],
                    yaxis_title=t['report_sensitivity'],
                    template='plotly_white',
                    height=450
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                csv_report = df_report.to_csv(index=False)
                st.download_button(
                    label=t["download_csv"],
                    data=csv_report,
                    file_name="sensitivity_report.csv",
                    mime="text/csv",
                    use_container_width=True
                )








    
    
    
#===========================================================================================================


# ============================================================



# ============================================================
# 2️
# ============================================================
with st.sidebar:
    st.divider()
    if st.button(t["clear_cache"], use_container_width=True):
        st.cache_data.clear()
        st.success(t["cache_cleared"])
        
# ============================================================


# ============================================================
# 
# ============================================================
with st.expander(t["upload_csv"], expanded=False):
    st.info(t["upload_help"])
    uploaded_file = st.file_uploader(t["upload_csv"], type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_patients = pd.read_csv(uploaded_file)
            st.dataframe(df_patients.head(5))
            
            if st.button(t["upload_run"], use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                
                for idx, row in df_patients.iterrows():
                    dp_patient = row.get('ΔP', 12.0)
                    kf_patient = row.get('Kf0', Kf0)
                    sigma_patient = row.get('sigma', sigma)
                    pi_patient = row.get('Pi0', Pi0)
                    jmax_patient = row.get('Jmax', Jmax)
                    km_patient = row.get('Km', Km)
                    dpi_patient = row.get('dPi', dPi)
                    
                    Kf_p = calc_Kf_nonlinear(kf_patient, dp_patient)
                    Pi_p = calc_Pi_nonlinear(pi_patient, dp_patient)
                    Jv_p = calc_Jv(dp_patient, Kf_p, alpha, sigma_patient, Pi_p, dpi_patient, P_hep)
                    Jlymph_p = calc_Jlymph(jmax_patient, km_patient, Pi_p)
                    Jnet_p = calc_Jnet(Jv_p, Jlymph_p)
                    
                    results.append({
                        'ΔP': round(dp_patient, 2),
                        'Jv': round(Jv_p, 2),
                        'Jnet': round(Jnet_p, 2),
                        'Status': t["upload_status"] if Jnet_p > 0 else t["upload_compensated"]
                    })
                    
                    progress_bar.progress((idx + 1) / len(df_patients))
                
                progress_bar.empty()
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                csv_results = df_results.to_csv(index=False)
                st.download_button(
                    label=t["upload_download"],
                    data=csv_results,
                    file_name="patient_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(t["upload_error"].format(e=e))

# ============================================================
# ============================================================

# ==========================================================
# ============================================================
try:
    import plotly.express as px
    
    with st.expander(t["3d_title"], expanded=False):
        st.info(t["3d_info"])
        
        col1, col2 = st.columns(2)
        with col1:
            param_3d_1 = st.selectbox(
                t["3d_param1"], 
                ["Kf₀ (Filtration)", "ΔP (Pressure)", "σ (Reflection)"], 
                key="3d_p1"
            )
        with col2:
            param_3d_2 = st.selectbox(
                t["3d_param2"], 
                ["Kf₀ (Filtration)", "ΔP (Pressure)", "σ (Reflection)"], 
                key="3d_p2", 
                index=1
            )
        
        output_3d = st.selectbox(
            "Output:", 
            ["Jv (Filtration)", "Jnet (Net)"], 
            key="3d_out"
        )
        
        if st.button(t["3d_plot"], key="btn_3d"):
            with st.spinner("⏳ Calculating..."):
                
                # محدوده‌های پارامترها
                param_ranges = {
                    "Kf₀ (Filtration)": (1.0, 8.0, 3.0),
                    "ΔP (Pressure)": (2.0, 20.0, 12.0),
                    "σ (Reflection)": (0.1, 0.4, 0.22)
                }
                
                p1_min, p1_max, _ = param_ranges[param_3d_1]
                p2_min, p2_max, _ = param_ranges[param_3d_2]
                
                n_points = 25
                x_vals = np.linspace(p1_min, p1_max, n_points)
                y_vals = np.linspace(p2_min, p2_max, n_points)
                X, Y = np.meshgrid(x_vals, y_vals)
                Z = np.zeros_like(X)
                
                progress_bar = st.progress(0)
                total_points = n_points * n_points
                counter = 0
                
                for i in range(n_points):
                    for j in range(n_points):
                        p1_val = X[i, j]
                        p2_val = Y[i, j]
                        
                        # مقداردهی اولیه با پارامترهای فعلی
                        temp_Kf0 = Kf0
                        temp_dp = deltaP_analysis
                        temp_sigma = sigma
                        
                        # تنظیم پارامتر اول
                        if param_3d_1 == "Kf₀ (Filtration)":
                            temp_Kf0 = p1_val
                        elif param_3d_1 == "ΔP (Pressure)":
                            temp_dp = p1_val
                        elif param_3d_1 == "σ (Reflection)":
                            temp_sigma = p1_val
                        
                        # تنظیم پارامتر دوم
                        if param_3d_2 == "Kf₀ (Filtration)":
                            temp_Kf0 = p2_val
                        elif param_3d_2 == "ΔP (Pressure)":
                            temp_dp = p2_val
                        elif param_3d_2 == "σ (Reflection)":
                            temp_sigma = p2_val
                        
                        # محاسبات
                        Kf_temp = calc_Kf_nonlinear(temp_Kf0, temp_dp)
                        Pi_temp = calc_Pi_nonlinear(Pi0, temp_dp)
                        Jv_temp = calc_Jv(temp_dp, Kf_temp, alpha, temp_sigma, Pi_temp, dPi, P_hep)
                        Jlymph_temp = calc_Jlymph(Jmax, Km, Pi_temp)
                        Jnet_temp = calc_Jnet(Jv_temp, Jlymph_temp)
                        
                        if output_3d == "Jv (Filtration)":
                            Z[i, j] = Jv_temp
                        else:
                            Z[i, j] = Jnet_temp
                        
                        counter += 1
                        if counter % 50 == 0:
                            progress_bar.progress(counter / total_points)
                
                progress_bar.empty()
                
                # رسم نمودار
                fig_3d = go.Figure(data=[
                    go.Surface(
                        z=Z, 
                        x=x_vals, 
                        y=y_vals, 
                        colorscale='Viridis',
                        hovertemplate=f'{param_3d_1}: %{{x:.2f}}<br>{param_3d_2}: %{{y:.2f}}<br>{output_3d}: %{{z:.2f}}<extra></extra>'
                    )
                ])
                
                fig_3d.update_layout(
                    title=f"<b>Effect of {param_3d_1} & {param_3d_2} on {output_3d}</b>",
                    scene=dict(
                        xaxis_title=param_3d_1,
                        yaxis_title=param_3d_2,
                        zaxis_title=output_3d,
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    height=600,
                    
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # نمایش مقادیر حداقلی و حداکثری
                col1, col2 = st.columns(2)
                with col1:
                    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
                    st.info(f"🟢 **Min {output_3d}**: {np.min(Z):.2f} at ({param_3d_1}={x_vals[min_idx[1]]:.2f}, {param_3d_2}={y_vals[min_idx[0]]:.2f})")
                with col2:
                    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
                    st.warning(f"🔴 **Max {output_3d}**: {np.max(Z):.2f} at ({param_3d_1}={x_vals[max_idx[1]]:.2f}, {param_3d_2}={y_vals[max_idx[0]]:.2f})")
                
                st.caption(f"📌 As {param_3d_1} and {param_3d_2} increase, {output_3d} changes nonlinearly.")
                
except ImportError:
    pass
#===============================================
# ============================================================
if 'validation_done' not in st.session_state:
    if Q_portal <= 0:
        st.warning(t["validation_warning_flow"])
    if A_portal <= 0 or A_hepatic <= 0:
        st.warning(t["validation_warning_area"])
    st.session_state.validation_done = True

# ===========================================================
# ============================================================
with st.sidebar:
    st.divider()
    if st.button(t["reset_title"], use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================================
# 8️
# ============================================================
if 'preset_values' in st.session_state:
    sc = st.session_state.preset_values
    st.info(t["preset_loaded"].format(name=preset))
    # 
# ============================================================



# ======================== فوتر ========================
st.divider()
st.caption(t["footer"])
st.caption("Ver:2.1.8")
st.caption("Ali Hosseini; email: ali.hosseini1387@icloud.com")
