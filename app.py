"""
Hepatic Hemodynamics Simulation App
Version: 5.0.0 (Dynamic Ascites Model with Saturation)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from models import mmHg_to_Pa, rho_blood, g, calc_mu_apparent, calc_shear_rate, calc_sinusoid_pressure_drop, calc_Kf_nonlinear, calc_Pi_nonlinear, calc_Jlymph, calc_Jnet, calc_alpha, calc_Jv, calc_Pi_dynamic, predict_ascites_volume_dynamic

from utils import get_clinical_interpretation


def get_plotly_template():
    if st.get_option("theme.base") == "dark":
        return "plotly_dark"
    return "plotly_white"


st.set_page_config(page_title="Hepatic Hemodynamics Simulator", page_icon="🩸", layout="wide")

# ============================================================
# TEXTS
# ============================================================
TEXTS = {
    "en": {
        "app_title": " Hepatic Filtration Simulator 🩸",
        "app_subtitle": "Based on the paper *Simulation of hepatic blood flow based on fluid mechanics principles*",
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
        "mu_inf": "u∞ (Pa·s)",
        "tau_y": "ty (Pa)",
        "r0": "r₀ (μm)",
        "L": "L (μm)",
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
        "P_hep": "P(hepatic vein)",
        "calc_details": "📊 Calculation Details",
        "total_flow": "Total Flow",
        "portal_vel": "Portal Vein Velocity",
        "hepatic_vel": "Hepatic Vein Velocity",
        "sinusoid_drop": "Sinusoid Pressure Drop",
        "height_drop": "Height Pressure Drop",
        "kinetic_drop": "Kinetic Pressure Drop",
        "total_drop": "Total Pressure Drop",
        "alpha": "α",
        "shear_rate": "Shear Rate",
        "mu_app": "u apparent",
        "dp_total": "Total ΔP",
        "auto_results": "📊 Auto-Mode Results",
        "manual_results": "📊 Manual-Mode Results",
        "deltaP": "ΔP",
        "kf_eff": "Kf",
        "pi_eff": "Pi",
        "jv": "Jv",
        "jnet": "Jnet",
        "jlymph": "Jlymph (ml/min)",
        "clinical_title": "🏥 Clinical Interpretation (ΔP = {dp:.2f} mmHg)",
        "status": "Status",
        "fluid_status": "Fluid Status",
        "ascites_pred": "Ascites Prediction",
        "key_values": "📊 Key Values at Different Points",
        "dp_mmHg": "ΔP (mmHg)",
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
        "pi_title": "Nonlinear Interstitial Pressure Pi",
        "pi_yaxis": "Pi (mmHg)",
        "key_points": "📊 Key Values",
        "clinical_interpretation": "🏥 Clinical Interpretation",
        "caption": "α = {alpha:.3f} | h = {h} cm | r₀ = {r0} μm | β = {beta} | Q_total = {q:.1f} L/min | μ = {mu:.5f} Pa·s",
        "info_text": "📌 Interpretation: At low pressures (< 12 mmHg), the lymphatic system can drain the fluid. Beyond the threshold, Kf grows exponentially and filtration exceeds lymphatic capacity.",
        "dynamic_title": "📈 Dynamic Ascites Prediction",
        "dynamic_subtitle": "🔬 Dynamic Model (Differential Equation with Saturation)",
        "dynamic_desc": "Interstitial pressure (Pi) increases with fluid volume but is limited to Pi_max (saturation). This causes gradual decrease in Jnet and eventually re-acceleration of ascites volume. Solved via Euler's method.",
        "dynamic_time": "⏱️ Simulation Time (hours)",
        "dynamic_k_elastance": "📊 Tissue Elastance Coefficient (mmHg/mL)",
        "dynamic_pi_max": "📊 Max Interstitial Pressure Pi_max (mmHg)",
        "dynamic_V0": "💧 Initial Ascites Volume (mL)",
        "dynamic_run": "🚀 Run Dynamic Simulation",
        "dynamic_success": "✅ Simulation for {time} hours completed successfully!",
        "dynamic_final_volume": "📊 Final Ascites Volume",
        "dynamic_initial_jnet": "📉 Initial Jnet",
        "dynamic_final_jnet": "📉 Final Jnet",
        "dynamic_jnet_reduction": "📉 Jnet Reduction",
        "dynamic_volume_title": "Dynamic Ascites Volume",
        "dynamic_jnet_title": "Jnet Changes Over Time",
        "dynamic_pi_title": "Interstitial Pressure (Pi) Changes Over Time",
        "dynamic_flows_title": "Jv and Jlymph Changes Over Time",
        "dynamic_time_axis": "Time (hours)",
        "dynamic_volume_axis": "Ascites Volume (mL)",
        "dynamic_jnet_axis": "Jnet (ml/min)",
        "dynamic_pi_axis": "Pi (mmHg)",
        "dynamic_flow_axis": "Flow (ml/min)",
        "dynamic_threshold": "Threshold 500 mL",
        "dynamic_equilibrium": "Equilibrium Point",
        "dynamic_pi_max_line": "Pi_max",
        "dynamic_comparison": "📊 Comparison of Static and Dynamic Models",
        "dynamic_time_col": "Time",
        "dynamic_static_col": "Static Model (Linear)",
        "dynamic_dynamic_col": "Dynamic Model (Nonlinear)",
        "dynamic_1h": "1 hour",
        "dynamic_6h": "6 hours",
        "dynamic_24h": "24 hours",
        "dynamic_interpretation": "📌 The static model assumes constant Jnet. The dynamic model considers compensatory mechanisms and saturation of Pi, showing more realistic predictions.",
        "dynamic_loading": "⏳ Solving differential equation...",
        "dynamic_phase_1": "Phase 1: Rapid accumulation",
        "dynamic_phase_2": "Phase 2: Compensation (slowdown)",
        "dynamic_phase_3": "Phase 3: Re-acceleration after Pi saturation",
        "clinical_expander": "📖 Complete Clinical Interpretation",
        "table_title": "📌 Clinical Ranges",
        "table_col1": "ΔP Range (mmHg)",
        "table_col2": "Status",
        "table_col3": "Ascites Risk",
        "table_col4": "Dominant Mechanism",
        "row1_1": "< 8", "row1_2": "Normal", "row1_3": "🟢 Very Low", "row1_4": "Starling Balance",
        "row2_1": "8 – 12", "row2_2": "Warning Zone", "row2_3": "🟡 Slight", "row2_4": "Onset of Kf Growth",
        "row3_1": "12 – 16", "row3_2": "Mild-Moderate HTN", "row3_3": "🔴 Moderate", "row3_4": "Hydraulic Breakdown",
        "row4_1": "> 16", "row4_2": "Severe HTN", "row4_3": "🔴 High", "row4_4": "Exponential Growth",
        "mechanisms_title": "🔬 Key Mechanisms",
        "mech1": "1. Hydraulic Breakdown at ΔP ≥ 12 mmHg",
        "mech2": "2. Clinical Threshold matches observations",
        "mech3": "3. Compensatory Mechanism via Pi",
        "mech4": "4. Lymphatic Saturation",
        "clinical_app": "🏥 Clinical Application",
        "clinical_app_text": "This model can be used for:",
        "clinical_app_1": "- Predicting ascites risk",
        "clinical_app_2": "- Evaluating treatments",
        "clinical_app_3": "- Designing CFD studies",
        "comparison_title": "📊 Comparison with Previous Models",
        "comparison_col1": "Feature",
        "comparison_col2": "Siggers (2013)",
        "comparison_col3": "Dongaonkar (2018)",
        "comparison_col4": "Dongaonkar (2020)",
        "comparison_col5": "**Our Model**",
        "comp_bernoulli": "Modified Bernoulli",
        "comp_viscous": "Viscous Pressure Drop",
        "comp_kf": "Kf Nonlinear",
        "comp_pi": "Pi Nonlinear",
        "comp_negative": "Negative Flux",
        "comp_casson": "Casson Model",
        "comp_lymph": "Lymphatic Drainage",
        "comp_dynamic": "Dynamic Model",
        "comp_saturation": "Pi Saturation",
        "innovation_title": "**Main Innovation:**",
        "innovation_text": "Combination of modified Bernoulli, Poiseuille with variable radius, Casson model, and Starling with nonlinear Kf and Pi, plus dynamic saturation model.",
        "sensitivity_title": "📊 Advanced Sensitivity Analysis",
        "sensitivity_subtitle": "Monte Carlo, Heatmap and Tornado Diagram",
        "sensitivity_1d": "📈 One-Dimensional",
        "sensitivity_2d": "🎯 Two-Dimensional",
        "sensitivity_tornado": "🌪️ Tornado Diagram",
        "sensitivity_monte": "🎲 Monte Carlo",
        "sensitivity_report": "📊 Comprehensive Report",
        "sensitivity_1d_desc": "Effect of one parameter on outputs.",
        "sensitivity_2d_desc": "Simultaneous effect of two parameters.",
        "sensitivity_tornado_desc": "Impact of all parameters prioritized.",
        "sensitivity_monte_desc": "Uncertainty analysis with random simulations.",
        "sensitivity_report_desc": "Complete sensitivity analysis summary.",
        "select_param": "🔍 Select Parameter:",
        "param_kf0": "Kf₀", "param_sigma": "σ", "param_pi0": "Pi₀",
        "param_jmax": "Jmax", "param_dpi": "Δπ", "param_km": "Km",
        "param_min": "min:", "param_max": "max:",
        "n_points": "Number of Points:",
        "fixed_deltaP": "ΔP (mmHg):",
        "output_type": "Output:",
        "output_jv": "Jv", "output_jnet": "Jnet", "output_both": "Both",
        "run_analysis": "🚀 Run",
        "param1": "First Parameter:", "param2": "Second Parameter:",
        "heatmap_output": "Output:",
        "heatmap_min": "Minimum", "heatmap_max": "Maximum",
        "tornado_output": "Output:",
        "mc_simulations": "Number of Simulations:",
        "mc_uncertainty": "Uncertainty Level:",
        "mc_low": "Low (±5%)", "mc_medium": "Medium (±15%)", "mc_high": "High (±30%)",
        "mc_mean": "Mean", "mc_ci": "95% CI", "mc_risk": "Ascites Risk",
        "report_generate": "📊 Generate Report",
        "report_param": "Parameter", "report_base": "Base Value",
        "report_min": "Jv_min", "report_max": "Jv_max",
        "report_sensitivity": "Sensitivity", "report_status": "Status",
        "status_low": "Low", "status_medium": "Medium", "status_high": "High",
        "download_csv": "📥 Download Report (CSV)",
        "nnn": "Prioritizing parameters",
        "sens_jv": "Jv Sensitivity", "sens_jnet": "Jnet Sensitivity",
        "sens_range": "Range",
        "sens_effect": "Effect of {param} on Outputs",
        "sens_base": "Base Value",
        "sens_heatmap_title": "Heatmap: {p1} vs {p2}",
        "sens_tornado_title": "Tornado Diagram: Effect on {output}",
        "sens_mc_title": "Jnet Distribution in {n} Simulations",
        "sens_mc_ci_label": "Ascites Threshold",
        "sens_box_title": "Jv and Jnet Distribution",
        "sens_anz": "📊 Sensitivity Analysis Table",
        "sens_anz2": "Comparison of Parameter Sensitivity",
        "bernoulli_title": "⚡ Bernoulli Sensitivity Analysis",
        "bernoulli_subtitle": "Effect of hemodynamic parameters on α and pressure drop",
        "bernoulli_1d": "📈 One-Dimensional",
        "bernoulli_2d": "🎯 Two-Dimensional",
        "bernoulli_report": "📊 Bernoulli Report",
        "bernoulli_desc": "Effect of hemodynamic parameters on α, pressure drop and flow.",
        "param_qportal": "Q_portal", "param_qartery": "Q_artery",
        "param_aportal": "A_portal", "param_ahepatic": "A_hepatic",
        "param_h": "h", "param_r0": "r₀", "param_L": "L",
        "param_beta": "β", "param_mu": "u∞", "param_tau": "ty",
        "output_alpha": "α", "output_dpsin": "ΔP_sin",
        "output_dptotal": "ΔP_total", "output_qtotal": "Q_total",
        "bernoulli_effect": "Effect of {param} on Hemodynamics",
        "bernoulli_heatmap": "Bernoulli Heatmap: {p1} vs {p2}",
        "bernoulli_report_title": "📊 Bernoulli Report",
        "bernoulli_sensitivity": "α Sensitivity",
        "bernoulli_alpha_min": "α_min", "bernoulli_alpha_max": "α_max",
        "bernoulli_high": "High", "bernoulli_low": "Low", "bernoulli_medium": "Medium",
        "sens_medium": "Medium", "kahesh": "Decrease", "afz": "Increase",
        "cache_clear": "🗑️ Clear Cache", "cache_cleared": "✅ Cache cleared!",
        "reset_title": "🔄 Reset All Settings",
        "upload_csv": "📤 Upload Patient Data (CSV)",
        "upload_help": "CSV columns: ΔP, Kf0, sigma, Pi0, Jmax, Km, dPi",
        "upload_run": "🚀 Run Model",
        "upload_status": "⚠️ Ascites Risk", "upload_compensated": "✅ Compensated",
        "upload_download": "📥 Download Results",
        "upload_error": "❌ Error: {e}",
        "3d_title": "📊 Interactive 3D Plot",
        "3d_info": "Select two parameters for 3D visualization:",
        "3d_param1": "Parameter 1 (X)", "3d_param2": "Parameter 2 (Y)",
        "3d_plot": "🎲 Draw 3D Plot",
        "validation_warning_flow": "⚠️ Flow must be positive!",
        "validation_warning_area": "⚠️ Area must be greater than zero!",
        "lang_label": "Language",
        "footer": "🩸 Hepatic Hemodynamics Simulator | Mandegar Alborz Research Center | 2025-2026",
    },
    "fa": {
        "app_title": "🩸 شبیه‌ساز تراوش کبد",
        "app_subtitle": "بر اساس مقاله *شبیه‌سازی جریان خون در کبد بر اساس اصول و معادلات مکانیک سیالات*",
        "settings": "⚙️ تنظیمات",
        "mode_label": "حالت α",
        "mode_auto": "🔄 خودکار", "mode_manual": "✋ دستی",
        "hemo_params": "پارامترهای همودینامیک",
        "portal_flow": "دبی ورید باب (L/min)", "artery_flow": "دبی سرخرگ کبدی (L/min)",
        "portal_area": "سطح مقطع ورید باب (cm²)", "hepatic_area": "سطح مقطع ورید فوق‌کبدی (cm²)",
        "height_diff": "اختلاف ارتفاع h (cm)",
        "sinusoid_params": "پارامترهای سینوزوئیدی",
        "mu_inf": "u∞ (Pa·s)", "tau_y": "ty (Pa)", "r0": "r₀ (μm)",
        "L": "L (μm)", "beta": "β (ضریب مخروطی)",
        "filtration_params": "پارامترهای تراوش",
        "kf0": "Kf₀", "sigma": "σ", "pi0": "Pi₀ (mmHg)", "dpi": "Δπ (mmHg)",
        "lymph_params": "پارامترهای لنفاوی",
        "jmax": "Jmax (ml/min)", "km": "Km (mmHg)", "max_dp": "حداکثر ΔP (mmHg)",
        "manual_alpha": "α (دستی)", "P_hep": "فشار سیاهرگ فوق کبدی",
        "calc_details": "📊 جزئیات محاسبات",
        "total_flow": "دبی کل", "portal_vel": "سرعت ورید باب",
        "hepatic_vel": "سرعت ورید فوق‌کبدی",
        "sinusoid_drop": "افت سینوزوئیدی", "height_drop": "افت ارتفاع",
        "kinetic_drop": "افت جنبشی", "total_drop": "کل افت",
        "alpha": "α", "shear_rate": "نرخ برش", "mu_app": "u ظاهری", "dp_total": "ΔP کل",
        "auto_results": "📊 نتایج خودکار", "manual_results": "📊 نتایج دستی",
        "deltaP": "ΔP", "kf_eff": "Kf", "pi_eff": "Pi", "jv": "Jv",
        "jnet": "Jnet", "jlymph": "Jlymph (ml/min)",
        "clinical_title": "🏥 تفسیر بالینی (ΔP = {dp:.2f} mmHg)",
        "status": "وضعیت", "fluid_status": "وضعیت مایعات", "ascites_pred": "پیش‌بینی آسیت",
        "key_values": "📊 مقادیر کلیدی", "dp_mmHg": "ΔP (mmHg)",
        "filtration_curves": "📊 منحنی‌های تراوش",
        "jv_curve": "Jv (تراوش)", "jlymph_curve": "Jlymph (لنفاوی)", "jnet_curve": "Jnet (خالص)",
        "zero_line": "J = 0", "threshold_line": "آستانه ۱۲ mmHg",
        "curves_title": "📊 منحنی‌های تراوش، تخلیه لنفاوی و تجمع خالص",
        "xaxis_dp": "ΔP (mmHg)", "yaxis_flow": "نرخ جریان (ml/min)",
        "nonlinear_behavior": "📊 رفتار غیرخطی",
        "kf_title": "ضریب فیلتراسیون غیرخطی Kf", "kf_yaxis": "Kf (ml/min/mmHg)",
        "pi_title": "فشار میان‌بافتی غیرخطی Pi", "pi_yaxis": "Pi (mmHg)",
        "key_points": "📊 مقادیر کلیدی", "clinical_interpretation": "🏥 تفسیر بالینی",
        "caption": "α = {alpha:.3f} | h = {h} cm | r₀ = {r0} μm | β = {beta} | Q_total = {q:.1f} L/min | μ = {mu:.5f} Pa·s",
        "info_text": "📌 در فشارهای پایین (< ۱۲ mmHg)، سیستم لنفاوی قادر به تخلیه است. پس از آستانه، Kf نمایی رشد کرده و تراوش از ظرفیت لنفاوی سبقت می‌گیرد.",
        "dynamic_title": "📈 پیش‌بینی دینامیک حجم آسیت",
        "dynamic_subtitle": "🔬 مدل دینامیک (معادله دیفرانسیل با اشباع)",
        "dynamic_desc": "در این مدل، فشار میان‌بافتی (Pi) با حجم افزایش می‌یابد اما به Pi_max محدود می‌شود. این محدودیت باعث کاهش تدریجی Jnet و سپس رشد دوباره حجم می‌شود. حل با روش اویلر.",
        "dynamic_time": "⏱️ مدت زمان شبیه‌سازی (ساعت)",
        "dynamic_k_elastance": "📊 ضریب الاستانس بافت (mmHg/mL)",
        "dynamic_pi_max": "📊 حداکثر فشار بین‌بافتی Pi_max (mmHg)",
        "dynamic_V0": "💧 حجم اولیه آسیت (mL)",
        "dynamic_run": "🚀 شبیه‌سازی دینامیک",
        "dynamic_success": "✅ شبیه‌سازی برای {time} ساعت انجام شد!",
        "dynamic_final_volume": "📊 حجم نهایی آسیت",
        "dynamic_initial_jnet": "📉 Jnet اولیه",
        "dynamic_final_jnet": "📉 Jnet نهایی",
        "dynamic_jnet_reduction": "📉 کاهش Jnet",
        "dynamic_volume_title": "دینامیک حجم آسیت",
        "dynamic_jnet_title": "تغییرات Jnet در طول زمان",
        "dynamic_pi_title": "تغییرات فشار میان‌بافتی (Pi)",
        "dynamic_flows_title": "تغییرات Jv و Jlymph در طول زمان",
        "dynamic_time_axis": "زمان (ساعت)",
        "dynamic_volume_axis": "حجم آسیت (mL)",
        "dynamic_jnet_axis": "Jnet (ml/min)",
        "dynamic_pi_axis": "Pi (mmHg)",
        "dynamic_flow_axis": "جریان (ml/min)",
        "dynamic_threshold": "آستانه ۵۰۰ mL",
        "dynamic_equilibrium": "نقطه تعادل",
        "dynamic_pi_max_line": "Pi_max",
        "dynamic_comparison": "📊 مقایسه مدل استاتیک و دینامیک",
        "dynamic_time_col": "زمان",
        "dynamic_static_col": "مدل استاتیک (خطی)",
        "dynamic_dynamic_col": "مدل دینامیک (غیرخطی)",
        "dynamic_1h": "۱ ساعت", "dynamic_6h": "۶ ساعت", "dynamic_24h": "۲۴ ساعت",
        "dynamic_interpretation": "📌 مدل استاتیک پیش‌بینی‌های غیرواقعی می‌دهد. مدل دینامیک با در نظر گرفتن مکانیسم‌های جبرانی و اشباع Pi، پیش‌بینی واقع‌بینانه‌تری دارد.",
        "dynamic_loading": "⏳ در حال حل معادله دیفرانسیل...",
        "dynamic_phase_1": "فاز ۱: تجمع سریع",
        "dynamic_phase_2": "فاز ۲: جبران (کندی)",
        "dynamic_phase_3": "فاز ۳: رشد دوباره پس از اشباع Pi",
        "clinical_expander": "📖 تفسیر بالینی کامل",
        "table_title": "📌 محدوده‌های بالینی",
        "table_col1": "محدوده ΔP (mmHg)", "table_col2": "وضعیت",
        "table_col3": "خطر آسیت", "table_col4": "مکانیسم غالب",
        "row1_1": "< ۸", "row1_2": "طبیعی", "row1_3": "🟢 بسیار کم", "row1_4": "تعادل استارلینگ",
        "row2_1": "۸ – ۱۲", "row2_2": "مرز هشدار", "row2_3": "🟡 جزئی", "row2_4": "شروع رشد Kf",
        "row3_1": "۱۲ – ۱۶", "row3_2": "پرفشاری خفیف-متوسط", "row3_3": "🔴 متوسط", "row3_4": "شکست هیدرولیکی",
        "row4_1": "> ۱۶", "row4_2": "پرفشاری شدید", "row4_3": "🔴 بالا", "row4_4": "رشد نمایی",
        "mechanisms_title": "🔬 مکانیسم‌های کلیدی",
        "mech1": "1. شکست هیدرولیکی در ΔP ≥ ۱۲ mmHg",
        "mech2": "2. آستانه بالینی ۱۲ mmHg",
        "mech3": "3. مکانیسم جبرانی Pi",
        "mech4": "4. اشباع لنفاوی",
        "clinical_app": "🏥 کاربرد بالینی",
        "clinical_app_text": "این مدل می‌تواند برای موارد زیر استفاده شود:",
        "clinical_app_1": "- پیش‌بینی خطر آسیت",
        "clinical_app_2": "- ارزیابی درمان‌ها",
        "clinical_app_3": "- طراحی مطالعات CFD",
        "comparison_title": "📊 مقایسه با مدل‌های پیشین",
        "comparison_col1": "ویژگی",
        "comparison_col2": "Siggers (2013)",
        "comparison_col3": "Dongaonkar (2018)",
        "comparison_col4": "Dongaonkar (2020)",
        "comparison_col5": "**مدل حاضر**",
        "comp_bernoulli": "برنولی اصلاح‌شده",
        "comp_viscous": "افت فشار ویسکوز",
        "comp_kf": "Kf غیرخطی",
        "comp_pi": "Pi غیرخطی",
        "comp_negative": "شار منفی",
        "comp_casson": "مدل کاسون",
        "comp_lymph": "تخلیه لنفاوی",
        "comp_dynamic": "مدل دینامیک",
        "comp_saturation": "اشباع Pi",
        "innovation_title": "**نوآوری اصلی:**",
        "innovation_text": "ترکیب برنولی اصلاح‌شده، پوازوی با شعاع متغیر، مدل کاسون، استارلینگ با Kf و Pi غیرخطی، و مدل دینامیک با اشباع Pi.",
        "sensitivity_title": "📊 تحلیل حساسیت پیشرفته",
        "sensitivity_subtitle": "Monte Carlo، Heatmap و Tornado Diagram",
        "sensitivity_1d": "📈 یک‌بعدی", "sensitivity_2d": "🎯 دو‌بعدی",
        "sensitivity_tornado": "🌪️ Tornado", "sensitivity_monte": "🎲 Monte Carlo",
        "sensitivity_report": "📊 گزارش جامع",
        "sensitivity_1d_desc": "تأثیر یک پارامتر بر خروجی‌ها.",
        "sensitivity_2d_desc": "تأثیر هم‌زمان دو پارامتر.",
        "sensitivity_tornado_desc": "تأثیر همه پارامترها.",
        "sensitivity_monte_desc": "تحلیل عدم‌قطعیت.",
        "sensitivity_report_desc": "خلاصه کامل تحلیل حساسیت.",
        "select_param": "🔍 انتخاب پارامتر:",
        "param_kf0": "Kf₀", "param_sigma": "σ", "param_pi0": "Pi₀",
        "param_jmax": "Jmax", "param_dpi": "Δπ", "param_km": "Km",
        "param_min": "min:", "param_max": "max:",
        "n_points": "تعداد نقاط:",
        "fixed_deltaP": "فشار ΔP (mmHg):",
        "output_type": "خروجی:",
        "output_jv": "Jv", "output_jnet": "Jnet", "output_both": "هر دو",
        "run_analysis": "🚀 اجرا",
        "param1": "پارامتر اول:", "param2": "پارامتر دوم:",
        "heatmap_output": "خروجی:",
        "heatmap_min": "حداقل", "heatmap_max": "حداکثر",
        "tornado_output": "خروجی:",
        "mc_simulations": "تعداد شبیه‌سازی:",
        "mc_uncertainty": "سطح عدم‌قطعیت:",
        "mc_low": "کم (±5%)", "mc_medium": "متوسط (±15%)", "mc_high": "زیاد (±30%)",
        "mc_mean": "میانگین", "mc_ci": "فاصله اطمینان 95%", "mc_risk": "خطر آسیت",
        "report_generate": "📊 تولید گزارش",
        "report_param": "پارامتر", "report_base": "مقدار پایه",
        "report_min": "Jv_min", "report_max": "Jv_max",
        "report_sensitivity": "حساسیت", "report_status": "وضعیت",
        "status_low": "پایین", "status_medium": "متوسط", "status_high": "بالا",
        "download_csv": "📥 دانلود گزارش (CSV)",
        "nnn": "اولویت‌بندی پارامترها",
        "sens_jv": "حساسیت Jv", "sens_jnet": "حساسیت Jnet",
        "sens_range": "محدوده",
        "sens_effect": "تأثیر {param} بر خروجی‌ها",
        "sens_base": "مقدار پایه",
        "sens_heatmap_title": "Heatmap: {p1} vs {p2}",
        "sens_tornado_title": "Tornado Diagram: تأثیر بر {output}",
        "sens_mc_title": "توزیع Jnet در {n} شبیه‌سازی",
        "sens_mc_ci_label": "آستانه آسیت",
        "sens_box_title": "توزیع Jv و Jnet",
        "sens_anz": "📊 جدول حساسیت",
        "sens_anz2": "مقایسه تحلیل حساسیت",
        "bernoulli_title": "⚡ تحلیل حساسیت برنولی",
        "bernoulli_subtitle": "تأثیر پارامترهای همودینامیک بر α و افت فشار",
        "bernoulli_1d": "📈 یک‌بعدی", "bernoulli_2d": "🎯 دو‌بعدی",
        "bernoulli_report": "📊 گزارش برنولی",
        "bernoulli_desc": "تأثیر پارامترهای همودینامیک بر α، افت فشار و دبی.",
        "param_qportal": "Q_portal", "param_qartery": "Q_artery",
        "param_aportal": "A_portal", "param_ahepatic": "A_hepatic",
        "param_h": "h", "param_r0": "r₀", "param_L": "L",
        "param_beta": "β", "param_mu": "u∞", "param_tau": "ty",
        "output_alpha": "α", "output_dpsin": "ΔP_sin",
        "output_dptotal": "ΔP_total", "output_qtotal": "Q_total",
        "bernoulli_effect": "تأثیر {param} بر همودینامیک",
        "bernoulli_heatmap": "Heatmap برنولی: {p1} vs {p2}",
        "bernoulli_report_title": "📊 گزارش برنولی",
        "bernoulli_sensitivity": "حساسیت α",
        "bernoulli_alpha_min": "α min", "bernoulli_alpha_max": "α max",
        "bernoulli_high": "زیاد", "bernoulli_low": "کم", "bernoulli_medium": "متوسط",
        "sens_medium": "متوسط", "kahesh": "کاهش", "afz": "افزایش",
        "cache_clear": "🗑️ پاک‌سازی کش", "cache_cleared": "✅ کش پاک شد!",
        "reset_title": "🔄 بازنشانی تنظیمات",
        "upload_csv": "📤 بارگذاری داده بیمار (CSV)",
        "upload_help": "ستون‌ها: ΔP, Kf0, sigma, Pi0, Jmax, Km, dPi",
        "upload_run": "🚀 اجرای مدل",
        "upload_status": "⚠️ خطر آسیت", "upload_compensated": "✅ جبران‌شده",
        "upload_download": "📥 دانلود نتایج",
        "upload_error": "❌ خطا: {e}",
        "3d_title": "📊 نمودار سه‌بعدی تعاملی",
        "3d_info": "دو پارامتر را برای نمایش سه‌بعدی انتخاب کنید:",
        "3d_param1": "پارامتر اول (X)", "3d_param2": "پارامتر دوم (Y)",
        "3d_plot": "🎲 رسم نمودار ۳D",
        "validation_warning_flow": "⚠️ دبی باید مثبت باشد!",
        "validation_warning_area": "⚠️ سطح مقطع باید بزرگتر از صفر باشد!",
        "lang_label": "زبان",
        "footer": "🩸 شبیه‌ساز همودینامیک کبد | پژوهش‌سرای ماندگار البرز | ۱۴۰۴-۱۴۰۵",
    }
}

# ============================================================
# Language
# ============================================================
if "lang" not in st.session_state:
    st.session_state.lang = "fa"


def set_lang_en():
    st.session_state.lang = "en"
    st.rerun()


def set_lang_fa():
    st.session_state.lang = "fa"
    st.rerun()


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.write("🌐 زبان / Language")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇬🇧 English", use_container_width=True):
            set_lang_en()
    with col2:
        if st.button("🇮🇷 فارسی", use_container_width=True):
            set_lang_fa()
    st.divider()

    lang = st.session_state.lang
    t = TEXTS[lang]

    if lang == "fa":
        st.markdown("""
        <style>
        h1, h2, h3, h4, h5, h6, .stMarkdown, .stMarkdown p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown p,
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

    Q_portal = st.number_input(t["portal_flow"], 0.3, 2.0, 1.1, 0.05) / 1000 / 60
    Q_artery = st.number_input(t["artery_flow"], 0.1, 0.8, 0.35, 0.05) / 1000 / 60
    A_portal = st.number_input(t["portal_area"], 0.5, 5.0, 1.1, 0.1) * 1e-4
    A_hepatic = st.number_input(t["hepatic_area"], 0.5, 10.0, 0.6, 0.1) * 1e-4
    P_hep = st.number_input(t["P_hep"], 0.0, 8.0, 4.0, 0.5)
    h_cm = st.number_input(t["height_diff"], 0.0, 10.0, 4.0, 0.1)
    h = h_cm / 100

    st.header(t["sinusoid_params"])
    mu_inf = st.number_input(t["mu_inf"], 0.001, 0.01, 0.0040, 0.0005, format="%.4f")
    tau_y = st.number_input(t["tau_y"], 0.001, 0.01, 0.005, 0.0005, format="%.4f")
    r0_um = st.number_input(t["r0"], 2.0, 6.0, 4.0, 0.05)
    r0 = r0_um * 1e-6
    L_um = st.number_input(t["L"], 100, 500, 365, 5)
    L = L_um * 1e-6
    beta = st.number_input(t["beta"], 0.0, 0.8, 0.10, 0.01)

    st.header(t["filtration_params"])
    Kf0 = st.slider(t["kf0"], 1.0, 8.0, 3.0, 0.1)
    sigma = st.slider(t["sigma"], 0.1, 0.4, 0.22, 0.01)
    Pi0 = st.slider(t["pi0"], 0.1, 2.0, 0.5, 0.1)
    dPi = st.slider(t["dpi"], 20, 25, 22, 1)

    st.header(t["lymph_params"])
    Jmax = st.number_input(t["jmax"], 10, 50, 35, 1)
    Km = st.number_input(t["km"], 0.1, 2.0, 0.63, 0.01)
    max_deltaP = st.slider(t["max_dp"], 12, 30, 20)

# ============================================================
# Core Calculations
# ============================================================
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
    dp_v = 0.5 * rho_blood * (vh ** 2 - vp ** 2)
    dp_total = dp_sin + dp_h + dp_v

params = {
    'alpha': alpha, 'Kf0': Kf0, 'sigma': sigma, 'Pi0': Pi0,
    'Jmax': Jmax, 'Km': Km, 'mu_inf': mu_inf, 'tau_y': tau_y,
    'r0': r0_um, 'beta': beta, 'dPi': dPi
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


# ============================================================
# Main UI
# ============================================================
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

if mode == t["mode_auto"]:
    st.subheader(t["auto_results"])
    deltaP_analysis = dp_total / mmHg_to_Pa
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
    st.info(f"ΔP = {deltaP_analysis:.2f} mmHg | {clinical['status']} | {clinical['ascites_prediction']}")

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
    st.info(f"ΔP = {deltaP_analysis:.2f} mmHg | {clinical['status']} | {clinical['ascites_prediction']}")

    st.subheader(t["filtration_curves"])
    deltaP_range = np.linspace(0, max_deltaP, 300)
    Jv_list, Kf_list, Pi_list, Jlymph_list, Jnet_list = [], [], [], [], []

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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=deltaP_range, y=Jv_list, mode='lines', name=t["jv_curve"], line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=deltaP_range, y=Jlymph_list, mode='lines', name=t["jlymph_curve"], line=dict(color='green', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=deltaP_range, y=Jnet_list, mode='lines', name=t["jnet_curve"], line=dict(color='red', width=3, dash='dot'), fill='tozeroy', fillcolor='rgba(255,0,0,0.1)'))
    fig.add_hline(y=0, line_dash='dot', line_color='gray', annotation_text=t["zero_line"])
    fig.add_vline(x=12, line_dash='dot', line_color='red', annotation_text=t["threshold_line"])
    fig.update_layout(title=t["curves_title"], xaxis_title=t["xaxis_dp"], yaxis_title=t["yaxis_flow"], template=get_plotly_template(), height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(t["nonlinear_behavior"])
    col1, col2 = st.columns(2)
    with col1:
        fig_kf = go.Figure()
        fig_kf.add_trace(go.Scatter(x=deltaP_range, y=Kf_list, mode='lines', name=t["kf_eff"], line=dict(color='purple', width=3)))
        fig_kf.add_hline(y=Kf0, line_dash='dot', line_color='gray')
        fig_kf.add_vline(x=12, line_dash='dot', line_color='red')
        fig_kf.update_layout(title=t["kf_title"], xaxis_title=t["xaxis_dp"], yaxis_title=t["kf_yaxis"], template=get_plotly_template(), height=350)
        st.plotly_chart(fig_kf, use_container_width=True)
    with col2:
        fig_pi = go.Figure()
        fig_pi.add_trace(go.Scatter(x=deltaP_range, y=Pi_list, mode='lines', name=t["pi_eff"], line=dict(color='orange', width=3)))
        fig_pi.add_hline(y=Pi0, line_dash='dot', line_color='gray')
        fig_pi.add_vline(x=12, line_dash='dot', line_color='red')
        fig_pi.update_layout(title=t["pi_title"], xaxis_title=t["xaxis_dp"], yaxis_title=t["pi_yaxis"], template=get_plotly_template(), height=350)
        st.plotly_chart(fig_pi, use_container_width=True)

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
    st.dataframe(df.style.map(color_jnet, subset=[t["jnet"]]), use_container_width=True, hide_index=True)

    st.subheader(t["clinical_interpretation"])
    cols = st.columns(3)
    for i, dp in enumerate([8, 12, 16]):
        idx = int(dp / max_deltaP * len(deltaP_range))
        if idx >= len(deltaP_range):
            idx = len(deltaP_range) - 1
        clinical = get_clinical_interpretation(dp, Jv_list[idx], Jnet_list[idx], lang=lang)
        with cols[i]:
            color_bg = '#d4edda' if clinical['status'] in [t["row1_2"], "طبیعی"] else '#fff3cd' if clinical['status'] in [t["row2_2"], "مرز هشدار"] else '#f8d7da'
            st.markdown(f"""
            <div style="background-color: {color_bg}; padding: 15px; border-radius: 10px; margin: 5px 0; border: 1px solid #ddd;">
                <h4 style="margin: 0; text-align: center;">{clinical['color']} ΔP = {dp} mmHg</h4>
                <hr style="margin: 10px 0;">
                <b>{t['status']}:</b> {clinical['status']}<br>
                <b>{t['jv']}:</b> {Jv_list[idx]:.2f} ml/min<br>
                <b>{t['jnet']}:</b> {Jnet_list[idx]:.2f} ml/min<br>
                <b>{clinical['ascites_prediction']}</b>
            </div>
            """, unsafe_allow_html=True)

    st.caption(t["caption"].format(alpha=alpha, h=h_cm, r0=r0_um, beta=beta, q=(Q_total * 1000 * 60), mu=mu_app))
    st.info(t["info_text"])

    st.divider()
    st.subheader(t["dynamic_title"])

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 15px; border-radius: 10px; margin-bottom: 15px; color: white;">
        <h4 style="margin: 0; color: #00d2ff;">{t['dynamic_subtitle']}</h4>
        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 13px;">{t['dynamic_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        time_hours = st.slider(t["dynamic_time"], 1, 72, 24, 1)
    with col2:
        k_elastance = st.slider(t["dynamic_k_elastance"], 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
    with col3:
        Pi_max = st.slider(t["dynamic_pi_max"], 2.0, 15.0, 10.0, 0.5)
    with col4:
        V0 = st.number_input(t["dynamic_V0"], 0, 1000, 0, 10)

    if st.button(t["dynamic_run"], use_container_width=True, type="primary"):
        with st.spinner(t["dynamic_loading"]):
            time_array, V_array, Jnet_array, Pi_array, Jv_array, Jlymph_array = predict_ascites_volume_dynamic(
                Kf=Kf, alpha=alpha, sigma=sigma, dpi=dPi,
                P_hepatic=P_hep, Pi0=Pi0, k_elastance=k_elastance,
                Jmax=Jmax, Km=Km, deltaP=deltaP_analysis,
                time_hours=time_hours, V0=V0, dt=0.01, Pi_max=Pi_max
            )

        st.success(t["dynamic_success"].format(time=time_hours))

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t["dynamic_final_volume"], f"{V_array[-1]:.1f} mL")
        with col2:
            st.metric(t["dynamic_initial_jnet"], f"{Jnet_array[0]:.2f} ml/min")
        with col3:
            st.metric(t["dynamic_final_jnet"], f"{Jnet_array[-1]:.2f} ml/min")
        with col4:
            if Jnet_array[0] != 0:
                reduction = ((Jnet_array[0] - Jnet_array[-1]) / Jnet_array[0] * 100)
                st.metric(t["dynamic_jnet_reduction"], f"{reduction:.1f}%")
            else:
                st.metric(t["dynamic_jnet_reduction"], "0%")

        fig_V = go.Figure()
        fig_V.add_trace(go.Scatter(x=time_array, y=V_array, mode='lines',
                                   name=t["dynamic_volume_title"],
                                   line=dict(color='#ff4b4b', width=3),
                                   fill='tozeroy', fillcolor='rgba(255,75,75,0.1)'))
        fig_V.add_hline(y=500, line_dash='dash', line_color='orange',
                        annotation_text=t["dynamic_threshold"], annotation_position='top right')
        fig_V.update_layout(title=f'<b>{t["dynamic_volume_title"]}</b>',
                            xaxis_title=t["dynamic_time_axis"],
                            yaxis_title=t["dynamic_volume_axis"],
                            template=get_plotly_template(), height=400, hovermode='x unified')
        st.plotly_chart(fig_V, use_container_width=True)

        fig_Jnet = go.Figure()
        fig_Jnet.add_trace(go.Scatter(x=time_array, y=Jnet_array, mode='lines',
                                      name='Jnet', line=dict(color='#7c3aed', width=3)))
        fig_Jnet.add_hline(y=0, line_dash='dot', line_color='gray',
                           annotation_text=t["dynamic_equilibrium"], annotation_position='bottom right')
        fig_Jnet.update_layout(title=f'<b>{t["dynamic_jnet_title"]}</b>',
                               xaxis_title=t["dynamic_time_axis"],
                               yaxis_title=t["dynamic_jnet_axis"],
                               template=get_plotly_template(), height=400, hovermode='x unified')
        st.plotly_chart(fig_Jnet, use_container_width=True)

        fig_Pi = go.Figure()
        fig_Pi.add_trace(go.Scatter(x=time_array, y=Pi_array, mode='lines',
                                    name='Pi', line=dict(color='#f9a825', width=3)))
        fig_Pi.add_hline(y=Pi_max, line_dash='dash', line_color='red',
                         annotation_text=t["dynamic_pi_max_line"], annotation_position='top right')
        fig_Pi.update_layout(title=f'<b>{t["dynamic_pi_title"]}</b>',
                             xaxis_title=t["dynamic_time_axis"],
                             yaxis_title=t["dynamic_pi_axis"],
                             template=get_plotly_template(), height=400, hovermode='x unified')
        st.plotly_chart(fig_Pi, use_container_width=True)

        fig_flows = go.Figure()
        fig_flows.add_trace(go.Scatter(x=time_array, y=Jv_array, mode='lines',
                                       name='Jv (Filtration)', line=dict(color='blue', width=3)))
        fig_flows.add_trace(go.Scatter(x=time_array, y=Jlymph_array, mode='lines',
                                       name='Jlymph (Lymphatic)', line=dict(color='green', width=3, dash='dash')))
        fig_flows.update_layout(title=f'<b>{t["dynamic_flows_title"]}</b>',
                                xaxis_title=t["dynamic_time_axis"],
                                yaxis_title=t["dynamic_flow_axis"],
                                template=get_plotly_template(), height=400, hovermode='x unified')
        st.plotly_chart(fig_flows, use_container_width=True)

        st.subheader("📊 فازهای دینامیک آسیت")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(t["dynamic_phase_1"])
        with col2:
            st.warning(t["dynamic_phase_2"])
        with col3:
            st.error(t["dynamic_phase_3"])

        st.subheader(t["dynamic_comparison"])
        V_static_1h = Jnet_array[0] * 60
        V_static_6h = Jnet_array[0] * 360
        V_static_24h = Jnet_array[0] * 1440

        comparison_data = {
            t["dynamic_time_col"]: [t["dynamic_1h"], t["dynamic_6h"], t["dynamic_24h"]],
            t["dynamic_static_col"]: [f"{V_static_1h:.0f} mL", f"{V_static_6h:.0f} mL", f"{V_static_24h:.0f} mL"],
            t["dynamic_dynamic_col"]: [
                f"{np.interp(1, time_array, V_array):.0f} mL",
                f"{np.interp(6, time_array, V_array):.0f} mL",
                f"{np.interp(24, time_array, V_array):.0f} mL"
            ]
        }
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        st.info(t["dynamic_interpretation"])

# ============================================================
# Clinical Expander
# ============================================================
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

# ============================================================
# Bernoulli Sensitivity
# ============================================================
with st.expander(t["bernoulli_title"], expanded=False):
    btab1, btab2, btab3 = st.tabs([t["bernoulli_1d"], t["bernoulli_2d"], t["bernoulli_report"]])

    bernoulli_params = {
        t["param_qportal"]: "Q_portal", t["param_qartery"]: "Q_artery",
        t["param_aportal"]: "A_portal", t["param_ahepatic"]: "A_hepatic",
        t["param_h"]: "h", t["param_r0"]: "r0", t["param_L"]: "L",
        t["param_beta"]: "beta", t["param_mu"]: "mu_inf", t["param_tau"]: "tau_y"
    }
    bernoulli_ranges = {
        "Q_portal": (1.0, 1.3, 1.1, 0.05), "Q_artery": (0.1, 0.6, 0.35, 0.05),
        "A_portal": (0.5, 3.0, 1.1, 0.1), "A_hepatic": (0.3, 1.2, 0.6, 0.1),
        "h": (3.0, 6.0, 4.0, 0.1), "r0": (4.0, 5.0, 4.5, 0.5),
        "L": (250, 400, 340, 5), "beta": (0.0, 0.4, 0.05, 0.05),
        "mu_inf": (0.002, 0.005, 0.004, 0.001), "tau_y": (0.002, 0.008, 0.005, 0.001)
    }

    with btab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_bparam = st.selectbox(t["select_param"], list(bernoulli_params.keys()), key="bern_param")
            bparam_key = bernoulli_params[selected_bparam]
            min_val, max_val, default_val, step = bernoulli_ranges[bparam_key]
            b_range_min = st.number_input(t["param_min"], min_val, max_val, min_val, step, format="%.2f", key="brmin")
            b_range_max = st.number_input(t["param_max"], min_val, max_val, max_val, step, format="%.2f", key="brmax")
            b_n_points = st.slider(t["n_points"], 10, 100, 30, 5, key="bnpts")
            b_output_type = st.selectbox(t["output_type"], [t["output_alpha"], t["output_dpsin"], t["output_dptotal"], t["output_qtotal"]], key="bout_type")
            run_b1d = st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_b1d")
        with col2:
            if run_b1d:
                b_param_range = np.linspace(b_range_min, b_range_max, b_n_points)
                alpha_vals, dp_sin_vals, dp_total_vals, Q_total_vals = [], [], [], []
                for val in b_param_range:
                    temp_Qp, temp_Qa, temp_Ap, temp_Ah, temp_h, temp_r0, temp_L, temp_beta, temp_mu, temp_tau = Q_portal, Q_artery, A_portal, A_hepatic, h, r0, L, beta, mu_inf, tau_y
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
                    temp_alpha, temp_Qtotal, _, _, temp_dpsin, _, _, temp_dptotal, _ = calc_alpha(temp_Qp, temp_Qa, temp_Ap, temp_Ah, temp_h, temp_r0, temp_beta, temp_L, temp_mu, temp_tau)
                    alpha_vals.append(temp_alpha)
                    dp_sin_vals.append(temp_dpsin / mmHg_to_Pa)
                    dp_total_vals.append(temp_dptotal / mmHg_to_Pa)
                    Q_total_vals.append(temp_Qtotal * 1000 * 60)
                fig_b = go.Figure()
                if b_output_type == t["output_alpha"]:
                    fig_b.add_trace(go.Scatter(x=b_param_range, y=alpha_vals, mode='lines+markers', name=t["output_alpha"], line=dict(color='#00d2ff', width=3)))
                elif b_output_type == t["output_dpsin"]:
                    fig_b.add_trace(go.Scatter(x=b_param_range, y=dp_sin_vals, mode='lines+markers', name=t["output_dpsin"], line=dict(color='#ff6b6b', width=3)))
                elif b_output_type == t["output_dptotal"]:
                    fig_b.add_trace(go.Scatter(x=b_param_range, y=dp_total_vals, mode='lines+markers', name=t["output_dptotal"], line=dict(color='#7c3aed', width=3)))
                elif b_output_type == t["output_qtotal"]:
                    fig_b.add_trace(go.Scatter(x=b_param_range, y=Q_total_vals, mode='lines+markers', name=t["output_qtotal"], line=dict(color='#f9a825', width=3)))
                fig_b.add_vline(x=default_val, line_dash='dash', line_color='orange')
                fig_b.update_layout(title=t["bernoulli_effect"].format(param=selected_bparam), xaxis_title=selected_bparam, yaxis_title="Value", template=get_plotly_template(), height=450)
                st.plotly_chart(fig_b, use_container_width=True)

    with btab2:
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
        if st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_bhm"):
            x_vals = np.linspace(br1_min, br1_max, bn1)
            y_vals = np.linspace(br2_min, br2_max, bn2)
            Z = np.zeros((bn2, bn1))
            for i, v1 in enumerate(x_vals):
                for j, v2 in enumerate(y_vals):
                    temp_Qp, temp_Qa, temp_Ap, temp_Ah, temp_h, temp_r0, temp_L, temp_beta, temp_mu, temp_tau = Q_portal, Q_artery, A_portal, A_hepatic, h, r0, L, beta, mu_inf, tau_y
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
                    temp_alpha, _, _, _, temp_dpsin, _, _, temp_dptotal, _ = calc_alpha(temp_Qp, temp_Qa, temp_Ap, temp_Ah, temp_h, temp_r0, temp_beta, temp_L, temp_mu, temp_tau)
                    if bhm_output == t["output_alpha"]:
                        Z[j, i] = temp_alpha
                    elif bhm_output == t["output_dpsin"]:
                        Z[j, i] = temp_dpsin / mmHg_to_Pa
                    else:
                        Z[j, i] = temp_dptotal / mmHg_to_Pa
            fig_bhm = go.Figure(data=go.Heatmap(z=Z, x=x_vals, y=y_vals, colorscale='Viridis'))
            fig_bhm.update_layout(title=t["bernoulli_heatmap"].format(p1=bparam1, p2=bparam2), xaxis_title=bparam1, yaxis_title=bparam2, height=550)
            st.plotly_chart(fig_bhm, use_container_width=True)
            fig_3d = go.Figure(data=[go.Surface(z=Z, x=x_vals, y=y_vals, colorscale='Viridis')])
            fig_3d.update_layout(title=f"3D: {bparam1} & {bparam2}", scene=dict(xaxis_title=bparam1, yaxis_title=bparam2, zaxis_title=bhm_output), height=600)
            st.plotly_chart(fig_3d, use_container_width=True)

    with btab3:
        if st.button(t["report_generate"], use_container_width=True, type="primary", key="gen_breport"):
            report_data = []
            for key, name in bernoulli_params.items():
                min_v, max_v, default_v, _ = bernoulli_ranges[name]
                temp_alpha_base, _, _, _, _, _, _, _, _ = calc_alpha(Q_portal, Q_artery, A_portal, A_hepatic, h, r0, beta, L, mu_inf, tau_y)
                temp_alpha_min, _, _, _, _, _, _, _, _ = calc_alpha(Q_portal if name != "Q_portal" else min_v / 1000 / 60, Q_artery if name != "Q_artery" else min_v / 1000 / 60, A_portal if name != "A_portal" else min_v * 1e-4, A_hepatic if name != "A_hepatic" else min_v * 1e-4, h if name != "h" else min_v / 100, r0 if name != "r0" else min_v * 1e-6, beta if name != "beta" else min_v, L if name != "L" else min_v * 1e-6, mu_inf if name != "mu_inf" else min_v, tau_y if name != "tau_y" else min_v)
                temp_alpha_max, _, _, _, _, _, _, _, _ = calc_alpha(Q_portal if name != "Q_portal" else max_v / 1000 / 60, Q_artery if name != "Q_artery" else max_v / 1000 / 60, A_portal if name != "A_portal" else max_v * 1e-4, A_hepatic if name != "A_hepatic" else max_v * 1e-4, h if name != "h" else max_v / 100, r0 if name != "r0" else max_v * 1e-6, beta if name != "beta" else max_v, L if name != "L" else max_v * 1e-6, mu_inf if name != "mu_inf" else max_v, tau_y if name != "tau_y" else max_v)
                sensitivity = (temp_alpha_max - temp_alpha_min) / (temp_alpha_base + 1e-10)
                report_data.append({t['report_param']: key, t['report_base']: default_v, 'α_min': temp_alpha_min, 'α_max': temp_alpha_max, t['report_sensitivity']: sensitivity, t['report_status']: t["status_low"] if abs(sensitivity) < 0.1 else t["status_medium"] if abs(sensitivity) < 0.3 else t["status_high"]})
            df_breport = pd.DataFrame(report_data)
            st.dataframe(df_breport, use_container_width=True, hide_index=True)
            fig_brep = go.Figure()
            fig_brep.add_trace(go.Bar(x=df_breport[t['report_param']], y=df_breport[t['report_sensitivity']], marker_color='#00d2ff'))
            fig_brep.update_layout(title=t["sens_anz2"], xaxis_title=t['report_param'], yaxis_title=t['report_sensitivity'], template=get_plotly_template(), height=450)
            st.plotly_chart(fig_brep, use_container_width=True)
            csv_breport = df_breport.to_csv(index=False)
            st.download_button(label=t["download_csv"], data=csv_breport, file_name="bernoulli_report.csv", mime="text/csv")

# ============================================================
# Advanced Sensitivity Analysis
# ============================================================
with st.expander(t["sensitivity_title"], expanded=False):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([t["sensitivity_1d"], t["sensitivity_2d"], t["sensitivity_tornado"], t["sensitivity_monte"], t["sensitivity_report"]])

    param_options = {t["param_kf0"]: "Kf0", t["param_sigma"]: "sigma", t["param_pi0"]: "Pi0", t["param_jmax"]: "Jmax", t["param_dpi"]: "dPi", t["param_km"]: "Km"}
    ranges = {"Kf0": (1.0, 5.0, 3.0, 0.5), "sigma": (0.1, 0.4, 0.22, 0.02), "Pi0": (0.1, 2.0, 0.5, 0.1), "Jmax": (30, 50, 40, 5), "dPi": (18, 26, 22, 1), "Km": (0.1, 1.5, 0.74, 0.1)}

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_param = st.selectbox(t["select_param"], list(param_options.keys()), key="sens_param1")
            param_key = param_options[selected_param]
            min_val, max_val, default_val, step = ranges[param_key]
            range_min = st.number_input(t["param_min"], min_val, max_val, min_val, step, format="%.3f", key="rmin")
            range_max = st.number_input(t["param_max"], min_val, max_val, max_val, step, format="%.3f", key="rmax")
            n_points = st.slider(t["n_points"], 10, 100, 50, 5, key="npts")
            fixed_deltaP = st.slider(t["fixed_deltaP"], 0.0, 25.0, 12.0, 0.5, key="fixed_dp1")
            output_type = st.selectbox(t["output_type"], [t["output_jv"], t["output_jnet"], t["output_both"]], key="out_type")
            run_1d = st.button(t["run_analysis"], use_container_width=True, type="primary", key="run1d")
        with col2:
            if run_1d:
                param_range = np.linspace(range_min, range_max, n_points)
                Jv_vals, Jnet_vals, Kf_vals = [], [], []
                for val in param_range:
                    temp_params = params.copy()
                    temp_params[param_key] = val
                    Kf = calc_Kf_nonlinear(temp_params['Kf0'], fixed_deltaP)
                    Pi = calc_Pi_nonlinear(temp_params['Pi0'], fixed_deltaP)
                    Jv = calc_Jv(fixed_deltaP, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'], P_hep)
                    Jlymph = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi)
                    Jnet = calc_Jnet(Jv, Jlymph)
                    Jv_vals.append(Jv)
                    Jnet_vals.append(Jnet)
                    Kf_vals.append(Kf)
                fig = go.Figure()
                if output_type in [t["output_jv"], t["output_both"]]:
                    fig.add_trace(go.Scatter(x=param_range, y=Jv_vals, mode='lines+markers', name=t["output_jv"], line=dict(color='#00d2ff', width=3)))
                if output_type in [t["output_jnet"], t["output_both"]]:
                    fig.add_trace(go.Scatter(x=param_range, y=Jnet_vals, mode='lines+markers', name=t["output_jnet"], line=dict(color='#ff6b6b', width=3)))
                fig.update_layout(title=t["sens_effect"].format(param=selected_param), xaxis_title=selected_param, yaxis_title=t["yaxis_flow"], template=get_plotly_template(), height=450)
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
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
        if st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_hm"):
            x_vals = np.linspace(r1_min, r1_max, n1)
            y_vals = np.linspace(r2_min, r2_max, n2)
            Z = np.zeros((n2, n1))
            for i, v1 in enumerate(x_vals):
                for j, v2 in enumerate(y_vals):
                    temp_params = params.copy()
                    temp_params[p1_key] = v1
                    temp_params[p2_key] = v2
                    Kf = calc_Kf_nonlinear(temp_params['Kf0'], fixed_dp_hm)
                    Pi = calc_Pi_nonlinear(temp_params['Pi0'], fixed_dp_hm)
                    Jv = calc_Jv(fixed_dp_hm, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'], P_hep)
                    Jlymph = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi)
                    Jnet = calc_Jnet(Jv, Jlymph)
                    Z[j, i] = Jnet
            fig_hm = go.Figure(data=go.Heatmap(z=Z, x=x_vals, y=y_vals, colorscale='RdYlGn', zmid=0))
            fig_hm.update_layout(title=t["sens_heatmap_title"].format(p1=param1, p2=param2), xaxis_title=param1, yaxis_title=param2, height=550)
            st.plotly_chart(fig_hm, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            dp_tornado = st.slider(t["fixed_deltaP"], 0.0, 25.0, 12.0, 0.5, key="tor_dp")
        with col2:
            output_tornado = st.selectbox(t["tornado_output"], ["Jnet", "Jv"], key="tor_out")
        if st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_tor"):
            results = []
            for key, name in param_options.items():
                min_v, max_v, default_v, _ = ranges[name]
                temp_params = params.copy()
                Kf = calc_Kf_nonlinear(temp_params['Kf0'], dp_tornado)
                Pi = calc_Pi_nonlinear(temp_params['Pi0'], dp_tornado)
                Jv = calc_Jv(dp_tornado, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'], P_hep)
                Jlymph = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi)
                Jnet = Jv - Jlymph
                base = Jnet if output_tornado == "Jnet" else Jv
                temp_params[name] = min_v
                Kf_min = calc_Kf_nonlinear(temp_params['Kf0'], dp_tornado)
                Pi_min = calc_Pi_nonlinear(temp_params['Pi0'], dp_tornado)
                Jv_min = calc_Jv(dp_tornado, Kf_min, temp_params['alpha'], temp_params['sigma'], Pi_min, temp_params['dPi'], P_hep)
                Jlymph_min = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi_min)
                Jnet_min = Jv_min - Jlymph_min
                val_min = Jnet_min if output_tornado == "Jnet" else Jv_min
                temp_params[name] = max_v
                Kf_max = calc_Kf_nonlinear(temp_params['Kf0'], dp_tornado)
                Pi_max_v = calc_Pi_nonlinear(temp_params['Pi0'], dp_tornado)
                Jv_max = calc_Jv(dp_tornado, Kf_max, temp_params['alpha'], temp_params['sigma'], Pi_max_v, temp_params['dPi'], P_hep)
                Jlymph_max = calc_Jlymph(temp_params['Jmax'], temp_params['Km'], Pi_max_v)
                Jnet_max = Jv_max - Jlymph_max
                val_max = Jnet_max if output_tornado == "Jnet" else Jv_max
                results.append({'parameter': key, 'base': base, 'min': val_min - base, 'max': val_max - base, 'range': abs(val_max - val_min)})
            results = sorted(results, key=lambda x: x['range'], reverse=True)
            fig_tor = go.Figure()
            names = [r['parameter'] for r in results]
            min_vals = [r['min'] for r in results]
            max_vals = [r['max'] for r in results]
            fig_tor.add_trace(go.Bar(y=names, x=min_vals, name=t["kahesh"], orientation='h', marker_color='#ff6b6b'))
            fig_tor.add_trace(go.Bar(y=names, x=max_vals, name=t["afz"], orientation='h', marker_color='#00d2ff'))
            fig_tor.update_layout(title=t["sens_tornado_title"].format(output=output_tornado), xaxis_title=t["sens_range"], template=get_plotly_template(), height=500, barmode='relative')
            st.plotly_chart(fig_tor, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            n_simulations = st.slider(t["mc_simulations"], 100, 100000, 1000, 100, key="mc_n")
            mc_dp = st.slider(t["fixed_deltaP"], 0.0, 25.0, 12.0, 0.5, key="mc_dp")
        with col2:
            uncertainty_level = st.select_slider(t["mc_uncertainty"], options=[t["mc_low"], t["mc_medium"], t["mc_high"]], key="mc_unc")
            unc_factor = {t["mc_low"]: 0.05, t["mc_medium"]: 0.15, t["mc_high"]: 0.30}[uncertainty_level]
        if st.button(t["run_analysis"], use_container_width=True, type="primary", key="run_mc"):
            np.random.seed(42)
            Kf_samples = np.clip(np.random.normal(params['Kf0'], params['Kf0'] * unc_factor, n_simulations), 0.5, 10)
            sigma_samples = np.clip(np.random.normal(params['sigma'], params['sigma'] * unc_factor, n_simulations), 0.05, 0.5)
            Pi_samples = np.clip(np.random.normal(params['Pi0'], params['Pi0'] * unc_factor, n_simulations), 0.05, 3)
            dPi_samples = np.clip(np.random.normal(params['dPi'], params['dPi'] * unc_factor, n_simulations), 15, 28)
            Jv_samples, Jnet_samples = [], []
            for i in range(n_simulations):
                Kf = calc_Kf_nonlinear(Kf_samples[i], mc_dp)
                Pi = calc_Pi_nonlinear(Pi_samples[i], mc_dp)
                Jv = calc_Jv(mc_dp, Kf, params['alpha'], sigma_samples[i], Pi, dPi_samples[i], P_hep)
                Jlymph = calc_Jlymph(params['Jmax'], params['Km'], Pi)
                Jnet = calc_Jnet(Jv, Jlymph)
                Jv_samples.append(Jv)
                Jnet_samples.append(Jnet)
            c1, c2, c3 = st.columns(3)
            c1.metric(t["mc_mean"], f"{np.mean(Jv_samples):.2f}", delta=f"±{np.std(Jv_samples):.2f}")
            c2.metric(t["mc_ci"], f"[{np.percentile(Jv_samples, 2.5):.2f}, {np.percentile(Jv_samples, 97.5):.2f}]")
            c3.metric(t["mc_risk"], f"{np.mean(np.array(Jnet_samples) > 0) * 100:.1f}%")
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(x=Jnet_samples, nbinsx=50, marker_color='#7c3aed'))
            fig_mc.add_vline(x=0, line_dash='dash', line_color='red')
            fig_mc.update_layout(title=t["sens_mc_title"].format(n=n_simulations), xaxis_title="Jnet (ml/min)", yaxis_title="Count", template=get_plotly_template(), height=400)
            st.plotly_chart(fig_mc, use_container_width=True)

    with tab5:
        if st.button(t["report_generate"], use_container_width=True, type="primary", key="gen_report"):
            report_data = []
            for key, name in param_options.items():
                min_v, max_v, default_v, _ = ranges[name]
                temp_params = params.copy()
                Kf = calc_Kf_nonlinear(temp_params['Kf0'], 12)
                Pi = calc_Pi_nonlinear(temp_params['Pi0'], 12)
                Jv_base = calc_Jv(12, Kf, temp_params['alpha'], temp_params['sigma'], Pi, temp_params['dPi'], P_hep)
                temp_params[name] = min_v
                Kf_min = calc_Kf_nonlinear(temp_params['Kf0'], 12)
                Pi_min = calc_Pi_nonlinear(temp_params['Pi0'], 12)
                Jv_min = calc_Jv(12, Kf_min, temp_params['alpha'], temp_params['sigma'], Pi_min, temp_params['dPi'], P_hep)
                temp_params[name] = max_v
                Kf_max = calc_Kf_nonlinear(temp_params['Kf0'], 12)
                Pi_max_v = calc_Pi_nonlinear(temp_params['Pi0'], 12)
                Jv_max = calc_Jv(12, Kf_max, temp_params['alpha'], temp_params['sigma'], Pi_max_v, temp_params['dPi'], P_hep)
                sensitivity = (Jv_max - Jv_min) / (Jv_base + 1e-10)
                report_data.append({t['report_param']: key, t['report_base']: default_v, t['report_min']: Jv_min, t['report_max']: Jv_max, t['report_sensitivity']: sensitivity, t['report_status']: t["status_low"] if abs(sensitivity) < 0.5 else t["status_medium"] if abs(sensitivity) < 0.8 else t["status_high"]})
            df_report = pd.DataFrame(report_data)
            st.dataframe(df_report, use_container_width=True, hide_index=True)
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=df_report[t['report_param']], y=df_report[t['report_sensitivity']], marker_color='#00d2ff'))
            fig_comp.update_layout(title=t["sens_anz2"], xaxis_title=t['report_param'], yaxis_title=t['report_sensitivity'], template=get_plotly_template(), height=450)
            st.plotly_chart(fig_comp, use_container_width=True)
            csv_report = df_report.to_csv(index=False)
            st.download_button(label=t["download_csv"], data=csv_report, file_name="sensitivity_report.csv", mime="text/csv")

# ============================================================
# Sidebar Extra
# ============================================================
with st.sidebar:
    st.divider()
    if st.button(t["cache_clear"], use_container_width=True):
        st.cache_data.clear()
        st.success(t["cache_cleared"])
    if st.button(t["reset_title"], use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ============================================================
# CSV Upload
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
                    results.append({'ΔP': round(dp_patient, 2), 'Jv': round(Jv_p, 2), 'Jnet': round(Jnet_p, 2), 'Status': t["upload_status"] if Jnet_p > 0 else t["upload_compensated"]})
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                csv_results = df_results.to_csv(index=False)
                st.download_button(label=t["upload_download"], data=csv_results, file_name="patient_results.csv", mime="text/csv")
        except Exception as e:
            st.error(t["upload_error"].format(e=e))

# ============================================================
# 3D Plot
# ============================================================
with st.expander(t["3d_title"], expanded=False):
    st.info(t["3d_info"])
    col1, col2 = st.columns(2)
    with col1:
        param_3d_1 = st.selectbox(t["3d_param1"], ["Kf₀", "ΔP", "σ"], key="3d_p1")
    with col2:
        param_3d_2 = st.selectbox(t["3d_param2"], ["Kf₀", "ΔP", "σ"], key="3d_p2", index=1)
    if st.button(t["3d_plot"], key="btn_3d"):
        param_ranges = {"Kf₀": (1.0, 8.0), "ΔP": (2.0, 20.0), "σ": (0.1, 0.4)}
        p1_min, p1_max = param_ranges[param_3d_1]
        p2_min, p2_max = param_ranges[param_3d_2]
        n_points = 25
        x_vals = np.linspace(p1_min, p1_max, n_points)
        y_vals = np.linspace(p2_min, p2_max, n_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.zeros_like(X)
        for i in range(n_points):
            for j in range(n_points):
                p1_val, p2_val = X[i, j], Y[i, j]
                temp_Kf0, temp_dp, temp_sigma = Kf0, deltaP_analysis, sigma
                if param_3d_1 == "Kf₀":
                    temp_Kf0 = p1_val
                elif param_3d_1 == "ΔP":
                    temp_dp = p1_val
                elif param_3d_1 == "σ":
                    temp_sigma = p1_val
                if param_3d_2 == "Kf₀":
                    temp_Kf0 = p2_val
                elif param_3d_2 == "ΔP":
                    temp_dp = p2_val
                elif param_3d_2 == "σ":
                    temp_sigma = p2_val
                Kf_temp = calc_Kf_nonlinear(temp_Kf0, temp_dp)
                Pi_temp = calc_Pi_nonlinear(Pi0, temp_dp)
                Jv_temp = calc_Jv(temp_dp, Kf_temp, alpha, temp_sigma, Pi_temp, dPi, P_hep)
                Jlymph_temp = calc_Jlymph(Jmax, Km, Pi_temp)
                Z[i, j] = calc_Jnet(Jv_temp, Jlymph_temp)
        fig_3d = go.Figure(data=[go.Surface(z=Z, x=x_vals, y=y_vals, colorscale='Viridis')])
        fig_3d.update_layout(title=f"3D: {param_3d_1} & {param_3d_2} on Jnet", scene=dict(xaxis_title=param_3d_1, yaxis_title=param_3d_2, zaxis_title="Jnet (ml/min)"), height=600)
        st.plotly_chart(fig_3d, use_container_width=True)

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(t["footer"])
st.caption("Ver:5.0.0 (Dynamic + Saturation)")
st.caption("Ali Hosseini; ali.hosseini1387@icloud.com")
