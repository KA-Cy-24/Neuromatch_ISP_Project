# config.py
"""
فایل تنظیمات برای پروژه Motor Imagery Analysis
"""

# =================================================
# تنظیمات داده
# =================================================
DATA_CONFIG = {
    'data_path': r"C:\Users\Asus\Downloads\MOTOR Imagery\motor_imagery.npz",
    'default_subject': 0,
}

# =================================================
# تنظیمات فیلتر
# =================================================
FILTER_CONFIG = {
    'lowcut': 1,
    'highcut': 120,
    'order': 4
}

# =================================================
# تنظیمات Epoch
# =================================================
EPOCH_CONFIG = {
    'baseline_ms': 500,
    'epoch_ms': 3000,
    'post_trial_ms': 1000,
    'stimulus_codes': {
        'hand': 12,
        'tongue': 11
    }
}

# =================================================
# تنظیمات TFR
# =================================================
TFR_CONFIG = {
    'freq_min': 5,
    'freq_max': 80,
    'freq_step': 1,
    'n_cycles_min': 2,
    'n_cycles_max': 6,
    'output': 'power',
    'decim': 1,
    'n_jobs': 4
}

# =================================================
# تنظیمات Baseline Correction
# =================================================
BASELINE_CONFIG = {
    'method': 'db',  # 'db', 'percent', 'zscore'
}

# =================================================
# تنظیمات باندهای فرکانسی
# =================================================
FREQUENCY_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80),
    'low_beta': (13, 20),
    'high_beta': (20, 30)
}

# =================================================
# تنظیمات نمایش
# =================================================
VISUALIZATION_CONFIG = {
    'figsize_per_electrode': 2.5,
    'max_electrodes_display': 20,
    'colormap': 'RdBu_r',
    'dpi': 300,
    'save_plots': True,
    'plot_format': 'png'
}

# =================================================
# تنظیمات کیفیت داده
# =================================================
QUALITY_CONFIG = {
    'rejection_threshold_multiplier': 5,  # چند برابر انحراف معیار
    'baseline_stability_threshold': 2,    # برای بررسی ثبات baseline
}

# =================================================
# مسیرهای ذخیره
# =================================================
OUTPUT_PATHS = {
    'plots_dir': './plots/',
    'results_dir': './results/',
    'logs_dir': './logs/'
}

# =================================================
# تنظیمات پیشرفته
# =================================================
ADVANCED_CONFIG = {
    'parallel_processing': True,
    'memory_mapping': False,
    'verbose_level': 1,  # 0: خاموش, 1: عادی, 2: جزئیات
}

# =================================================
# توابع کمکی برای کانفیگ
# =================================================

def get_frequency_array():
    """بازگرداندن آرایه فرکانس‌ها بر اساس تنظیمات"""
    import numpy as np
    return np.arange(
        TFR_CONFIG['freq_min'], 
        TFR_CONFIG['freq_max'], 
        TFR_CONFIG['freq_step']
    )

def get_n_cycles():
    """بازگرداندن آرایه n_cycles بر اساس تنظیمات"""
    import numpy as np
    freqs = get_frequency_array()
    return np.linspace(
        TFR_CONFIG['n_cycles_min'], 
        TFR_CONFIG['n_cycles_max'], 
        len(freqs)
    )

def create_output_directories():
    """ایجاد پوشه‌های خروجی"""
    import os
    for path in OUTPUT_PATHS.values():
        os.makedirs(path, exist_ok=True)
        print(f"📁 Directory created/verified: {path}")

def validate_config():
    """اعتبارسنجی تنظیمات"""
    errors = []
    
    # بررسی فرکانس‌ها
    if TFR_CONFIG['freq_min'] >= TFR_CONFIG['freq_max']:
        errors.append("freq_min باید کمتر از freq_max باشد")
    
    # بررسی baseline
    if EPOCH_CONFIG['baseline_ms'] < 0:
        errors.append("baseline_ms نمی‌تواند منفی باشد")
    
    # بررسی کدهای محرک
    stim_codes = list(EPOCH_CONFIG['stimulus_codes'].values())
    if len(set(stim_codes)) != len(stim_codes):
        errors.append("کدهای محرک نباید تکراری باشند")
    
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ Configuration validation passed")
        return True

        
CONNECTIVITY_PARAMS = {
    'method': 'imcoh',
    'fmin': 70,
    'fmax': 100,
    'baseline_normalization': 'subtraction_baseline',
    'baseline_ms' : 500,
    'srate' : 1000
}
COMMON_ROIS = ['Brodmann area 4', 'Brodmann area 6', 'Brodmann area 2', 'Brodmann area 9', 'Brodmann area 43', 'Brodmann area 45', 'Brodmann area 46']


IMPORTANT_CONNECTIONS = [
    (0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
    (2, 3), (0, 4), (1, 4), (2, 4)
]

def print_config_summary():
    """نمایش خلاصه تنظیمات"""
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"📊 Data: {DATA_CONFIG['data_path']}")
    print(f"🔧 Filter: {FILTER_CONFIG['lowcut']}-{FILTER_CONFIG['highcut']} Hz")
    print(f"📦 Epochs: {EPOCH_CONFIG['baseline_ms']}ms baseline + {EPOCH_CONFIG['epoch_ms']}ms")
    print(f"🌊 TFR: {TFR_CONFIG['freq_min']}-{TFR_CONFIG['freq_max']} Hz")
    print(f"📏 Baseline: {BASELINE_CONFIG['method']} method")
    print(f"🎨 Visualization: {VISUALIZATION_CONFIG['max_electrodes_display']} max electrodes")
    
    print("="*60)

if __name__ == "__main__":
    # اعتبارسنجی و نمایش تنظیمات
    validate_config()
    print_config_summary()
    create_output_directories()