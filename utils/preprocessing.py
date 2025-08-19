# preprocessing.py
import numpy as np
from scipy.signal import butter, filtfilt
from epoch_extractor import EpochExtractor
from config import EPOCH_CONFIG, COMMON_ROIS
from typing import Dict, List, Optional

class MotorImageryPreprocessor:
    """کلاس پیش‌پردازش داده‌های Motor Imagery"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.alldat = None
        self.current_subject = None
        self.srate = None
        self.t_on = None
        self.stim_id = None
        self.brodmann_labels = None
        self.V = None
        self.roi_names = None
        
    def load_data(self, data_path=None):
        """بارگذاری داده از فایل npz"""
        if data_path:
            self.data_path = data_path
        
        if not self.data_path:
            raise ValueError("Data path not provided")
            
        self.alldat = np.load(self.data_path, allow_pickle=True)['dat']
        print(f"✅ Data loaded successfully. Found {len(self.alldat)} subjects.")
        return self
    
    def set_subject(self, subject_idx, task_type):
        """انتخاب سوژه و استخراج اطلاعات اولیه"""
        if self.alldat is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if subject_idx >= len(self.alldat):
            raise ValueError(f"Subject index {subject_idx} out of range. Max: {len(self.alldat)-1}")
        
        if task_type == 'imagery':
            task_idx = 0
        elif task_type == 'real':
            task_idx = 1

            
        self.current_subject = subject_idx
        sub_data = self.alldat[subject_idx][task_idx]
        
        # استخراج اطلاعات اساسی
        self.srate = sub_data['srate']
        self.t_on = sub_data['t_on']
        self.stim_id = sub_data['stim_id']
        self.brodmann_labels = sub_data['Brodmann_Area']
        self.V = sub_data['V'] * sub_data['scale_uv']
        self.V = self.V.T  # Transpose to have (channels, time)
        
        # استخراج نام‌های ROI منحصر به فرد
        self.roi_names = list(set(self.brodmann_labels))
        
        print(f"📊 Subject {subject_idx} selected:")
        print(f"   - Sampling rate: {self.srate} Hz")
        print(f"   - Channels: {self.V.shape[0]}")
        print(f"   - Time points: {self.V.shape[1]}")
        print(f"   - ROIs found: {len(self.roi_names)}")
        
        return self
    
    def preprocess_signal(self):
        """اعمال پیش‌پردازش‌های پایه روی سیگنال"""
        if self.V is None:
            raise ValueError("No data loaded. Call set_subject() first.")
        
        # حذف DC آفست
        self.V = self.V - np.mean(self.V, axis=1, keepdims=True)
        print("🔧 DC offset removed from signal.")
        
        # تغییر نوع داده
        self.V = self.V.astype(np.float32)
        print(f"🔧 Data type converted to {self.V.dtype}.")
        
        return self

    def bandpass_filter(self, lowcut, highcut, order=4):
        """اعمال فیلتر باند پس"""
        if self.V is None:
            raise ValueError("No data loaded. Call set_subject() first.")
            
        nyq = 0.5 * self.srate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        self.V = filtfilt(b, a, self.V, axis=-1)
        
        print(f"🔧 Bandpass filter applied: {lowcut}-{highcut} Hz")
        return self

    def extract_epochs_with_baseline(self, stim_code, baseline_ms=500, epoch_ms=3000, post_trial_ms=0):
        """استخراج epochs با baseline"""
        if self.V is None:
            raise ValueError("No data loaded. Call set_subject() first.")

        extractor = EpochExtractor(self.srate, self.t_on, self.stim_id)
        epochs, baseline_samples = extractor.extract_epochs_with_baseline(
            self.V, stim_code, baseline_ms, epoch_ms, post_trial_ms
        )
        return epochs, baseline_samples

    def extract_epochs_simple(self, stim_code, epoch_start_ms=0, epoch_end_ms=3000):
        """استخراج epochs ساده بدون baseline"""
        if self.V is None:
            raise ValueError("No data loaded. Call set_subject() first.")

        extractor = EpochExtractor(self.srate, self.t_on, self.stim_id)
        epochs = extractor.extract_epochs_simple(
            self.V, stim_code, epoch_start_ms, epoch_end_ms
        )
        return epochs
    
    def build_roi_channel_map(self, roi_names=None):
        """
        ساخت نگاشت ROI -> ایندکس کانال‌ها - نسخه مقاوم‌سازی شده
        """
        if self.brodmann_labels is None:
            raise ValueError("Brodmann labels not available. Call set_subject() first.")

        # لیبل‌های موجود در داده را یک بار برای همیشه تمیز کن
        cleaned_data_labels = [str(label).strip() for label in self.brodmann_labels]
        
        # اگر لیست ROI هدف داده نشده، از لیبل‌های تمیز شده منحصر به فرد استفاده کن
        if roi_names is None:
            roi_names = np.unique(cleaned_data_labels)

        roi_channel_map = {}
        
        for roi in roi_names:
            # نام ROI هدف را هم تمیز کن
            cleaned_target_roi = str(roi).strip()
            
            # مقایسه بین دو لیست تمیز شده انجام می‌شود
            indices = [i for i, label in enumerate(cleaned_data_labels) if label == cleaned_target_roi]
            
            # از نام اصلی roi به عنوان کلید دیکشنری استفاده کن
            roi_channel_map[roi] = indices
            
        return roi_channel_map