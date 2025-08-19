import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from config import CONNECTIVITY_PARAMS, COMMON_ROIS

class EpochConnectivityProcessor:
    """
    یک کلاس بهبود یافته برای پردازش کامل کانکتیویتی یک Epoch
    """

    def __init__(self, epoch_data: np.ndarray, srate: int, brodmann_labels: np.ndarray, verbose=False):
        """
        سازنده کلاس.
        
        Args:
            verbose (bool): نمایش پیام‌های اضافی یا نه
        """
        self.epoch_data = epoch_data
        self.srate = srate
        self.brodmann_labels = brodmann_labels
        self.verbose = verbose
        
        # ویژگی‌هایی برای نگهداری نتایج میانی و نهایی
        self.corrected_conn_matrix = None
        self.roi_matrix = None
        self.roi_mapping = None  # جدید: نگهداری mapping کانال‌ها
        
        # بهبود 1: ایجاد mapping یک بار
        self._create_roi_mapping()

    def _create_roi_mapping(self):
        """
        بهبود 1: یک بار mapping کانال‌ها به ROI ها را بساز
        """
        cleaned_labels = np.array([str(label).strip() for label in self.brodmann_labels])
        self.roi_mapping = {}
        
        for i, roi in enumerate(COMMON_ROIS):
            roi_cleaned = roi.strip()
            indices = np.where(cleaned_labels == roi_cleaned)[0]
            self.roi_mapping[i] = {
                'name': roi_cleaned,
                'indices': indices,
                'count': len(indices)
            }
        
        if self.verbose:
            print("📋 ROI Mapping created:")
            for i, info in self.roi_mapping.items():
                print(f"   ROI {i} - {info['name']}: {info['count']} channels")

    def _compute_segment_connectivity(self, segment_data: np.ndarray) -> np.ndarray:
        """
        محاسبه کانکتیویتی یک قطعه از داده
        بهبود 2: اضافه کردن بررسی خطا
        """
        if segment_data.shape[1] < 100:  # کمتر از 100 نمونه
            if self.verbose:
                print(f"⚠️  Warning: Short segment ({segment_data.shape[1]} samples)")
        
        try:
            segment_data_reshaped = segment_data[np.newaxis, :, :]
            
            conn = spectral_connectivity_epochs(
                segment_data_reshaped, 
                method=CONNECTIVITY_PARAMS['method'], 
                sfreq=self.srate,
                fmin=CONNECTIVITY_PARAMS['fmin'], 
                fmax=CONNECTIVITY_PARAMS['fmax'], 
                faverage=True, 
                verbose=False
            )
            
            return conn.get_data(output='dense')[:, :, 0]
            
        except Exception as e:
            print(f"❌ Error in connectivity computation: {e}")
            # بازگشت ماتریس صفر در صورت خطا
            n_channels = segment_data.shape[0]
            return np.zeros((n_channels, n_channels))

    def compute_corrected_connectivity(self, method: str = 'subtraction'):
        """
        بهبود 3: اضافه کردن بررسی‌های اضافی
        """
        baseline_duration_sec = CONNECTIVITY_PARAMS['baseline_ms'] / 1000.0
        baseline_end_sample = int(baseline_duration_sec * self.srate)
        
        # بررسی طول داده
        if baseline_end_sample >= self.epoch_data.shape[1]:
            raise ValueError(f"Baseline ({baseline_end_sample}) longer than epoch ({self.epoch_data.shape[1]})")
        
        baseline_data = self.epoch_data[:, :baseline_end_sample]
        trial_data = self.epoch_data[:, baseline_end_sample:]
        
        if self.verbose:
            print(f"📊 Baseline: {baseline_data.shape[1]} samples, Trial: {trial_data.shape[1]} samples")
        
        conn_baseline = self._compute_segment_connectivity(baseline_data)
        conn_trial = self._compute_segment_connectivity(trial_data)
        
        if method == 'subtraction':
            self.corrected_conn_matrix = conn_trial - conn_baseline
        elif method == 'percent':
            denom = np.where(conn_baseline == 0, np.finfo(float).eps, conn_baseline)
            self.corrected_conn_matrix = 100.0 * (conn_trial - conn_baseline) / denom
        else:
            raise ValueError("Method must be 'subtraction' or 'percent'")
        
        return self

    def convert_to_roi_matrix(self):
        """
        بهبود 4: استفاده از mapping آماده و کاهش محاسبات
        """
        if self.corrected_conn_matrix is None:
            raise ValueError("Please run compute_corrected_connectivity() first.")
            
        n_rois = len(COMMON_ROIS)
        matrix = np.zeros((n_rois, n_rois))
        
        if self.verbose:
            print("\n--- شروع تبدیل به ماتریس ROI ---")
        
        # استفاده از mapping آماده
        for i in range(n_rois):
            idx_i = self.roi_mapping[i]['indices']
            
            for j in range(n_rois):
                idx_j = self.roi_mapping[j]['indices']
                
                if len(idx_i) > 0 and len(idx_j) > 0:
                    submatrix = self.corrected_conn_matrix[np.ix_(idx_i, idx_j)]
                    matrix[i, j] = np.mean(submatrix)
                    
                    if self.verbose and i <= 1 and j <= 1:  # فقط چند تا اول
                        roi_i = self.roi_mapping[i]['name']
                        roi_j = self.roi_mapping[j]['name']
                       
        if self.verbose:
            print("--- پایان تبدیل ---")
        
        self.roi_matrix = matrix
        return self

    def get_roi_info(self):
        """
        بهبود 5: متد جدید برای دریافت اطلاعات ROI
        """
        return self.roi_mapping.copy()

    def validate_data(self):
        """
        بهبود 6: متد جدید برای بررسی صحت داده
        """
        issues = []
        
        # بررسی تعداد کانال‌های هر ROI
        for i, info in self.roi_mapping.items():
            if info['count'] == 0:
                issues.append(f"ROI '{info['name']}' has no channels")
        
        # بررسی NaN یا Inf
        if np.any(np.isnan(self.epoch_data)) or np.any(np.isinf(self.epoch_data)):
            issues.append("Data contains NaN or Inf values")
        
        # بررسی تنوع سیگنال
        signal_std = np.std(self.epoch_data, axis=1)
        if np.any(signal_std < 1e-10):
            issues.append("Some channels have very low signal variance")
        
        return issues

    def process(self, correction_method: str = 'subtraction', validate: bool = True) -> tuple:
        if validate:
            issues = self.validate_data()
            if issues and self.verbose:
                print("⚠️  Data validation issues:")
                for issue in issues:
                    print(f"   - {issue}")
        
        self.compute_corrected_connectivity(method=correction_method)
        self.convert_to_roi_matrix()
        
        # استخراج مثلث بالایی (بدون قطر اصلی) و تبدیل به بردار
        upper_tri = np.triu(self.roi_matrix, k=1)
        self.roi_matrix = upper_tri[np.triu_indices_from(upper_tri, k=1)]
        
        # ساخت feature names (فقط یک بار)
        if not hasattr(self, '_feature_names'):
            roi_names = [f"BA{name.split()[-1]}" for name in COMMON_ROIS] 
            self._feature_names = []
            for i in range(len(roi_names)):
                for j in range(i+1, len(roi_names)):
                    self._feature_names.append(f"{roi_names[i]}-{roi_names[j]}")
        
        return self.roi_matrix, self._feature_names
    def get_summary(self):
        """
        بهبود 8: متد جدید برای خلاصه نتایج
        """
        if self.roi_matrix is None:
            return "No ROI matrix computed yet"
        
        summary = {
            'roi_matrix_shape': self.roi_matrix.shape,
            'connectivity_range': (float(np.min(self.roi_matrix)), float(np.max(self.roi_matrix))),
            'mean_connectivity': float(np.mean(self.roi_matrix)),
            'channels_per_roi': {info['name']: info['count'] for info in self.roi_mapping.values()}
        }
        return summary