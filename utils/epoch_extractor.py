# epoch_extractor.py
import numpy as np

class EpochExtractor:
    """کلاس استخراج Epochs از داده‌های پیوسته"""
    
    def __init__(self, srate, t_on, stim_id):
        self.srate = srate
        self.t_on = t_on
        self.stim_id = stim_id
    
    def extract_epochs_with_baseline(self, data, stim_code, baseline_ms=0, 
                                   epoch_ms=3000, post_trial_ms=0):
        """
        استخراج epochs همراه با دوره baseline
        
        Parameters:
        -----------
        data : array, shape (channels, time)
        stim_code : int, کد محرک
        baseline_ms : int, مدت baseline قبل از محرک
        epoch_ms : int, مدت بعد از محرک
        post_trial_ms : int, مدت اضافی بعد از epoch
        """
        idx = np.where(self.stim_id == stim_code)[0]
        baseline_samples = int(baseline_ms * self.srate / 1000)
        epoch_samples = int(epoch_ms * self.srate / 1000)
        post_samples = int(post_trial_ms * self.srate / 1000)
        
        epochs = []
        valid_trials = 0
        
        for i in idx:
            start = self.t_on[i] - baseline_samples
            stop = self.t_on[i] + epoch_samples + post_samples
            
            epochs.append(data[:, start:stop])
            valid_trials += 1
        
        epochs = np.array(epochs)
        
        print(f"📦 Epochs with baseline extracted for stimulus {stim_code}:")
        print(f"   - Valid trials: {valid_trials}/{len(idx)}")
        print(f"   - Baseline: {baseline_ms} ms ({baseline_samples} samples)")
        print(f"   - Epoch: {epoch_ms} ms ({epoch_samples} samples)")
        print(f"   - Post-trial: {post_trial_ms} ms ({post_samples} samples)")
        print(f"   - Total shape: {epochs.shape}")
        
        return epochs, baseline_samples

    def extract_epochs_simple(self, data, stim_code, epoch_start_ms=0, epoch_end_ms=3000):
        """
        استخراج epochs ساده بدون baseline
        
        Parameters:
        -----------
        data : array, shape (channels, time)
        stim_code : int, کد محرک
        epoch_start_ms : int, شروع epoch نسبت به محرک (ms)
        epoch_end_ms : int, پایان epoch نسبت به محرک (ms)
        """
        idx = np.where(self.stim_id == stim_code)[0]
        start_samples = int(epoch_start_ms * self.srate / 1000)
        end_samples = int(epoch_end_ms * self.srate / 1000)
        
        epochs = []
        valid_trials = 0
        
        for i in idx:
            start = self.t_on[i] + start_samples
            stop = self.t_on[i] + end_samples
            
            epochs.append(data[:, start:stop])
            valid_trials += 1
        
        epochs = np.array(epochs)
        
        print(f"📦 Simple epochs extracted for stimulus {stim_code}:")
        print(f"   - Valid trials: {valid_trials}/{len(idx)}")
        print(f"   - Epoch duration: {epoch_start_ms} to {epoch_end_ms} ms")
        print(f"   - Shape: {epochs.shape}")
        
        return epochs

    def create_time_vector(self, epochs_shape, baseline_samples=0):
        """
        ایجاد بردار زمان برای epochs
        
        Parameters:
        -----------
        epochs_shape : tuple, شکل epochs
        baseline_samples : int, تعداد نمونه‌های baseline
        """
        n_times = epochs_shape[-1]
        time_vector = np.arange(n_times) / self.srate - (baseline_samples / self.srate)
        
        print(f"⏰ Time vector created: {time_vector[0]:.3f} to {time_vector[-1]:.3f} s")
        return time_vector

class EpochValidator:
    """کلاس اعتبارسنجی epochs"""
    
    @staticmethod
    def check_epoch_quality(epochs, reject_threshold=None):
        """بررسی کیفیت epochs و حذف outlierها"""
        n_trials, n_channels, n_times = epochs.shape
        
        if reject_threshold is None:
            # تعیین threshold بر اساس انحراف معیار
            reject_threshold = 5 * np.std(epochs)
        
        # پیدا کردن trials با آرتیفکت
        max_values = np.max(np.abs(epochs), axis=(1, 2))  # max برای هر trial
        bad_trials = np.where(max_values > reject_threshold)[0]
        
        # حذف bad trials
        clean_epochs = np.delete(epochs, bad_trials, axis=0)
        
        print(f"🔍 Epoch quality check:")
        print(f"   - Original trials: {n_trials}")
        print(f"   - Rejected trials: {len(bad_trials)}")
        print(f"   - Clean trials: {clean_epochs.shape[0]}")
        print(f"   - Rejection threshold: {reject_threshold:.2f}")
        
        return clean_epochs, bad_trials
    
    @staticmethod
    def check_baseline_stability(epochs, baseline_samples):
        """بررسی ثبات دوره baseline"""
        baseline_data = epochs[:, :, :baseline_samples]
        
        # محاسبه واریانس baseline برای هر trial
        baseline_var = np.var(baseline_data, axis=2)  # variance across time
        mean_var = np.mean(baseline_var, axis=1)  # mean across channels
        
        # پیدا کردن trials با baseline ناپایدار
        threshold = np.mean(mean_var) + 2 * np.std(mean_var)
        unstable_trials = np.where(mean_var > threshold)[0]
        
        print(f"📊 Baseline stability check:")
        print(f"   - Stable trials: {len(epochs) - len(unstable_trials)}")
        print(f"   - Unstable trials: {len(unstable_trials)}")
        
        return unstable_trials

