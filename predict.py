import joblib
import os
import numpy as np
from scipy.signal import firwin, lfilter, resample, stft
from scipy.interpolate import interp1d
from typing import List, Dict, Any
from skimage.filters import threshold_otsu
from wettbewerb import get_6montages
from sklearn.preprocessing import StandardScaler
from scipy.signal import stft, find_peaks
from scipy.stats import kurtosis, skew, entropy

### Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels: List[str], data: np.ndarray, fs: float, reference_system: str, model_name: str = 'random_forest_model_gross_f1.pkl') -> Dict[str, Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models, das ihr beispielsweise bei Abgabe genannt habt. 
        Kann verwendet werden, um korrektes Modell aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        Enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

    # Initialisiere Rückgabewerte mit Standardwerten
    seizure_present = False
    seizure_confidence = 0.0
    onset = 0.0
    onset_confidence = 0.0
    offset = 0.0
    offset_confidence = 0.0

    def apply_fir_filter(data, fs, cutoff, filter_type='lowpass', numtaps=101):
        nyquist = 0.5 * fs
        fir_coeff = firwin(numtaps, cutoff / nyquist, pass_zero=(filter_type == 'lowpass'))
        return np.array([lfilter(fir_coeff, [1.0], channel) for channel in data])

    def resample_eeg_data(data, original_fs, target_fs):
        num_samples = int(data.shape[1] * target_fs / original_fs)
        return resample(data, num_samples, axis=1)

    def replace_nan(data):
        clean_data = data.copy()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(clean_data[i, j, :]).any():
                    valid_mask = ~np.isnan(clean_data[i, j, :])
                    if valid_mask.sum() > 1:
                        interp_func = interp1d(np.where(valid_mask)[0], clean_data[i, j, valid_mask], bounds_error=False, fill_value="extrapolate")
                        clean_data[i, j, :] = interp_func(np.arange(clean_data.shape[2]))
                    else:
                        clean_data[i, j, :] = np.nanmean(clean_data[i, j, :])
        return clean_data

    def z_score_normalize(data):
        mean = np.mean(data, axis=2, keepdims=True)
        std = np.std(data, axis=2, keepdims=True)
        std[std == 0] = 1
        return (data - mean) / std

    def process_eeg_data_with_sliding_window(data, fs, window_size, overlap=0.5):
    """
    Teilt EEG-Daten in überlappende Fenster auf. Fügt Zero-Padding hinzu, wenn die Daten zu kurz sind.

    Parameters
    ----------
    data : np.ndarray
        Das EEG-Datenarray mit der Form (Kanäle, Zeitpunkte).
    fs : float
        Sampling-Frequenz in Hz.
    window_size : float
        Größe des Fensters in Sekunden.
    overlap : float, optional
        Überlappungsrate zwischen den Fenstern (zwischen 0 und 1).

    Returns
    -------
    np.ndarray
        Array der Fenster mit der Form (Anzahl Fenster, Kanäle, Zeitpunkte).
    """
    # Berechnung der Fenstergröße in Datenpunkten
    samples_per_window = int(fs * window_size)
    step_size = int(samples_per_window * (1 - overlap))

    # Berechnung der minimal benötigten Länge der Daten
    num_windows = max(1, (data.shape[1] - samples_per_window) // step_size + 1)
    required_length = (num_windows - 1) * step_size + samples_per_window

    # Zero-Padding hinzufügen, wenn die Daten zu kurz sind
    if data.shape[1] < required_length:
        padding_length = required_length - data.shape[1]
        data = np.pad(
            data,
            pad_width=((0, 0), (0, padding_length)),  # Zero-Padding nur entlang der Zeitachse
            mode="constant",
            constant_values=0
        )
        print(f"Zero-Padding hinzugefügt: {padding_length} zusätzliche Werte.")

    # Sliding-Window-Segmentierung
    return np.array([
        data[:, i * step_size:i * step_size + samples_per_window]
        for i in range(num_windows)
    ])

    # Feature Berechnung
    def robust_kurtosis(channel):
        # Entferne NaN- und Inf-Werte
        channel = channel[np.isfinite(channel)]

        # Prüfe, ob das Signal leer oder konstant ist
        if len(channel) == 0 or np.var(channel) == 0:
            return 0  # Setze Kurtosis auf 0 für konstante oder ungültige Signale

        # Berechne die Kurtosis
        return kurtosis(channel, fisher=False)

    # Funktion zur Berechnung der Bandpower und des Otsu-Schwellenwerts direkt während der STFT
    def compute_features(window, fs, bands):
        n_channels = window.shape[0]
        band_powers = []
        all_flattened_values = []
        kurtosis_values = []
        peak_counts = []
        p2p = []
        skew_values = []
        entropy_values = []
        spectral_skew_values = []
        spectral_entropy_values = []
        spectral_kurtosis_values = []
        bandwidths = []


        for channel in window:
            #kurtosis
            channel_kurtosis = robust_kurtosis(channel)
            kurtosis_values.append(channel_kurtosis)

            #p2p
            p2p_value = np.ptp(channel)
            p2p.append(p2p_value)

            #peak count
            threshold2 = np.mean(channel) + 2 * np.std(channel)  # Dynamischer Schwellenwert
            peaks, _ = find_peaks(channel, height=threshold2)
            peak_count = len(peaks)
            peak_counts.append(peak_count)

            # Skewness (Schiefe)
            channel_skew = skew(channel, bias=False, nan_policy='omit')
            if np.isnan(channel_skew):
                channel_skew = 0
            skew_values.append(channel_skew)

            # Entropy (Entropie)
            prob_density, _ = np.histogram(channel, bins=256, density=True)
            prob_density = prob_density[prob_density > 0]  # Nullwerte entfernen
            channel_entropy = entropy(prob_density)
            entropy_values.append(channel_entropy)

            # Variance (Varianz) hatte 0 feature importance also entfernt

            # STFT der Fenster
            f, t, Zxx = stft(channel, fs=fs, nperseg=256)
            abs_Zxx = np.abs(Zxx)
            all_flattened_values.extend(abs_Zxx.flatten())

            # Spectral Skewness
            spectral_skew = skew(abs_Zxx.flatten(), bias=False, nan_policy='omit')
            if np.isnan(spectral_skew):
                spectral_skew = 0
            spectral_skew_values.append(spectral_skew)

            # Spectral Entropy
            spectral_prob_density = np.mean(abs_Zxx, axis=1) / np.sum(abs_Zxx)
            spectral_entropy = entropy(spectral_prob_density)
            spectral_entropy_values.append(spectral_entropy)

            # Spectral Kurtosis
            spectral_kurt = robust_kurtosis(abs_Zxx.flatten())
            spectral_kurtosis_values.append(spectral_kurt)

            # Bandwidth
            power_spectrum = np.sum(abs_Zxx**2, axis=1)
            centroid = np.sum(f * power_spectrum) / np.sum(power_spectrum)
            bandwidth = np.sqrt(np.sum(((f - centroid)**2) * power_spectrum) / np.sum(power_spectrum))
            bandwidths.append(bandwidth)

            for band in bands:
                band_indices = (f >= band[0]) & (f <= band[1])
                band_power = np.mean(np.abs(Zxx[band_indices])**2)
                band_powers.append(band_power)

        # Berechnung des Schwellenwerts
        threshold = threshold_otsu(np.array(all_flattened_values))
        #berechnung mean kurtosis
        mean_kurtosis = np.mean(kurtosis_values)
        # Mittelwert der Band-Power über alle Kanäle und das Threshold hinzufügen
        features = [np.mean(band_powers[i::len(bands)]) for i in range(len(bands))]#0-9
        features.append(mean_kurtosis)#10
        features.append(threshold)#11
        features.append(np.max(p2p))#12
        features.append(np.mean(peak_counts))#13
        features.append(np.mean(skew_values))#14
        features.append(np.mean(entropy_values))#15
        features.append(np.mean(spectral_skew_values))#16
        features.append(np.mean(spectral_entropy_values))#17
        features.append(np.mean(spectral_kurtosis_values))#18
        features.append(np.mean(bandwidths))#19

        return features


    # Modell und Scaler laden
    try:
        saved_objects = joblib.load(model_name)
        model = saved_objects["model"]
        scaler = saved_objects["scaler"]
    except Exception as e:
        raise ValueError(f"Fehler beim Laden von Modell und Scaler: {e}")

    # Datenvorverarbeitung
    try:
        window_size = 2
        overlap = 0.5
        target_fs = 250

        eeg_data = apply_fir_filter(data, fs, cutoff=0.5, filter_type='highpass')
        eeg_data = apply_fir_filter(eeg_data, fs, cutoff=50, filter_type='lowpass')

        if fs != target_fs:
            eeg_data = resample_eeg_data(eeg_data, fs, target_fs)
            fs = target_fs

        montages, montage_data, montage_missing = get_6montages(channels, eeg_data)

        if montage_missing:
            print("Warnung: Fehlende Kanäle, Montagen unvollständig.")

        montages = process_eeg_data_with_sliding_window(montage_data, fs, window_size, overlap)
        if montages is None:
            print("Keine gültigen Montagen gefunden.")
            return {
                "seizure_present": seizure_present,
                "seizure_confidence": seizure_confidence,
                "onset": onset,
                "onset_confidence": onset_confidence,
                "offset": offset,
                "offset_confidence": offset_confidence
            }
        montages = replace_nan(montages)
        montages = z_score_normalize(montages)

        bands = [(0, 3), (3, 7), (7, 9), (9, 13), (13, 17), (17, 21), (21, 23), (23, 31), (31, 35), (35, 42)]
        features = [compute_features(window, fs, bands) for window in montages]
        features = np.array(features).reshape(len(features), -1)

        if features.size == 0:
            raise ValueError("Keine Features für die Vorhersage verfügbar!")

        features_scaled = scaler.transform(features)

        # NaN-Werte durch 0 ersetzen
        features_scaled = np.nan_to_num(features_scaled)

    except Exception as e:
        raise ValueError(f"Fehler bei der Datenvorverarbeitung: {e}")

    # Vorhersage
    try:
        print("Beginne Vorhersage...")
        print(f"Shape der Features: {features_scaled.shape}")

        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[:, 1]

        print(f"Predictions: {predictions}")
        print(f"Probabilities: {probabilities}")

        seizure_present = bool(np.any(predictions))
        seizure_confidence = float(np.max(probabilities)) if seizure_present else 0.0
        seizure_windows = np.where(predictions == 1)[0]

        print(f"Seizure windows: {seizure_windows}")

        if seizure_present and seizure_windows.size > 0:
            # Berechnung der Zeitstempel für jedes erkannte Fenster
            time_stamps = seizure_windows * window_size * (1 - overlap)

            # Clusterbildung basierend auf max_gap_seconds
            max_gap_seconds = 5  # Maximale Zeitdifferenz, um Cluster zu definieren
            clusters = []
            current_cluster = [time_stamps[0]]

            for i in range(1, len(time_stamps)):
                if time_stamps[i] - time_stamps[i - 1] <= max_gap_seconds:
                    # Fenster gehört zum aktuellen Cluster
                    current_cluster.append(time_stamps[i])
                else:
                    # Neues Cluster starten
                    clusters.append(current_cluster)
                    current_cluster = [time_stamps[i]]

            # Letztes Cluster hinzufügen
            clusters.append(current_cluster)

            # Wähle den ersten Cluster
            first_cluster = clusters[0]

            # Berechne Onset und Offset basierend auf dem ersten Cluster
            onset = first_cluster[0]
            offset = first_cluster[-1] + window_size * (1 - overlap)

            # Wahrscheinlichkeiten für Onset und Offset
            onset_confidence = float(probabilities[seizure_windows[0]])
            offset_confidence = float(probabilities[seizure_windows[-1]])
        else:
            print("Kein Anfall erkannt.")

    except Exception as e:
        raise ValueError(f"Fehler bei der Modellvorhersage: {e}")

    prediction = {
        "seizure_present": seizure_present,
        "seizure_confidence": seizure_confidence,
        "onset": onset,
        "onset_confidence": onset_confidence,
        "offset": offset,
        "offset_confidence": offset_confidence
    }
    return prediction
