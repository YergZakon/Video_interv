import threading
import numpy as np
import whisper
import librosa
import parselmouth
import torch
from parselmouth.praat import call

class AudioAnalyzer:
    def __init__(self, sample_rate=16000, model_size="large", language="ru"):
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = whisper.load_model(model_size, device=self.device)
        self.language = language
        self.buffer = []
        self.lock = threading.Lock()
        self.baseline_pitch = None
        self.baseline_intensity = None
        self.latest_results = None
        
    def process_audio(self, audio_chunk):
        with self.lock:
            self.buffer.extend(audio_chunk)
            
            # Если накопили достаточно данных для анализа
            if len(self.buffer) >= self.sample_rate * 3:  # 3 секунды аудио
                audio_data = np.array(self.buffer[:self.sample_rate * 3])
                self.buffer = self.buffer[self.sample_rate:]
                
                # Анализ аудио
                self.latest_results = self._analyze_audio(audio_data)
                return self.latest_results
        return None
        
    def _analyze_audio(self, audio_data):
        results = {}
        
        # Распознавание речи
        audio_data_float = audio_data.astype(np.float32) / 32768.0
        
        try:
            result = self.model.transcribe(audio_data_float, language=self.language)
            results['text'] = result["text"]
        except Exception as e:
            print(f"Error transcribing speech: {e}")
            results['text'] = ""
        
        # Анализ голосовых характеристик с помощью Parselmouth
        try:
            sound = parselmouth.Sound(audio_data_float, self.sample_rate)
            
            # Извлечение высоты голоса
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
            results['pitch'] = mean_pitch
            
            # Извлечение интенсивности
            intensity = call(sound, "To Intensity", 100, 0, "yes")
            mean_intensity = call(intensity, "Get mean", 0, 0)
            results['intensity'] = mean_intensity
            
            # Установка базовых значений при первом запуске
            if self.baseline_pitch is None:
                self.baseline_pitch = mean_pitch
                self.baseline_intensity = mean_intensity
            
            # Определение отклонений
            pitch_deviation = abs(mean_pitch - self.baseline_pitch) / self.baseline_pitch if self.baseline_pitch else 0
            intensity_deviation = abs(mean_intensity - self.baseline_intensity) / self.baseline_intensity if self.baseline_intensity else 0
            
            results['pitch_deviation'] = pitch_deviation
            results['intensity_deviation'] = intensity_deviation
            results['voice_change'] = pitch_deviation > 0.15 or intensity_deviation > 0.2
            
        except Exception as e:
            print(f"Error analyzing voice: {e}")
            results['pitch'] = 0
            results['intensity'] = 0
            results['pitch_deviation'] = 0
            results['intensity_deviation'] = 0
            results['voice_change'] = False
            
        return results
        
    def get_latest_results(self):
        return self.latest_results
        
    def reset_baseline(self):
        self.baseline_pitch = None
        self.baseline_intensity = None
        with self.lock:
            self.buffer = []
            self.latest_results = None 