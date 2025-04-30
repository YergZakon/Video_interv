import time
import threading

class MultimodalIntegrator:
    def __init__(self, video_analyzer, audio_analyzer):
        self.video_analyzer = video_analyzer
        self.audio_analyzer = audio_analyzer
        self.video_results = []
        self.audio_results = []
        self.sync_threshold = 1.0  # секунды
        self.multimodal_results = []
        
    def add_video_result(self, result, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        self.video_results.append((timestamp, result))
        
        # Ограничиваем размер истории
        if len(self.video_results) > 100:
            self.video_results = self.video_results[-100:]
        
    def add_audio_result(self, result, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        self.audio_results.append((timestamp, result))
        
        # Ограничиваем размер истории
        if len(self.audio_results) > 100:
            self.audio_results = self.audio_results[-100:]
        
    def get_synchronized_data(self, current_time):
        video_data = self._get_closest_result(self.video_results, current_time)
        audio_data = self._get_closest_result(self.audio_results, current_time)
        
        if video_data and audio_data:
            return {
                'video': video_data[1],
                'audio': audio_data[1],
                'timestamp': current_time
            }
        return None
        
    def _get_closest_result(self, results, target_time):
        if not results:
            return None
            
        # Найти ближайший результат к целевому времени
        closest = min(results, key=lambda x: abs(x[0] - target_time))
        
        # Если результат в пределах порога синхронизации
        if abs(closest[0] - target_time) <= self.sync_threshold:
            return closest
        return None
        
    def analyze_multimodal(self, current_time):
        data = self.get_synchronized_data(current_time)
        if not data:
            return None
            
        # Извлечение данных видеоанализа
        video_distortions = self.video_analyzer.get_distortions()
        any_video_distortion = any(video_distortions.values())
        
        # Извлечение данных аудиоанализа
        audio_data = data['audio']
        audio_change = audio_data.get('voice_change', False) if audio_data else False
        
        # Комбинирование результатов
        result = {
            'combined_stress_indicator': any_video_distortion and audio_change,
            'timestamp': current_time,
            'video_distortions': video_distortions,
            'audio_data': audio_data
        }
        
        # Сохраняем результат
        self.multimodal_results.append(result)
        if len(self.multimodal_results) > 100:
            self.multimodal_results = self.multimodal_results[-100:]
            
        return result
    
    def get_recent_results(self, count=10):
        return self.multimodal_results[-count:] if self.multimodal_results else []
        
    def clear_history(self):
        self.video_results = []
        self.audio_results = []
        self.multimodal_results = [] 