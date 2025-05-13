import cv2
import numpy as np

class VideoAnalyzer:
    def __init__(self):
        # Загружаем каскады для обнаружения лица и других элементов
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Области лица (будут определяться приблизительно)
        self.face_regions = {
            'лоб': 'верх',
            'брови': 'верх-средний',
            'глаза': 'средний',
            'нос': 'средний',
            'рот': 'нижний-средний',
            'подбородок': 'низ'
        }
        
        # Базовые шаблоны для нормального состояния
        self.baseline_templates = {}
        self.baseline_positions = {}
        self.is_baseline_set = False
        self.distortion_results = {}
        self.frame_counter = 0
        self.face_detected_counter = 0
        
    def process_frame(self, frame):
        self.frame_counter += 1
        
        # Сброс результатов обнаружения искажений
        self.distortion_results = {region: False for region in self.face_regions.keys()}
        self.distortion_results['pose'] = False
        
        processed_frame = frame.copy()
        
        # Конвертируем кадр в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Обнаружение лиц
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            self.face_detected_counter += 1
            
            # Берем первое (предположительно основное) лицо
            (x, y, w, h) = faces[0]
            
            # Рисуем рамку вокруг лица
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Если это первые 10 кадров с лицом, устанавливаем базовый уровень
            if self.face_detected_counter <= 10 and not self.is_baseline_set:
                self._update_baseline(gray, x, y, w, h)
                if self.face_detected_counter == 10:
                    self.is_baseline_set = True
                    print("Базовый уровень установлен")
            
            # Анализ областей лица
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = processed_frame[y:y+h, x:x+w]
            
            # Детекция глаз
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Детекция улыбки
            smile = self.smile_cascade.detectMultiScale(face_roi_gray, 1.8, 20)
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(face_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                self.distortion_results['mouth'] = True
            
            # Проверяем изменения в лице
            if self.is_baseline_set:
                self._check_facial_changes(x, y, w, h)
            
            # Визуализируем области лица
            processed_frame = self._visualize_face_regions(processed_frame, x, y, w, h)
            
            # Если детектировали улыбку, отображаем индикатор
            if len(smile) > 0:
                cv2.putText(processed_frame, "Улыбка обнаружена", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Проверка наклона головы (изменение позиции)
            if self.is_baseline_set:
                pose_distortion = self._check_pose_distortion(x, y, w, h)
                if pose_distortion:
                    self.distortion_results['pose'] = True
                    cv2.putText(processed_frame, "Наклон головы", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return processed_frame
    
    def _update_baseline(self, gray, x, y, w, h):
        """Обновляет базовые значения для лица"""
        self.baseline_positions['face'] = (x, y, w, h)
        
        # Обрезаем область лица
        face_roi = gray[y:y+h, x:x+w]
        
        # Сохраняем информацию об областях лица
        h_face, w_face = face_roi.shape
        
        # Разделяем лицо на области и сохраняем средние значения интенсивности
        self.baseline_templates['forehead'] = np.mean(face_roi[0:int(h_face*0.2), :])
        self.baseline_templates['eyebrows'] = np.mean(face_roi[int(h_face*0.2):int(h_face*0.35), :])
        self.baseline_templates['eyes'] = np.mean(face_roi[int(h_face*0.35):int(h_face*0.5), :])
        self.baseline_templates['nose'] = np.mean(face_roi[int(h_face*0.5):int(h_face*0.65), :])
        self.baseline_templates['mouth'] = np.mean(face_roi[int(h_face*0.65):int(h_face*0.8), :])
        self.baseline_templates['chin'] = np.mean(face_roi[int(h_face*0.8):, :])
    
    def _check_facial_changes(self, x, y, w, h):
        """Проверяет изменения в областях лица"""
        baseline_x, baseline_y, baseline_w, baseline_h = self.baseline_positions['face']
        
        # Проверяем значительные изменения в размере лица
        size_change = abs(w*h - baseline_w*baseline_h) / (baseline_w*baseline_h)
        
        if size_change > 0.2:  # Если размер лица изменился более чем на 20%
            self.distortion_results['pose'] = True
    
    def _check_pose_distortion(self, x, y, w, h):
        """Проверка наклона головы"""
        if not self.is_baseline_set:
            return False
        
        baseline_x, baseline_y, baseline_w, baseline_h = self.baseline_positions['face']
        
        # Проверяем смещение лица
        x_shift = abs(x - baseline_x) / baseline_w
        y_shift = abs(y - baseline_y) / baseline_h
        
        # Если лицо существенно сдвинулось, считаем это изменением позы
        return x_shift > 0.15 or y_shift > 0.15
    
    def _visualize_face_regions(self, frame, x, y, w, h):
        """Отображает области лица с индикацией искажений"""
        # Рассчитываем приблизительные координаты областей
        regions_coords = {
            'лоб': (x, y, w, int(h * 0.2)),
            'брови': (x, y + int(h * 0.2), w, int(h * 0.15)),
            'глаза': (x, y + int(h * 0.35), w, int(h * 0.15)),
            'нос': (x + int(w * 0.25), y + int(h * 0.5), int(w * 0.5), int(h * 0.15)),
            'рот': (x + int(w * 0.25), y + int(h * 0.65), int(w * 0.5), int(h * 0.15)),
            'подбородок': (x, y + int(h * 0.8), w, int(h * 0.2))
        }
        
        # Отрисовка областей
        for region_name, (rx, ry, rw, rh) in regions_coords.items():
            is_distorted = self.distortion_results.get(region_name, False)
            color = (0, 0, 255) if is_distorted else (0, 255, 0)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)
            cv2.putText(frame, region_name, (rx, ry - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def get_distortions(self):
        """Возвращает текущие результаты анализа искажений"""
        return self.distortion_results
    
    def reset_baseline(self):
        """Сбрасывает базовые значения"""
        self.baseline_templates = {}
        self.baseline_positions = {}
        self.is_baseline_set = False
        self.face_detected_counter = 0 