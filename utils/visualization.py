import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import io
from PIL import Image

def plot_voice_characteristics(pitch_values, intensity_values, timestamps=None):
    """
    Создает график характеристик голоса
    """
    if not pitch_values or not intensity_values:
        return None
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    x = range(len(pitch_values)) if timestamps is None else timestamps
    
    # График высоты голоса
    ax1.plot(x, pitch_values, 'b-')
    ax1.set_title('Высота голоса')
    ax1.set_ylabel('Частота (Гц)')
    
    # График интенсивности
    ax2.plot(x, intensity_values, 'r-')
    ax2.set_title('Интенсивность голоса')
    ax2.set_ylabel('дБ')
    
    plt.tight_layout()
    return fig

def plot_multimodal_timeline(results):
    """
    Создает временную шкалу с отмеченными событиями
    """
    if not results:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Извлечение временных меток и индикаторов
    timestamps = [r['timestamp'] for r in results]
    
    # Нормализация временных меток для графика
    start_time = min(timestamps)
    normalized_times = [t - start_time for t in timestamps]
    
    # Значения индикаторов
    stress_indicators = [1 if r.get('combined_stress_indicator', False) else 0 for r in results]
    
    # Построение графика
    ax.scatter(normalized_times, stress_indicators, c='red', marker='o', s=50, alpha=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Норма', 'Стресс'])
    ax.set_xlabel('Время (сек)')
    ax.set_title('Временная шкала индикаторов стресса')
    
    # Добавление горизонтальных линий для удобства
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.2)
    ax.axhline(y=1, color='red', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    return fig

def create_distortion_summary(recent_results):
    """
    Создает сводку по искажениям для отображения в интерфейсе
    """
    if not recent_results:
        return "Данных для анализа недостаточно."
    
    # Подсчет количества искажений по каждой области
    face_regions_count = {}
    audio_changes_count = 0
    combined_stress_count = 0
    
    for result in recent_results:
        if result.get('combined_stress_indicator', False):
            combined_stress_count += 1
            
        if 'video_distortions' in result:
            for region, is_distorted in result['video_distortions'].items():
                if is_distorted:
                    face_regions_count[region] = face_regions_count.get(region, 0) + 1
        
        if 'audio_data' in result and result['audio_data'] and result['audio_data'].get('voice_change', False):
            audio_changes_count += 1
    
    # Форматирование результата
    summary = []
    summary.append(f"Общее количество индикаторов стресса: {combined_stress_count}")
    
    if face_regions_count:
        summary.append("\nИскажения по областям лица и позы:")
        for region, count in face_regions_count.items():
            summary.append(f"- {region}: {count} раз")
    
    summary.append(f"\nИзменения голосовых характеристик: {audio_changes_count} раз")
    
    return "\n".join(summary) 