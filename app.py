import streamlit as st
import cv2
import threading
import time
import numpy as np
import pyaudio
import queue
import os
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

from modules.video_analyzer import VideoAnalyzer
from modules.audio_analyzer import AudioAnalyzer
from modules import MultimodalIntegrator
from modules.data_manager import QuestionManager
from utils.visualization import plot_voice_characteristics, plot_multimodal_timeline, create_distortion_summary

# Глобальные переменные
frame_queue = queue.Queue(maxsize=5)
audio_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()
session_active = False

# Функция для захвата видео
def video_capture_thread(video_analyzer, integrator):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            # Анализ кадра
            processed_frame = video_analyzer.process_frame(frame)
            
            if not frame_queue.full():
                frame_queue.put(processed_frame)
                
            timestamp = time.time()
            # Добавление результата в интегратор
            video_result = {'frame': processed_frame}
            integrator.add_video_result(video_result, timestamp)
            
            time.sleep(0.03)  # ~30 FPS
    
    cap.release()

# Функция для захвата аудио
def audio_capture_thread(audio_analyzer, integrator):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    while not stop_event.is_set():
        try:
            audio_data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
            
            if not audio_queue.full():
                audio_queue.put(audio_chunk)
            
            # Анализ аудио
            audio_result = audio_analyzer.process_audio(audio_chunk)
            if audio_result:
                timestamp = time.time()
                integrator.add_audio_result(audio_result, timestamp)
        except Exception as e:
            print(f"Ошибка при захвате аудио: {e}")
            
    stream.stop_stream()
    stream.close()
    p.terminate()

# Функция для мультимодального анализа
def multimodal_analysis_thread(integrator):
    while not stop_event.is_set():
        current_time = time.time()
        result = integrator.analyze_multimodal(current_time)
        time.sleep(0.5)  # Анализируем каждые 0.5 секунд

# Инициализация анализаторов
@st.cache_resource
def init_analyzers():
    # Загрузка конфигурации
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    audio_config = config.get("audio", {})
    whisper_model = audio_config.get("whisper_model", "tiny")
    whisper_language = audio_config.get("whisper_language", "ru")

    video_analyzer = VideoAnalyzer()
    audio_analyzer = AudioAnalyzer(model_size=whisper_model, language=whisper_language)  # Передаем параметры модели и языка
    integrator = MultimodalIntegrator(video_analyzer, audio_analyzer)
    
    # Создаем папку для вопросов, если она не существует
    data_dir = Path("data")
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    
    questions_path = data_dir / "questions.csv"
    if not questions_path.exists():
        # Если файл с вопросами не существует, создаем базовый пример
        import pandas as pd
        questions = pd.DataFrame({
            'id': range(1, 4),
            'text': [
                "Расскажите о вашем опыте работы в команде",
                "Какие технологии и инструменты вы использовали в последнем проекте?",
                "Как вы решаете конфликтные ситуации?"
            ],
            'category': ['общие', 'технические', 'личностные']
        })
        questions.to_csv(questions_path, index=False)
    
    question_manager = QuestionManager(str(questions_path))
    
    return video_analyzer, audio_analyzer, integrator, question_manager

# Функция запуска сессии
def start_session(video_analyzer, audio_analyzer, integrator):
    global stop_event, session_active
    
    if stop_event.is_set():
        stop_event.clear()
    
    # Сброс базовых значений
    video_analyzer.reset_baseline()
    audio_analyzer.reset_baseline()
    integrator.clear_history()
    
    # Запуск потоков захвата и анализа
    threading.Thread(target=video_capture_thread, args=(video_analyzer, integrator), daemon=True).start()
    threading.Thread(target=audio_capture_thread, args=(audio_analyzer, integrator), daemon=True).start()
    threading.Thread(target=multimodal_analysis_thread, args=(integrator,), daemon=True).start()
    
    session_active = True
    st.session_state.session_active = True

# Функция остановки сессии
def stop_session():
    global stop_event, session_active
    stop_event.set()
    session_active = False
    st.session_state.session_active = False

# Основной интерфейс Streamlit
def main():
    st.set_page_config(
        page_title="Мультимодальный анализ ответов",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Мультимодальный анализ ответов")
    
    # Инициализация анализаторов
    video_analyzer, audio_analyzer, integrator, question_manager = init_analyzers()
    
    # Боковая панель с управлением
    with st.sidebar:
        st.header("Управление")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Начать сессию", key="start_session", 
                        disabled=st.session_state.get('session_active', False)):
                start_session(video_analyzer, audio_analyzer, integrator)
        
        with col2:
            if st.button("Завершить сессию", key="stop_session", 
                        disabled=not st.session_state.get('session_active', False)):
                stop_session()
        
        if st.button("Следующий вопрос", key="next_question",
                    disabled=not st.session_state.get('session_active', False)):
            question = question_manager.next_question()
            if question is None:
                st.warning("Вопросы закончились. Сессия завершена.")
                stop_session()
        
        st.header("Текущий вопрос")
        current_question = question_manager.get_current_question()
        if current_question is not None:
            st.markdown(f"**{current_question['text']}**")
            
            # Отображение прогресса
            current, total, percent = question_manager.get_progress()
            st.progress(percent)
            st.text(f"Вопрос {current} из {total}")
    
    # Основная область с анализом
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Видео анализ")
        video_placeholder = st.empty()
    
    with col2:
        st.header("Результаты анализа")
        
        # Создаем контейнеры для разных типов результатов
        speech_container = st.container()
        with speech_container:
            st.subheader("Распознанная речь")
            speech_placeholder = st.empty()
        
        indicators_container = st.container()
        with indicators_container:
            st.subheader("Индикаторы")
            indicators_placeholder = st.empty()
    
    # Раздел с графиками
    st.header("Визуализация данных")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Характеристики голоса")
        voice_chart_placeholder = st.empty()
    
    with chart_col2:
        st.subheader("Временная шкала индикаторов")
        timeline_chart_placeholder = st.empty()
    
    # Отображение сводки
    st.header("Сводный анализ")
    summary_placeholder = st.empty()
    
    # Запускаем обновление UI
    update_ui(
        video_analyzer, audio_analyzer, integrator,
        video_placeholder, speech_placeholder, indicators_placeholder,
        voice_chart_placeholder, timeline_chart_placeholder, summary_placeholder
    )

def update_ui(
    video_analyzer, audio_analyzer, integrator,
    video_placeholder, speech_placeholder, indicators_placeholder,
    voice_chart_placeholder, timeline_chart_placeholder, summary_placeholder
):
    """Функция обновления UI"""
    
    pitch_values = []
    intensity_values = []
    timestamps = []
    last_speech_text = ""
    summary_counter = 0  # Счетчик для создания уникальных ключей
    
    while True:
        try:
            # Выход из цикла, если сессия не активна
            if not st.session_state.get('session_active', False):
                time.sleep(0.1)
                continue
                
            # Обновление видео
            if not frame_queue.empty():
                frame = frame_queue.get()
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            # Получение последних аудио-результатов
            audio_results = audio_analyzer.get_latest_results()
            if audio_results and 'text' in audio_results and audio_results['text']:
                last_speech_text = audio_results['text']
                speech_placeholder.markdown(f"*{last_speech_text}*")
                
                # Добавление данных для графиков
                if 'pitch' in audio_results and 'intensity' in audio_results:
                    pitch_values.append(audio_results['pitch'])
                    intensity_values.append(audio_results['intensity'])
                    timestamps.append(time.time())
                    
                    # Ограничение данных для графика
                    max_points = 50
                    if len(pitch_values) > max_points:
                        pitch_values = pitch_values[-max_points:]
                        intensity_values = intensity_values[-max_points:]
                        timestamps = timestamps[-max_points:]
            
            # Получение данных искажений
            distortions = video_analyzer.get_distortions()
            if distortions:
                # Создание текста с индикаторами
                indicators_text = []
                
                # Проверка лицевых областей
                for region, is_distorted in distortions.items():
                    status = "🔴" if is_distorted else "🟢"
                    indicators_text.append(f"{status} {region}")
                
                # Проверка голосовых изменений
                if audio_results and 'voice_change' in audio_results:
                    voice_status = "🔴" if audio_results['voice_change'] else "🟢"
                    indicators_text.append(f"{voice_status} голос")
                
                # Отображение в интерфейсе
                indicators_placeholder.markdown("<br>".join(indicators_text), unsafe_allow_html=True)
            
            # Обновление графиков
            if pitch_values and intensity_values:
                voice_fig = plot_voice_characteristics(pitch_values, intensity_values)
                if voice_fig:
                    voice_chart_placeholder.pyplot(voice_fig)
            
            # Обновление временной шкалы
            recent_results = integrator.get_recent_results()
            if recent_results:
                timeline_fig = plot_multimodal_timeline(recent_results)
                if timeline_fig:
                    timeline_chart_placeholder.pyplot(timeline_fig)
                
                # Обновление сводки
                summary_text = create_distortion_summary(recent_results)
                summary_counter += 1  # Увеличиваем счетчик для создания уникального ключа
                summary_placeholder.text_area("Результаты анализа:", summary_text, height=200, key=f"summary_{summary_counter}")
            
            # Задержка для снижения нагрузки на CPU
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Ошибка при обновлении интерфейса: {e}")
            time.sleep(1)

if __name__ == "__main__":
    # Инициализация состояния сессии
    if 'session_active' not in st.session_state:
        st.session_state.session_active = False
    
    main() 