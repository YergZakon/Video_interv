import streamlit as st
import logging # Added
import threading
import time
import numpy as np
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

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Default level
# Avoid adding multiple handlers if script re-runs (e.g. Streamlit auto-reload)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Attempt to import cv2 and set a flag
cv2_available = False
cv2 = None
try:
    import cv2
    cv2_available = True
    logger.info("OpenCV (cv2) imported successfully.")
except ImportError:
    logger.error("OpenCV (cv2) not found. Video analysis functions will be unavailable.", exc_info=False)
    st.error("OpenCV (cv2) не найден. Функции видеоанализа будут недоступны.")

# Attempt to import pyaudio and set a flag
pyaudio_available = False
pyaudio = None
try:
    import pyaudio
    pyaudio_available = True
    logger.info("PyAudio imported successfully.")
except ImportError:
    logger.error("PyAudio not found. Audio analysis functions will be unavailable.", exc_info=False)
    st.error("PyAudio не найден. Функции аудиоанализа будут недоступны.")

# Глобальные переменные
frame_queue = queue.Queue(maxsize=5)
audio_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()
session_active = False

# Функция для захвата видео
def video_capture_thread(video_analyzer, integrator):
    logger.info("Video capture thread initiated.")
    if not cv2_available or video_analyzer is None:
        logger.warning("Video capture thread not starting: cv2 or video_analyzer unavailable.")
        st.warning("Видеозахват недоступен, так как cv2 или video_analyzer не инициализирован.")
        return

    cap = None
    try:
        logger.info("Attempting to open webcam.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open webcam.")
            st.error("Не удалось открыть веб-камеру.")
            return
        
        logger.info("Webcam opened successfully.")
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
                if integrator:
                    integrator.add_video_result(video_result, timestamp)
                
                time.sleep(0.03)  # ~30 FPS
            else:
                logger.warning("Failed to retrieve frame from webcam. Retrying...")
                time.sleep(0.1) # Wait a bit if reading fails
    except Exception as e:
        logger.exception("Exception in video_capture_thread.")
        st.error(f"Критическая ошибка в потоке видеозахвата: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()
            logger.info("Webcam released.")
        logger.info("Video capture thread stopping.")

# Функция для захвата аудио
def audio_capture_thread(audio_analyzer, integrator):
    logger.info("Audio capture thread initiated.")
    if not pyaudio_available or audio_analyzer is None:
        logger.warning("Audio capture thread not starting: pyaudio or audio_analyzer unavailable.")
        st.warning("Аудиозахват недоступен, так как pyaudio или audio_analyzer не инициализирован.")
        return

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    p = None
    stream = None
    try:
        logger.info("Initializing PyAudio.")
        p = pyaudio.PyAudio()
        logger.info("Attempting to open PyAudio stream.")
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        logger.info("PyAudio stream opened successfully.")
        st.info("Аудиозахват начат.")
        
        while not stop_event.is_set():
            try:
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(audio_data, dtype=np.int16)
                
                if not audio_queue.full():
                    audio_queue.put(audio_chunk)
                
                if audio_analyzer: # Ensure analyzer exists
                    audio_result = audio_analyzer.process_audio(audio_chunk)
                    if audio_result and integrator: # Ensure integrator exists
                        timestamp = time.time()
                        integrator.add_audio_result(audio_result, timestamp)
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed: # type: ignore
                    logger.warning("Input audio buffer overflowed.", exc_info=False) # Traceback not very useful here
                    st.warning("Переполнение входного буфера аудио. Некоторые аудиоданные могли быть потеряны.")
                else:
                    logger.error("IOError reading from audio stream.", exc_info=True)
                    st.error(f"Ошибка чтения из аудиопотока: {e}")
                    break # Exit loop on other IOErrors
            except Exception as e:
                logger.exception("Exception processing audio chunk.")
                st.error(f"Ошибка при обработке аудио: {e}")
                time.sleep(0.1) # Avoid rapid error logging if in a tight loop
                
    except Exception as e:
        logger.exception("Critical error initializing audio capture (PyAudio setup or stream open).")
        st.error(f"Критическая ошибка инициализации аудиозахвата: {e}")
    finally:
        if stream:
            try:
                if not stream.is_stopped(): # type: ignore
                    stream.stop_stream()
                stream.close()
                logger.info("PyAudio stream stopped and closed.")
            except Exception as e:
                logger.error("Error stopping/closing audio stream.", exc_info=True)
                st.error(f"Ошибка при остановке аудиопотока: {e}")
        if p:
            try:
                p.terminate()
                logger.info("PyAudio terminated.")
            except Exception as e:
                logger.error("Error terminating PyAudio.", exc_info=True)
                st.error(f"Ошибка при закрытии PyAudio: {e}")
        st.info("Аудиозахват завершен.") # This message might be confusing if capture never started
        logger.info("Audio capture thread stopping.")


# Функция для мультимодального анализа
def multimodal_analysis_thread(integrator):
    logger.info("Multimodal analysis thread started.")
    try:
        while not stop_event.is_set():
            if integrator:
                current_time = time.time()
                # result = integrator.analyze_multimodal(current_time) # Result not currently used
                integrator.analyze_multimodal(current_time) 
            else:
                logger.warning("MultimodalIntegrator is None, skipping analysis.")
                # If integrator is critical, thread could stop. For now, it just waits.
            time.sleep(0.5)
    except Exception as e:
        logger.exception("Exception in multimodal_analysis_thread.")
        # Consider st.error if this thread's failure is critical to user experience
    finally:
        logger.info("Multimodal analysis thread stopping.")

# Инициализация анализаторов
@st.cache_resource
def init_analyzers():
    logger.info("Initializing analyzers, models, and managers...")
    config = {}
    try:
        logger.debug("Loading configuration from config.yaml.")
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("config.yaml loaded successfully.")
    except FileNotFoundError:
        logger.error("config.yaml not found. Using default configurations.")
        st.error("Файл конфигурации config.yaml не найден. Будут использованы значения по умолчанию.")
    except Exception as e:
        logger.exception("Failed to load or parse config.yaml.")
        st.error(f"Ошибка загрузки конфигурации (config.yaml): {e}. Будут использованы значения по умолчанию.")

    audio_config = config.get("audio", {})
    whisper_model = audio_config.get("whisper_model", "tiny")
    whisper_language = audio_config.get("whisper_language", "ru")
    logger.info(f"Audio config: whisper_model='{whisper_model}', language='{whisper_language}'.")

    video_analyzer = None
    if cv2_available:
        logger.info("Attempting to initialize VideoAnalyzer...")
        try:
            video_analyzer = VideoAnalyzer()
            logger.info("VideoAnalyzer initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize VideoAnalyzer.")
            st.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать VideoAnalyzer: {e}. Видеоанализ будет невозможен.")
            video_analyzer = None
    else:
        logger.warning("VideoAnalyzer will not be initialized as cv2 is unavailable.")
        pass # st.error for cv2 unavailability is already handled at import time

    audio_analyzer = None
    if pyaudio_available:
        logger.info("Attempting to initialize AudioAnalyzer...")
        try:
            audio_analyzer = AudioAnalyzer(model_size=whisper_model, language=whisper_language)
            logger.info("AudioAnalyzer initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize AudioAnalyzer.")
            st.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать AudioAnalyzer: {e}. Аудиоанализ будет невозможен.")
            audio_analyzer = None
    else:
        logger.warning("AudioAnalyzer will not be initialized as pyaudio is unavailable.")
        pass # st.error for pyaudio unavailability is already handled at import time
        
    integrator = None
    logger.info("Attempting to initialize MultimodalIntegrator...")
    try:
        integrator = MultimodalIntegrator(video_analyzer, audio_analyzer)
        logger.info("MultimodalIntegrator initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize MultimodalIntegrator.")
        st.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать MultimodalIntegrator: {e}. Интеграция и анализ будут недоступны.")
        integrator = None
    
    question_manager = None
    logger.info("Attempting to initialize QuestionManager...")
    try:
        data_dir = Path("data")
        if not data_dir.exists():
            logger.info(f"Data directory {data_dir} not found, creating it.")
            data_dir.mkdir(parents=True)
        
        questions_path = data_dir / "questions.csv"
        if not questions_path.exists():
            logger.info(f"Questions file {questions_path} not found, creating a default one.")
            import pandas as pd # Local import is fine here
            default_questions = pd.DataFrame({
                'id': range(1, 4),
                'text': [
                    "Расскажите о вашем опыте работы в команде",
                    "Какие технологии и инструменты вы использовали в последнем проекте?",
                    "Как вы решаете конфликтные ситуации?"
                ],
                'category': ['общие', 'технические', 'личностные']
            })
            default_questions.to_csv(questions_path, index=False)
            logger.info(f"Default questions.csv created at {questions_path}.")
        
        question_manager = QuestionManager(str(questions_path))
        logger.info("QuestionManager initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize QuestionManager.")
        st.error(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать менеджер вопросов: {e}. Функциональность вопросов будет недоступна.")
        question_manager = None # Ensure it's None on failure
    
    logger.info("Finished initializing analyzers and question manager.")
    return video_analyzer, audio_analyzer, integrator, question_manager

# Функция запуска сессии
def start_session(video_analyzer, audio_analyzer, integrator):
    global stop_event, session_active
    logger.info("Start session called.")
    
    if stop_event.is_set():
        logger.info("Stop event was set, clearing it now for new session.")
        stop_event.clear()
    
    # Сброс базовых значений
    if video_analyzer:
        try:
            video_analyzer.reset_baseline()
            logger.info("VideoAnalyzer baseline reset.")
        except Exception as e:
            logger.exception("Error resetting VideoAnalyzer baseline.")
            st.warning("Не удалось сбросить состояние видеоанализатора.")
    if audio_analyzer:
        try:
            audio_analyzer.reset_baseline()
            logger.info("AudioAnalyzer baseline reset.")
        except Exception as e:
            logger.exception("Error resetting AudioAnalyzer baseline.")
            st.warning("Не удалось сбросить состояние аудиоанализатора.")
    if integrator:
        try:
            integrator.clear_history()
            logger.info("MultimodalIntegrator history cleared.")
        except Exception as e:
            logger.exception("Error clearing MultimodalIntegrator history.")
            st.warning("Не удалось сбросить историю анализатора.")
    
    # Запуск потоков захвата и анализа
    if cv2_available and video_analyzer:
        logger.info("Starting video_capture_thread.")
        threading.Thread(target=video_capture_thread, args=(video_analyzer, integrator), daemon=True).start()
    else:
        logger.warning("Video capture thread will not be started as video analysis is unavailable.")
        # st.warning already handled by init and main UI
        
    if pyaudio_available and audio_analyzer:
        logger.info("Starting audio_capture_thread.")
        threading.Thread(target=audio_capture_thread, args=(audio_analyzer, integrator), daemon=True).start()
    else:
        logger.warning("Audio capture thread will not be started as audio analysis is unavailable.")
        # st.warning already handled by init and main UI
        
    if integrator: 
        logger.info("Starting multimodal_analysis_thread.")
        threading.Thread(target=multimodal_analysis_thread, args=(integrator,), daemon=True).start()
    else:
        logger.warning("Multimodal analysis thread will not be started as integrator is unavailable.")

    session_active = True
    st.session_state.session_active = True
    logger.info("Session marked as active.")

# Функция остановки сессии
def stop_session():
    global stop_event, session_active
    logger.info("Stop session called.")
    stop_event.set()
    session_active = False
    st.session_state.session_active = False
    logger.info("Session marked as inactive and stop_event set.")

# Основной интерфейс Streamlit
def main():
    logger.info("Streamlit application main() function started.")
    st.set_page_config(
        page_title="Мультимодальный анализ ответов",
        page_icon="📊",
        layout="wide"
    )
    logger.debug("Streamlit page configured.")
    
    st.title("Мультимодальный анализ ответов")
    logger.debug("Page title set.")
    
    # Инициализация анализаторов
    video_analyzer, audio_analyzer, integrator, question_manager = init_analyzers()
    logger.info("Analyzers and managers initialization process completed in main().")
    
    # Боковая панель с управлением
    with st.sidebar:
        st.header("Управление")
        
        # Кнопки управления сессией
        # Определяем, можно ли вообще начать сессию
        # Добавим проверку на существование integrator
        can_start_session = ((video_analyzer is not None and cv2_available) or \
                            (audio_analyzer is not None and pyaudio_available)) and \
                            integrator is not None

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Начать сессию", key="start_session",
                        disabled=st.session_state.get('session_active', False) or not can_start_session):
                if can_start_session:
                    start_session(video_analyzer, audio_analyzer, integrator)
                else:
                    # Более подробное сообщение об ошибке
                    error_messages = []
                    if not cv2_available:
                        error_messages.append("OpenCV (cv2) не доступен.")
                    logger.debug("Start session condition: OpenCV not available.")
                    elif video_analyzer is None:
                        error_messages.append("VideoAnalyzer не инициализирован.")
                        logger.debug("Start session condition: VideoAnalyzer is None.")
                    if not pyaudio_available:
                        error_messages.append("PyAudio не доступен.")
                        logger.debug("Start session condition: PyAudio not available.")
                    elif audio_analyzer is None:
                        error_messages.append("AudioAnalyzer не инициализирован.")
                        logger.debug("Start session condition: AudioAnalyzer is None.")
                    if integrator is None:
                        error_messages.append("MultimodalIntegrator не инициализирован.")
                        logger.debug("Start session condition: MultimodalIntegrator is None.")
                    
                    err_msg_for_user = f"Невозможно начать сессию: {', '.join(error_messages)}"
                    logger.error(f"Cannot start session: {err_msg_for_user}")
                    st.error(err_msg_for_user)
        
        with col2:
            if st.button("Завершить сессию", key="stop_session",
                        disabled=not st.session_state.get('session_active', False)):
                logger.info("Stop session button clicked.")
                stop_session()

        if not can_start_session:
             logger.warning("Session start is not possible due to component initialization issues or missing dependencies. UI warning displayed.")
             st.warning("Функциональность ограничена из-за проблем с инициализацией компонентов или отсутствием зависимостей.")
        
        # Управление вопросами должно работать, даже если анализаторы не загрузились
        if question_manager is not None:
            logger.debug("Question manager available, setting up question controls.")
            if st.button("Следующий вопрос", key="next_question",
                        disabled=not st.session_state.get('session_active', False)): 
                logger.info("Next question button clicked.")
                question = question_manager.next_question()
                if question is None:
                    logger.info("No more questions available. Stopping session.")
                    st.warning("Вопросы закончились. Сессия завершена.")
                    stop_session() 
                else:
                    logger.info(f"Displaying new question ID: {question.get('id', 'N/A')}")
            
            st.header("Текущий вопрос")
            current_question = question_manager.get_current_question()
            if current_question is not None:
                logger.debug(f"Displaying question: {current_question['text']}")
                st.markdown(f"**{current_question['text']}**")
                current, total, percent = question_manager.get_progress()
                st.progress(percent)
                st.text(f"Вопрос {current} из {total}")
            elif st.session_state.get('session_active', False): 
                 st.info("Нажмите 'Следующий вопрос', чтобы начать или если вопрос не отображается.")
            else: 
                 st.info("Начните сессию, чтобы увидеть вопросы.")
        else:
            logger.error("Question manager is None. Question functionality unavailable in UI.")
            st.error("Менеджер вопросов не инициализирован. Функциональность вопросов недоступна.")
    logger.debug("Sidebar UI setup complete.")
    
    # Основная область с анализом
    col1, col2 = st.columns([2, 1])
    
    video_placeholder = None # Инициализируем до использования
    with col1:
        st.header("Видео анализ")
        if cv2_available and video_analyzer:
            video_placeholder = st.empty()
        elif cv2_available and video_analyzer is None: # cv2 есть, но анализатор не создался
            st.error("VideoAnalyzer не смог инициализироваться. Видеоанализ недоступен.")
        elif not cv2_available: # cv2 не импортирован
            st.warning("Видеоанализ недоступен: OpenCV (cv2) не найден.")
        else: # Общий случай
            st.warning("Видеоанализ недоступен.")
            
    speech_placeholder = None # Инициализируем
    indicators_placeholder = None # Инициализируем
    with col2:
        st.header("Результаты анализа")
        
        st.subheader("Распознанная речь")
        if pyaudio_available and audio_analyzer:
            speech_placeholder = st.empty()
        elif pyaudio_available and audio_analyzer is None:
            st.error("AudioAnalyzer не смог инициализироваться. Распознавание речи недоступно.")
        elif not pyaudio_available:
            st.warning("Распознавание речи недоступно: PyAudio не найден.")
        else:
            st.warning("Распознавание речи недоступно.")

        st.subheader("Индикаторы")
        # Индикаторы могут частично работать, если интегратор есть, даже без полного аудио/видео
        if integrator:
            indicators_placeholder = st.empty()
            if video_analyzer is None and cv2_available:
                 st.warning("Часть индикаторов (связанных с видео) будет недоступна из-за ошибки VideoAnalyzer.")
            if audio_analyzer is None and pyaudio_available:
                 st.warning("Часть индикаторов (связанных с аудио) будет недоступна из-за ошибки AudioAnalyzer.")
        else:
            st.error("Индикаторы недоступны: MultimodalIntegrator не инициализирован.")
            
    # Раздел с графиками
    st.header("Визуализация данных")
    chart_col1, chart_col2 = st.columns(2)
    
    voice_chart_placeholder = None # Инициализируем
    with chart_col1:
        st.subheader("Характеристики голоса")
        if pyaudio_available and audio_analyzer:
            voice_chart_placeholder = st.empty()
        elif pyaudio_available and audio_analyzer is None:
            st.error("AudioAnalyzer не смог инициализироваться. График характеристик голоса недоступен.")
        elif not pyaudio_available:
            st.warning("График характеристик голоса недоступен: PyAudio не найден.")
        else:
            st.warning("График характеристик голоса недоступен.")

    timeline_chart_placeholder = None # Инициализируем
    with chart_col2:
        st.subheader("Временная шкала индикаторов")
        if integrator:
            timeline_chart_placeholder = st.empty()
        else:
            st.error("Временная шкала недоступна: MultimodalIntegrator не инициализирован.")
            
    # Отображение сводки
    summary_placeholder = None # Инициализируем
    st.header("Сводный анализ")
    if integrator:
        summary_placeholder = st.empty()
    else:
        st.error("Сводный анализ недоступен: MultimodalIntegrator не инициализирован.")
        
    # Запускаем обновление UI
    # Убедимся, что update_ui вызывается, даже если компоненты None,
    # так как она содержит логику отображения сообщений об ошибках в некоторых случаях
    # или может выполнять другие фоновые задачи Streamlit.
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
    """Функция обновления UI. Должна быть устойчива к None компонентам."""
    
    pitch_values = []
    intensity_values = []
    timestamps = []
    last_speech_text = ""
    summary_counter = 0  # Счетчик для создания уникальных ключей
    
    while True:
        try:
            # Выход из цикла, если сессия не активна
            if not st.session_state.get('session_active', False):
                time.sleep(0.1) # Небольшая задержка, чтобы не загружать CPU в неактивном состоянии
                continue # Пропускаем остальную часть цикла, если сессия не активна
                
            # Обновление видео
            if cv2_available and video_analyzer and video_placeholder and not frame_queue.empty():
                try:
                    frame = frame_queue.get_nowait() # Используем get_nowait для неблокирующего чтения
                    video_placeholder.image(frame, channels="BGR", use_column_width=True)
                except queue.Empty:
                    pass # Очередь пуста, ничего страшного
            
            audio_results = None 
            if pyaudio_available and audio_analyzer:
                try:
                    # Предполагаем, что get_latest_results() не блокирующий или быстро возвращает
                    audio_results = audio_analyzer.get_latest_results() 
                    if audio_results and 'text' in audio_results and audio_results['text']:
                        last_speech_text = audio_results['text']
                        if speech_placeholder:
                            speech_placeholder.markdown(f"*{last_speech_text}*")
                        
                        if 'pitch' in audio_results and 'intensity' in audio_results:
                            pitch_values.append(audio_results['pitch'])
                            intensity_values.append(audio_results['intensity'])
                            timestamps.append(time.time())
                            max_points = 50
                            if len(pitch_values) > max_points:
                                pitch_values = pitch_values[-max_points:]
                                intensity_values = intensity_values[-max_points:]
                                timestamps = timestamps[-max_points:]
                except Exception as e:
                    # st.warning(f"Ошибка при получении аудио результатов: {e}") # Может быть слишком много сообщений
                    pass # Просто пропускаем, если есть проблема

            distortions = {}
            if cv2_available and video_analyzer:
                try:
                    # Предполагаем, что get_distortions() не блокирующий
                    distortions = video_analyzer.get_distortions()
                except Exception as e:
                    # st.warning(f"Ошибка при получении видео искажений: {e}")
                    pass

            if indicators_placeholder: 
                indicators_text = []
                if cv2_available and video_analyzer and distortions:
                    for region, is_distorted in distortions.items():
                        status = "🔴" if is_distorted else "🟢"
                        indicators_text.append(f"{status} {region}")
                
                if pyaudio_available and audio_analyzer and audio_results and 'voice_change' in audio_results:
                    voice_status = "🔴" if audio_results['voice_change'] else "🟢"
                    indicators_text.append(f"{voice_status} голос")
                
                if not indicators_text and (video_analyzer or audio_analyzer): # Если есть хотя бы один анализатор, но нет данных
                    indicators_text.append("Данные для индикаторов пока отсутствуют или обрабатываются.")
                elif not video_analyzer and not audio_analyzer : # Если оба анализатора неактивны
                     indicators_text.append("Индикаторы недоступны (анализаторы не инициализированы).")


                indicators_placeholder.markdown("<br>".join(indicators_text), unsafe_allow_html=True)
            
            if pyaudio_available and audio_analyzer and voice_chart_placeholder and pitch_values and intensity_values:
                try:
                    voice_fig = plot_voice_characteristics(pitch_values, intensity_values)
                    if voice_fig:
                        voice_chart_placeholder.pyplot(voice_fig)
                except Exception as e:
                    # st.warning(f"Ошибка при построении графика голоса: {e}")
                    pass
            
            if integrator and timeline_chart_placeholder and summary_placeholder:
                try:
                    recent_results = integrator.get_recent_results()
                    if recent_results:
                        if timeline_chart_placeholder:
                            timeline_fig = plot_multimodal_timeline(recent_results)
                            if timeline_fig:
                                timeline_chart_placeholder.pyplot(timeline_fig)
                        
                        if summary_placeholder:
                            summary_text = create_distortion_summary(recent_results)
                            summary_counter += 1
                            summary_placeholder.text_area("Результаты анализа:", summary_text, height=200, key=f"summary_{summary_counter}")
                except Exception as e:
                    # st.warning(f"Ошибка при обновлении временной шкалы или сводки: {e}")
                    pass
            
            # Задержка для снижения нагрузки на CPU.
            # Важно, чтобы эта задержка была здесь, чтобы UI оставался отзывчивым.
            time.sleep(0.1)
            
        except Exception as e:
            logger.exception("Unhandled error in update_ui loop.") # Logs full traceback
            st.error(f"Критическая ошибка при обновлении интерфейса: {e}")
            time.sleep(1) # Prevent rapid error logging if in a tight loop

if __name__ == "__main__":
    logger.info("Application script execution started (__main__ block).")
    # Инициализация состояния сессии
    if 'session_active' not in st.session_state:
        logger.debug("Initializing 'session_active' in st.session_state to False.")
        st.session_state.session_active = False
    
    main()
    logger.info("Application script execution finished (__main__ block).")