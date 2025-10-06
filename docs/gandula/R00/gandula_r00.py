# Importa as bibliotecas necessarias
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO # Importa a biblioteca para controle da GPIO
import time

# --- OTIMIZACOES ---
PROCESSAR_A_CADA_N_FRAMES = 6 # So vai rodar a IA a cada 4 frames. Aumente para mais velocidade.

# --- 1. CONFIGURACAO INICIAL COMBINADA ---

# --- Configuracao dos Motores (GPIO) ---
MOTOR1_PINO = 17      # Pino para o motor esquerdo (movimentacao)
MOTOR2_PINO = 18      # Pino para o motor direito (movimentacao)
MOTOREIXO_PINO = 15   # Pino para o motor do eixo/captura

# --- Configuracao do PWM ---
FREQUENCIA_PWM = 100              # Frequencia em Hz para o PWM
DUTY_CYCLE_MOVIMENTO = 50         # Velocidade dos motores (0 a 100)
DUTY_CYCLE_MOVIMENTO_CURVA = 70

# Configura o modo da GPIO (BCM refere-se aos numeros dos pinos GPIO)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Configura os pinos dos motores
GPIO.setup(MOTOR1_PINO, GPIO.OUT)
GPIO.setup(MOTOR2_PINO, GPIO.OUT)
GPIO.setup(MOTOREIXO_PINO, GPIO.OUT)

# Cria objetos PWM para os motores de movimentacao
pwm_motor1 = GPIO.PWM(MOTOR1_PINO, FREQUENCIA_PWM)
pwm_motor2 = GPIO.PWM(MOTOR2_PINO, FREQUENCIA_PWM)

# Inicia o PWM com duty cycle 0 (motores parados)
pwm_motor1.start(0)
pwm_motor2.start(0)

# Garante que o motor do eixo comece desligado
GPIO.output(MOTOREIXO_PINO, GPIO.LOW)

# --- Funcoes de Controle dos Motores (com PWM) ---
def avancar():
    """Ativa ambos os motores com o duty cycle definido e ativa o eixo."""
    pwm_motor1.ChangeDutyCycle(DUTY_CYCLE_MOVIMENTO)
    pwm_motor2.ChangeDutyCycle(DUTY_CYCLE_MOVIMENTO)
    GPIO.output(MOTOREIXO_PINO, GPIO.HIGH)
    print(f"Motores: AVANCAR a {DUTY_CYCLE_MOVIMENTO}% e CAPTURAR")

def virar_esquerda():
    """Ativa o motor direito para virar a esquerda."""
    pwm_motor1.ChangeDutyCycle(0)
    pwm_motor2.ChangeDutyCycle(DUTY_CYCLE_MOVIMENTO_CURVA)
    time.sleep(0.1)
    pwm_motor2.ChangeDutyCycle(0)
    time.sleep(0.1)
    print(f"Motores: VIRAR A ESQUERDA a {DUTY_CYCLE_MOVIMENTO_CURVA}%")

def virar_direita():
    """Ativa o motor esquerdo para virar a direita."""
    pwm_motor1.ChangeDutyCycle(DUTY_CYCLE_MOVIMENTO_CURVA)
    pwm_motor2.ChangeDutyCycle(0)
    time.sleep(0.1)
    pwm_motor1.ChangeDutyCycle(0)
    time.sleep(0.1)
    print(f"Motores: VIRAR A DIREITA a {DUTY_CYCLE_MOVIMENTO_CURVA}%")

def parar():
    """Para todos os motores."""
    pwm_motor1.ChangeDutyCycle(0)
    pwm_motor2.ChangeDutyCycle(0)
    GPIO.output(MOTOREIXO_PINO, GPIO.LOW)
    print("Motores: PARAR")
    
def controlar_movimento_pelo_centro(posicao_x_bola, centro_tela_x, largura_tela):
    """
    Identifica se a bolinha esta no meio e aciona a funcao de motor correspondente.
    Retorna o texto da acao.
    """
    tolerancia = largura_tela * 0.1 # Tolerancia de 10% da largura da tela
    
    if posicao_x_bola < (centro_tela_x - tolerancia):
        virar_esquerda()
        return "ACAO: Virar a Esquerda"
    elif posicao_x_bola > (centro_tela_x + tolerancia):
        virar_direita()
        return "ACAO: Virar a Direita"
    else:
        # A bolinha esta no centro! Aciona os 2 motores para avancar.
        avancar()
        return "ACAO: Alinhado / Avancar"

def cleanup_gpio():
    """Funcao final para parar os motores e limpar a GPIO."""
    parar()
    pwm_motor1.stop()
    pwm_motor2.stop()
    GPIO.cleanup()
    print("Recursos da GPIO liberados.")


# --- Configuracao do Machine Learning ---
MODEL_PATH = "model.tflite"
LABEL_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.50

# Carrega o modelo TFLite e aloca os tensores.
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Obtem os detalhes dos tensores de entrada e saida do modelo ML.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
ml_height = input_details[0]['shape'][1]
ml_width = input_details[0]['shape'][2]

# Carrega os labels (nomes das classes) do modelo ML.
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

# --- Configuracao do OpenCV para Localizacao ---
laranja_inferior = np.array([5, 105, 105])
laranja_superior = np.array([15, 255, 255])

# --- INICIALIZA CAMERA ---
cap = cv2.VideoCapture(0)

# --- VARIAVEIS PARA O LOOP OTIMIZADO ---
contador_frames = 0
ml_result_text = "Iniciando..."
direcao_texto = "Aguardando..."

# --- 2. LOOP PRINCIPAL COM LOGICA HIBRIDA ---
try:
    while True:
        ret, frame_original = cap.read()
        if not ret:
            print("Erro ao capturar imagem da camera.")
            break

        frame = cv2.resize(frame_original, (600, 400))
        altura_tela, largura_tela, _ = frame.shape
        Y_THRESHOLD_AVANCO = altura_tela * 0.85

        contador_frames += 1
        if contador_frames % PROCESSAR_A_CADA_N_FRAMES == 0:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (ml_width, ml_height))
            input_data = np.expand_dims(image_resized, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            if output_data.dtype == np.uint8:
                output_data = (output_data.astype(np.float32) / 255.0)
            
            class_id = int(np.argmax(output_data))
            score = float(output_data[class_id])
            object_name = labels[class_id]

            ml_result_text = f"ML: {object_name} ({score*100:.1f}%)"
            
            if object_name == 'bola' and score > CONFIDENCE_THRESHOLD:
                blurred = cv2.GaussianBlur(frame, (11, 11), 0)
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                mascara = cv2.inRange(hsv, laranja_inferior, laranja_superior)
                mascara = cv2.erode(mascara, None, iterations=2)
                mascara = cv2.dilate(mascara, None, iterations=2)
                cv2.imshow("Mascara Laranja (OpenCV)", mascara)
                contornos, _ = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contornos) > 0:
                    c = max(contornos, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)

                    if radius > 10:
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                        centro_bola = (int(x), int(y))
                        
                        if centro_bola[1] > Y_THRESHOLD_AVANCO:
                            direcao_texto = "ACAO: Captura final!"
                            avancar()
                            time.sleep(1)
                            parar()
                            time.sleep(1)
                        else:
                            # --- LOGICA DE MOVIMENTO SIMPLIFICADA ---
                            centro_tela_x = largura_tela // 2
                            direcao_texto = controlar_movimento_pelo_centro(centro_bola[0], centro_tela_x, largura_tela)
                    else:
                        direcao_texto = "Bola muito longe"
                        parar()
                else:
                    direcao_texto = "ML ve, OpenCV nao localizou"
                    parar()
            else:
                direcao_texto = "Nenhuma bola validada pelo ML"
                parar()
        
        cv2.putText(frame, ml_result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, direcao_texto, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Robo Gandula - Hibrido (ML + OpenCV)", frame)

        if cv2.waitKey(1) == ord("q"):
            break
finally:
    # --- FINALIZACAO SEGURA ---
    print("\nFinalizando o programa...")
    cap.release()
    cv2.destroyAllWindows()
    cleanup_gpio()
